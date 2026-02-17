"""Extractor for nsf-awards — direct NSF API pagination.

Fetches awards by paginating the public NSF API at
api.nsf.gov/services/v1/awards.json with offset/rpp params, transforms
the raw fields to match the MCP-normalized format, then uses fixed
question categories to generate Q&A pairs via LLM.
"""

import json
import re

import httpx

from ..generators.factoids import generate_factoid_pairs
from ..generators.incremental import compute_entity_hash
from ..generators.judge import evaluate_pairs
from ..llm_client import BaseLLMClient, get_judge_client, get_llm_client
from ..models import ExtractionResult, QAPair
from ..question_categories import build_system_prompt, build_user_prompt, generate_bonus_pairs
from .base import BaseExtractor, ExtractionOutput, ExtractionReport

NSF_API_URL = "https://api.nsf.gov/services/v1/awards.json"

# Fields to request from the NSF API (same set the MCP server uses)
NSF_PRINT_FIELDS = ",".join(
    [
        "id",
        "title",
        "abstractText",
        "piFirstName",
        "piLastName",
        "coPDPI",
        "poName",
        "awardeeName",
        "fundsObligatedAmt",
        "estimatedTotalAmt",
        "startDate",
        "expDate",
        "primaryProgram",
        "fundProgramName",
        "ueiNumber",
    ]
)

NSF_PAGE_SIZE = 100  # Max allowed by NSF API


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _format_currency(amount_str: str) -> str:
    """Format a numeric string as USD currency (e.g., '1234567' → '$1,234,567')."""
    if not amount_str:
        return ""
    try:
        amount = int(amount_str)
        return f"${amount:,}"
    except (ValueError, TypeError):
        return amount_str


def _transform_nsf_award(raw: dict) -> dict:
    """Transform a raw NSF API award record to the MCP-normalized format.

    The NSF API returns field names like piFirstName, abstractText, expDate.
    The extractors and tests expect the MCP-normalized format with fields like
    principalInvestigator, abstract, endDate.
    """
    pi_first = (raw.get("piFirstName") or "").strip()
    pi_last = (raw.get("piLastName") or "").strip()
    pi = f"{pi_first} {pi_last}".strip()

    co_pdpi = raw.get("coPDPI") or ""
    if isinstance(co_pdpi, list):
        co_pis = [name.strip() for name in co_pdpi if name.strip()]
    elif co_pdpi:
        co_pis = [name.strip() for name in co_pdpi.split(";") if name.strip()]
    else:
        co_pis = []

    raw_program = raw.get("primaryProgram") or raw.get("fundProgramName") or ""
    if isinstance(raw_program, list):
        primary_program = "; ".join(raw_program)
    else:
        primary_program = raw_program

    return {
        "awardNumber": raw.get("id") or "",
        "title": raw.get("title") or "",
        "institution": raw.get("awardeeName") or "",
        "principalInvestigator": pi,
        "coPIs": co_pis,
        "totalIntendedAward": _format_currency(raw.get("estimatedTotalAmt") or ""),
        "totalAwardedToDate": _format_currency(raw.get("fundsObligatedAmt") or ""),
        "startDate": raw.get("startDate") or "",
        "endDate": raw.get("expDate") or "",
        "abstract": raw.get("abstractText") or "",
        "primaryProgram": primary_program,
        "programOfficer": raw.get("poName") or "",
        "ueiNumber": raw.get("ueiNumber") or "",
    }


class NSFAwardsExtractor(BaseExtractor):
    """Extract Q&A pairs from NSF awards using direct API pagination."""

    server_name = "nsf-awards"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()
        self.judge_client = None
        if not self.extraction_config.no_judge:
            try:
                self.judge_client = get_judge_client()
            except (ValueError, ImportError):
                pass

    async def run(self) -> ExtractionOutput:
        """Run extraction — no MCPClient needed (uses direct API)."""
        # Overrides BaseExtractor.run() which creates an MCPClient context.
        # This extractor fetches from api.nsf.gov directly.
        return await self.extract()

    async def run_report(self) -> ExtractionReport:
        """Run report — no MCPClient needed (uses direct API)."""
        return await self.report()

    async def report(self) -> ExtractionReport:
        """Fetch first page to get a sample and estimate total."""
        params = self._build_query_params()
        params["offset"] = 1
        params["rpp"] = NSF_PAGE_SIZE

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(NSF_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        awards_wrapper = data.get("response", {})
        raw_awards = awards_wrapper.get("award", [])
        transformed = [_transform_nsf_award(a) for a in raw_awards]

        ids = [a["awardNumber"] for a in transformed if a["awardNumber"]]

        return ExtractionReport(
            server_name=self.server_name,
            strategy="list-all",
            queries_used=[],
            total_fetched=len(transformed),
            unique_entities=len(ids),
            sample_ids=ids[:5],
        )

    def _build_query_params(self) -> dict:
        """Build base query params for the NSF API.

        Currently returns just the printFields specification. This is the
        extension point for future pre-filtering.

        # TODO: Add pre-filtering support (keywords, date ranges, programs)
        # The NSF API supports these filter params:
        #   keyword       — full-text search across title + abstract
        #   startDateStart, startDateEnd — filter by award start date (MM/DD/YYYY)
        #   expDateStart, expDateEnd     — filter by expiration date
        #   fundProgramName              — filter by funding program name
        #   awardeeName                  — filter by institution name
        #   piFirstName, piLastName      — filter by PI name
        #   coPDPI                       — filter by co-PI name
        #   primaryProgram               — filter by primary program
        #   poName                       — filter by program officer
        # Example: to restrict to recent awards:
        #   params["startDateStart"] = "01/01/2020"
        #   params["keyword"] = "cyberinfrastructure"
        """
        return {"printFields": NSF_PRINT_FIELDS}

    async def _fetch_all_awards(self) -> list[dict]:
        """Paginate the NSF API and return all awards (MCP-normalized format).

        Uses offset-based pagination (1-indexed, rpp=100 max).
        Stops when fewer than 100 results returned or --max-entities cap hit.
        """
        all_awards: list[dict] = []
        max_entities = self.extraction_config.max_entities
        offset = 1

        async with httpx.AsyncClient(timeout=30.0) as http:
            while True:
                params = self._build_query_params()
                params["offset"] = offset
                params["rpp"] = NSF_PAGE_SIZE

                resp = await http.get(NSF_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                awards_wrapper = data.get("response", {})
                raw_awards = awards_wrapper.get("award", [])

                if not raw_awards:
                    break

                transformed = [_transform_nsf_award(a) for a in raw_awards]
                all_awards.extend(transformed)

                page_num = (offset - 1) // NSF_PAGE_SIZE + 1
                if page_num == 1 or page_num % 50 == 0:
                    print(f"  NSF API page {page_num}: {len(all_awards)} total awards")

                if max_entities and len(all_awards) >= max_entities:
                    return all_awards[:max_entities]

                if len(raw_awards) < NSF_PAGE_SIZE:
                    break  # Last page

                offset += NSF_PAGE_SIZE

        print(f"  NSF API: {len(all_awards)} total awards fetched")
        return all_awards

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all NSF awards."""
        pairs: ExtractionResult = []
        raw_data: dict = {}

        awards = await self._fetch_all_awards()
        print(f"  Fetched {len(awards)} awards, generating Q&A pairs...")

        system_prompt = build_system_prompt("nsf-awards")

        entity_count = 0
        seen_ids: set[str] = set()

        for award in awards:
            award_number = str(award.get("awardNumber", ""))
            title = award.get("title", "")
            if not award_number or not title:
                continue

            if award_number in seen_ids:
                continue
            seen_ids.add(award_number)

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            clean_award = self._clean_award_data(award)

            # Incremental: skip LLM + factoid if entity data unchanged
            entity_hash = compute_entity_hash(clean_award)
            used_cache = False
            if self.incremental_cache:
                if self.incremental_cache.is_unchanged(
                    "nsf-awards", award_number, entity_hash
                ):
                    cached_pairs = self.incremental_cache.get_cached_pairs(
                        "nsf-awards", award_number
                    )
                    if cached_pairs:
                        pairs.extend(cached_pairs)
                        used_cache = True

            if not used_cache:
                award_pairs = await self._generate_qa_pairs(
                    award_number, clean_award, system_prompt
                )
                pairs.extend(award_pairs)

                # Generate factoid Q&A pairs from templates (zero LLM)
                factoid_data = {**clean_award, "award_number": award_number}
                factoid_pairs = generate_factoid_pairs(
                    "nsf-awards", award_number, factoid_data
                )
                pairs.extend(factoid_pairs)

                # Generate bonus (exploratory) Q&A pairs from rich text
                bonus_pairs = []
                if not self.extraction_config.no_bonus:
                    bonus_pairs = generate_bonus_pairs(
                        "nsf-awards", award_number, clean_award,
                        self.llm, self.extraction_config.max_tokens,
                    )
                    pairs.extend(bonus_pairs)

                # Judge evaluation: score all pairs for this entity
                all_entity_pairs = award_pairs + factoid_pairs + bonus_pairs
                if self.judge_client:
                    evaluate_pairs(
                        all_entity_pairs, {"award": clean_award}, self.judge_client
                    )

                if self.incremental_cache:
                    self.incremental_cache.store(
                        "nsf-awards",
                        award_number,
                        entity_hash,
                        all_entity_pairs,
                    )

            raw_data[award_number] = {
                "name": title,
                "award_number": award_number,
                "pi": award.get("principalInvestigator", ""),
                "institution": award.get("institution", ""),
                "total_award": award.get("totalIntendedAward", ""),
                "primary_program": award.get("primaryProgram", ""),
                "has_co_pis": bool(award.get("coPIs")),
            }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    def _clean_award_data(self, award: dict) -> dict:
        """Clean award data for LLM consumption."""
        cleaned = {
            "title": award.get("title", ""),
            "principal_investigator": award.get("principalInvestigator", ""),
            "institution": award.get("institution", ""),
            "total_intended_award": award.get("totalIntendedAward", ""),
            "primary_program": award.get("primaryProgram", ""),
        }

        for field in [
            "abstract",
            "startDate",
            "endDate",
            "totalAwardedToDate",
            "programOfficer",
        ]:
            value = award.get(field, "")
            if value:
                if field == "abstract":
                    cleaned[field] = strip_html(value)
                else:
                    cleaned[field] = value

        co_pis = award.get("coPIs", [])
        if co_pis:
            cleaned["co_pis"] = co_pis

        return cleaned

    async def _generate_qa_pairs(
        self, award_number: str, award: dict, system_prompt: str
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from award data."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(award, indent=2)
        user_prompt = build_user_prompt("nsf-awards", award_number, entity_json)

        try:
            response = self.llm.generate(
                system=system_prompt,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )

            response_text = response.text
            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                for qa in qa_list:
                    category = qa.get("category", "")
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if category and question and answer:
                        pair_id = f"nsf-awards_{award_number}_{category}"

                        complexity = "simple"
                        if any(
                            term in question.lower()
                            for term in ["how to", "steps", "process", "compare"]
                        ):
                            complexity = "moderate"

                        pairs.append(
                            QAPair.create(
                                id=pair_id,
                                question=question,
                                answer=answer,
                                source_ref=f"mcp://nsf-awards/awards/{award_number}",
                                domain="nsf-awards",
                                complexity=complexity,
                                source_data={"award": award},
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for NSF award {award_number}: {e}")

        return pairs
