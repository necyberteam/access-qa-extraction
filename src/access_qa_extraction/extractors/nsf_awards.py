"""Extractor for nsf-awards MCP server.

Uses an LLM to generate Q&A pairs based on actual data returned from MCP tools.
The LLM analyzes each NSF award and generates contextually appropriate questions
based on what information is actually available.
"""

import hashlib
import json
import re

from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from .base import BaseExtractor, ExtractionOutput


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


SYSTEM_PROMPT = (
    "You are a Q&A pair generator for NSF awards related to ACCESS-CI. "
    "Your task is to generate high-quality question-answer pairs based on "
    "the provided NSF award data.\n\n"
    "Guidelines:\n"
    "1. Only generate questions that can be accurately answered from the "
    "provided data\n"
    "2. Do not make up or infer information that isn't explicitly in the data\n"
    "3. Generate a variety of question types (what is the award about, who is "
    "the PI, what institution, funding amount, program, dates, etc.)\n"
    "4. Questions should be natural - the kind a researcher or student might "
    "actually ask\n"
    "5. Answers should be informative but concise\n"
    "6. Each answer must end with the citation marker provided\n\n"
    "Output format: Return a JSON array of objects with \"question\" and "
    "\"answer\" fields.\n"
    "Example:\n"
    "[\n"
    '  {"question": "What is NSF award 2345678?", '
    '"answer": "NSF award 2345678 is titled \'Advanced Computing for '
    "Climate Research' and was awarded to Dr. Jane Smith at MIT. The total "
    "intended award is $1,500,000 and it runs from 2024-01-01 to "
    '2027-12-31.\\n\\n<<SRC:nsf-awards:2345678>>"},\n'
    '  {"question": "Who is the PI on NSF award 2345678?", '
    '"answer": "The principal investigator on NSF award 2345678 is '
    'Dr. Jane Smith from MIT.\\n\\n<<SRC:nsf-awards:2345678>>"}\n'
    "]"
)


USER_PROMPT_TEMPLATE = (
    "Generate Q&A pairs for this NSF award.\n\n"
    "Award Number: {award_number}\n"
    "Citation marker to use: <<SRC:nsf-awards:{award_number}>>\n\n"
    "Award Data:\n"
    "{award_json}\n\n"
    "Generate appropriate Q&A pairs based on what information is "
    "actually available. Return only the JSON array."
)


# The nsf-awards server requires at least one search parameter (no list-all fallback).
# These queries are organized by dimension to maximize coverage across the dataset,
# with deduplication by award number to avoid processing duplicates.
NSF_AWARD_QUERIES = [
    # NSF programs
    {"query": "cyberinfrastructure", "limit": 3},
    # {"query": "OAC", "limit": 50},
    # # Disciplines
    # {"query": "high performance computing", "limit": 50},
    # {"query": "artificial intelligence", "limit": 50},
    # {"query": "machine learning", "limit": 50},
    # {"query": "bioinformatics", "limit": 50},
    # {"query": "climate", "limit": 50},
    # {"query": "materials science", "limit": 50},
    # {"query": "quantum computing", "limit": 50},
    # {"query": "genomics", "limit": 50},
    # {"query": "molecular dynamics", "limit": 50},
    # {"query": "physics", "limit": 50},
    # {"query": "astronomy", "limit": 50},
    # {"query": "chemistry", "limit": 50},
    # # Institutions (major HPC users)
    # {"query": "university of illinois", "limit": 50},
    # {"query": "stanford", "limit": 50},
    # {"query": "MIT", "limit": 50},
    # {"query": "carnegie mellon", "limit": 50},
    # {"query": "georgia tech", "limit": 50},
    # {"query": "university of texas", "limit": 50},
    # {"query": "university of california", "limit": 50},
    # {"query": "purdue", "limit": 50},
    # {"query": "cornell", "limit": 50},
    # # General
    # {"query": "research", "limit": 50},
    # {"query": "software", "limit": 50},
    # {"query": "workforce", "limit": 50},
    # {"query": "education", "limit": 50},
    # {"query": "training", "limit": 50},
]


class NSFAwardsExtractor(BaseExtractor):
    """Extract Q&A pairs from nsf-awards server using LLM."""

    server_name = "nsf-awards"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all NSF awards."""
        pairs: ExtractionResult = []
        raw_data: dict = {}
        seen_ids: set[str] = set()

        awards = []
        for i, params in enumerate(NSF_AWARD_QUERIES):
            result = await self.client.call_tool("search_nsf_awards", params)
            items = result.get("items", result.get("awards", []))
            new_count = sum(
                1 for a in items
                if str(a.get("awardNumber", "")) not in seen_ids
            )
            print(
                f"  [{i + 1}/{len(NSF_AWARD_QUERIES)}] "
                f"'{params['query']}' → {len(items)} results, {new_count} new"
            )
            awards.extend(items)

        for award in awards:
            award_number = str(award.get("awardNumber", ""))
            title = award.get("title", "")
            if not award_number or not title:
                continue

            if award_number in seen_ids:
                continue
            seen_ids.add(award_number)

            clean_award = self._clean_award_data(award)
            source_data = {"award": clean_award}

            award_pairs = await self._generate_qa_pairs(
                award_number, clean_award, source_data
            )
            pairs.extend(award_pairs)

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
        self, award_number: str, award: dict, source_data: dict
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from award data."""
        pairs: ExtractionResult = []

        award_json = json.dumps(award, indent=2)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            award_number=award_number,
            award_json=award_json,
        )

        try:
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=2048,
            )

            response_text = response.text

            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                for qa in qa_list:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if question and answer:
                        q_slug = re.sub(
                            r"[^a-z0-9]+", "_", question.lower()
                        )[:30]
                        q_hash = hashlib.md5(question.encode()).hexdigest()[:6]
                        pair_id = f"nsf_{award_number}_{q_slug}_{q_hash}"

                        complexity = "simple"
                        if any(
                            term in question.lower()
                            for term in [
                                "how to",
                                "steps",
                                "process",
                                "compare",
                            ]
                        ):
                            complexity = "moderate"

                        pairs.append(
                            QAPair.create(
                                id=pair_id,
                                question=question,
                                answer=answer,
                                source_ref=(
                                    f"mcp://nsf-awards/awards/{award_number}"
                                ),
                                domain="nsf-awards",
                                complexity=complexity,
                                source_data=source_data,
                            )
                        )

        except Exception as e:
            print(
                f"Error generating Q&A for NSF award {award_number}: {e}"
            )

        return pairs
