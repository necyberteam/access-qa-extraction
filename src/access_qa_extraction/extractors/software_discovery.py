"""Extractor for software-discovery MCP server.

Uses an LLM with fixed question categories to generate Q&A pairs based on
software catalog data. Fetches all software via list_all_software MCP tool
(list-all strategy).
"""

import json
import re

from ..generators.factoids import generate_factoid_pairs
from ..generators.incremental import compute_entity_hash
from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from ..question_categories import build_system_prompt, build_user_prompt
from .base import BaseExtractor, ExtractionOutput, ExtractionReport


class SoftwareDiscoveryExtractor(BaseExtractor):
    """Extract Q&A pairs from software-discovery server using LLM."""

    server_name = "software-discovery"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def report(self) -> ExtractionReport:
        """Fetch all software from MCP and return coverage stats."""
        result = await self.client.call_tool("list_all_software", {"limit": 2000})
        items = result.get("items", result.get("software", []))

        seen: set[str] = set()
        for s in items:
            name = s.get("name", "").lower()
            if name:
                seen.add(name)

        return ExtractionReport(
            server_name=self.server_name,
            strategy="list-all",
            queries_used=[],
            total_fetched=len(items),
            unique_entities=len(seen),
            sample_ids=sorted(seen)[:5],
        )

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for software packages."""
        pairs: ExtractionResult = []
        raw_data: dict = {}

        # Fetch all software via MCP list_all_software (uses wildcard ["*"] internally)
        limit = self.extraction_config.max_entities or 2000
        result = await self.client.call_tool(
            "list_all_software",
            {"limit": limit, "include_ai_metadata": True},
        )
        software_list = result.get("items", result.get("software", []))

        system_prompt = build_system_prompt("software-discovery")

        entity_count = 0
        seen_software: set[str] = set()

        for software in software_list:
            name = software.get("name", "").lower()
            if not name or name in seen_software:
                continue
            seen_software.add(name)

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            # Clean up data for LLM (also serves as source_data for review)
            clean_software = self._clean_software_data(software)

            # Incremental: skip LLM + factoid if entity data unchanged
            entity_hash = compute_entity_hash(clean_software)
            used_cache = False
            if self.incremental_cache:
                if self.incremental_cache.is_unchanged(
                    "software-discovery", name, entity_hash
                ):
                    cached_pairs = self.incremental_cache.get_cached_pairs(
                        "software-discovery", name
                    )
                    if cached_pairs:
                        pairs.extend(cached_pairs)
                        used_cache = True

            if not used_cache:
                # Generate Q&A pairs using LLM
                software_pairs = await self._generate_qa_pairs(
                    name, clean_software, system_prompt
                )
                pairs.extend(software_pairs)

                # Generate factoid Q&A pairs from templates (zero LLM)
                factoid_pairs = generate_factoid_pairs(
                    "software-discovery", name, clean_software
                )
                pairs.extend(factoid_pairs)

                if self.incremental_cache:
                    self.incremental_cache.store(
                        "software-discovery",
                        name,
                        entity_hash,
                        software_pairs + factoid_pairs,
                    )

            # Store normalized data for comparison generation
            raw_data[name] = {
                "name": software.get("name", name),
                "software_id": name,
                "resources": self._extract_resource_ids(software),
                "tags": clean_software.get("tags", []),
                "research_area": clean_software.get("research_area"),
                "software_type": clean_software.get("software_type"),
            }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    def _extract_resource_ids(self, software: dict) -> list[str]:
        """Extract resource IDs where software is available."""
        resources = software.get("available_on_resources", [])
        if isinstance(resources, list):
            resource_ids = []
            for r in resources:
                if isinstance(r, dict):
                    resource_ids.append(r.get("resource_id", r.get("name", "")))
                else:
                    resource_ids.append(str(r))
            return [r for r in resource_ids if r]
        return []

    def _clean_software_data(self, software: dict) -> dict:
        """Clean software data for LLM consumption."""
        cleaned = {}

        # Core fields
        if software.get("name"):
            cleaned["name"] = software["name"]
        if software.get("description"):
            cleaned["description"] = software["description"]
        if software.get("versions"):
            cleaned["versions"] = software["versions"]
        if software.get("available_on_resources"):
            cleaned["available_on_resources"] = software["available_on_resources"]
        if software.get("documentation"):
            cleaned["documentation"] = software["documentation"]
        if software.get("website"):
            cleaned["website"] = software["website"]

        # AI-enhanced metadata
        ai_meta = software.get("ai_metadata", {})
        if ai_meta:
            if ai_meta.get("tags"):
                cleaned["tags"] = ai_meta["tags"]
            if ai_meta.get("research_area"):
                cleaned["research_area"] = ai_meta["research_area"]
            if ai_meta.get("research_field"):
                cleaned["research_field"] = ai_meta["research_field"]
            if ai_meta.get("software_type"):
                cleaned["software_type"] = ai_meta["software_type"]
            if ai_meta.get("core_features"):
                cleaned["core_features"] = ai_meta["core_features"]
            if ai_meta.get("example_use"):
                example = ai_meta["example_use"]
                if len(example) > 1500:
                    example = example[:1500] + "..."
                cleaned["example_use"] = example

        return cleaned

    async def _generate_qa_pairs(
        self, software_name: str, software: dict, system_prompt: str
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from software data."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(software, indent=2)
        user_prompt = build_user_prompt("software-discovery", software_name, entity_json)

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
                        pair_id = f"software-discovery_{software_name}_{category}"

                        complexity = "simple"
                        if any(
                            term in question.lower()
                            for term in ["how do i", "how to", "example", "versions"]
                        ):
                            complexity = "moderate"

                        pairs.append(
                            QAPair.create(
                                id=pair_id,
                                question=question,
                                answer=answer,
                                source_ref=(f"mcp://software-discovery/software/{software_name}"),
                                domain="software-discovery",
                                complexity=complexity,
                                source_data=software,
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for {software_name}: {e}")

        return pairs
