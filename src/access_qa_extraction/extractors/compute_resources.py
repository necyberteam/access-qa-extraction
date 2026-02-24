"""Extractor for compute-resources MCP server.

Uses an LLM with freeform extraction to generate Q&A pairs based on actual data
returned from MCP tools. The LLM generates as many pairs as the data warrants,
guided by per-domain key areas but not constrained to them.

Fetches all resources via search_resources({}) (list-all strategy), then fetches
hardware details per resource.
"""

import json
import re

from ..generators.incremental import compute_entity_hash
from ..generators.judge import evaluate_pairs
from ..llm_client import BaseLLMClient, get_judge_client, get_llm_client
from ..models import ExtractionResult, QAPair
from ..question_categories import (
    build_battery_system_prompt,
    build_discovery_system_prompt,
    build_user_prompt,
)
from .base import BaseExtractor, ExtractionOutput, ExtractionReport


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class ComputeResourcesExtractor(BaseExtractor):
    """Extract Q&A pairs from compute-resources server using LLM."""

    server_name = "compute-resources"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()
        self.judge_client = None
        if not self.extraction_config.no_judge:
            try:
                self.judge_client = get_judge_client()
            except (ValueError, ImportError):
                pass

    async def report(self) -> ExtractionReport:
        """Fetch all resources from MCP and return coverage stats."""
        result = await self.client.call_tool("search_resources", {"query": ""})
        resources = result.get("resources", result.get("items", []))

        ids = []
        for r in resources:
            rid = r.get("id", "")
            name = r.get("name", "")
            if rid and name and not ("COMING SOON" in name and not r.get("description")):
                ids.append(rid)

        return ExtractionReport(
            server_name=self.server_name,
            strategy="list-all",
            queries_used=[],
            total_fetched=len(resources),
            unique_entities=len(ids),
            sample_ids=ids[:5],
        )

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all compute resources."""
        pairs: ExtractionResult = []
        raw_data: dict = {}

        # GUIDED-TOUR.md § Step 3A.1 — fetch all entities from MCP
        result = await self.client.call_tool("search_resources", {"query": ""})
        resources = result.get("resources", result.get("items", []))

        # GUIDED-TOUR.md § Step 3A.2 — build battery system prompt once, reused per entity
        system_prompt = build_battery_system_prompt("compute-resources")

        # GUIDED-TOUR.md § Step 3A.3 — per-entity loop (filter, cap, fetch hardware, clean, hash)
        entity_count = 0
        for resource in resources:
            resource_id = resource.get("id", "")
            resource_name = resource.get("name", "")
            if not resource_id or not resource_name:
                continue

            # Normalize "COMING SOON" label to lowercase — keep the data, tone down the caps
            if "COMING SOON" in resource_name:
                resource_name = resource_name.replace("- COMING SOON", "(coming soon)").replace("COMING SOON", "(coming soon)").strip()
                resource["name"] = resource_name

            # Filter to specific entity IDs if requested
            if self.extraction_config.entity_ids is not None:
                if resource_id not in self.extraction_config.entity_ids:
                    continue

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            # Fetch hardware details
            hardware = {}
            try:
                hw_result = await self.client.call_tool(
                    "get_resource_hardware", {"id": resource_id}
                )
                hardware = hw_result
            except Exception:
                pass

            # Clean up data for LLM
            clean_resource = self._clean_resource_data(resource)
            clean_hardware = self._clean_hardware_data(hardware)

            # Merge hardware into entity data so the generic user prompt works
            entity_data = {**clean_resource}
            if clean_hardware:
                entity_data["hardware"] = clean_hardware

            # GUIDED-TOUR.md § Step 3A.4 — hash entity data; check cache; skip LLM if unchanged
            entity_hash = compute_entity_hash(entity_data)
            used_cache = False
            if self.incremental_cache:
                if self.incremental_cache.is_unchanged(
                    "compute-resources", resource_id, entity_hash
                ):
                    cached_pairs = self.incremental_cache.get_cached_pairs(
                        "compute-resources", resource_id
                    )
                    if cached_pairs:
                        pairs.extend(cached_pairs)
                        used_cache = True

            if not used_cache:
                source_data = {
                    "resource": clean_resource,
                    "hardware": clean_hardware if clean_hardware else None,
                }

                # Generate Q&A pairs using LLM (freeform — variable count)
                resource_pairs = await self._generate_qa_pairs(
                    resource_id, entity_data, source_data, system_prompt
                )
                pairs.extend(resource_pairs)

                # GUIDED-TOUR.md § Step 3A.6 — judge scores all pairs for this entity (3rd LLM call)
                if self.judge_client:
                    evaluate_pairs(resource_pairs, source_data, self.judge_client)

                # GUIDED-TOUR.md § Step 3A.7 — store pairs + scores in cache for next incremental run
                if self.incremental_cache:
                    self.incremental_cache.store(
                        "compute-resources",
                        resource_id,
                        entity_hash,
                        resource_pairs,
                    )

            # GUIDED-TOUR.md § Step 3A.8 — collect normalized raw_data for ComparisonGenerator
            raw_data[resource_id] = {
                "name": self._clean_name(resource_name),
                "resource_id": resource_id,
                "organizations": resource.get("organization_names", []),
                "has_gpu": resource.get("hasGpu", False),
                "gpu_types": self._extract_gpu_types(clean_hardware),
                "features": clean_resource.get("feature_names", []),
                "resource_type": resource.get("resourceType", ""),
            }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    def _clean_name(self, name: str) -> str:
        """Clean resource name by removing status indicators."""
        name = re.sub(r"\s*-\s*(COMING SOON|RETIRED|BETA).*$", "", name, flags=re.IGNORECASE)
        return name.strip()

    def _extract_gpu_types(self, hardware: dict) -> list[str]:
        """Extract GPU type names from hardware data."""
        gpu_types = []
        if not hardware:
            return gpu_types

        for gpu in hardware.get("gpus", []):
            gpu_name = gpu.get("name", "")
            if gpu_name:
                gpu_types.append(gpu_name)

        for node in hardware.get("compute_nodes", []):
            details = node.get("details", "")
            gpu_patterns = [
                r"NVIDIA\s+([A-Z]\d+\s*\d*\s*(?:GB)?)",
                r"(A100|V100|H100|A40|A30|RTX\s*\d+)",
            ]
            for pattern in gpu_patterns:
                matches = re.findall(pattern, details, re.IGNORECASE)
                for match in matches:
                    normalized = match.strip().upper()
                    if normalized and normalized not in gpu_types:
                        gpu_types.append(normalized)

        return gpu_types

    def _clean_resource_data(self, resource: dict) -> dict:
        """Clean resource data for LLM consumption."""
        useful_fields = [
            "name",
            "description",
            "organization_names",
            "feature_names",
            "hasGpu",
            "resourceType",
            "accessAllocated",
        ]
        cleaned = {k: v for k, v in resource.items() if k in useful_fields}

        if "feature_names" in cleaned:
            cleaned["feature_names"] = [
                f for f in cleaned["feature_names"] if not f.startswith("Unknown")
            ]

        if "description" in cleaned:
            cleaned["description"] = strip_html(cleaned["description"])

        return cleaned

    def _clean_hardware_data(self, hardware: dict) -> dict:
        """Clean hardware data for LLM consumption."""
        if not hardware:
            return {}

        result = {}
        hw = hardware.get("hardware", hardware)

        for category in ["gpus", "compute_nodes", "storage", "memory"]:
            if category in hw and hw[category]:
                items = []
                for item in hw[category]:
                    cleaned_item = {
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "details": strip_html(item.get("details", "")),
                    }
                    if cleaned_item["details"] and len(cleaned_item["details"]) > 50:
                        items.append(cleaned_item)
                if items:
                    result[category] = items

        return result

    async def _generate_qa_pairs(
        self, resource_id: str, entity_data: dict, source_data: dict, system_prompt: str
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from resource data."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(entity_data, indent=2)
        user_prompt = build_user_prompt("compute-resources", resource_id, entity_json)

        try:
            # GUIDED-TOUR.md § Step 3A — battery LLM call (guaranteed field coverage)
            response = self.llm.generate(
                system=system_prompt,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )

            qa_list = self._parse_qa_response(response.text)

            # GUIDED-TOUR.md § Step 3A — discovery LLM call (finds what battery missed)
            # Discovery call: find what the battery missed
            if qa_list:
                existing = [{"question": qa["question"], "answer": qa["answer"]} for qa in qa_list]
                discovery_prompt = build_discovery_system_prompt("compute-resources", existing)
                discovery_response = self.llm.generate(
                    system=discovery_prompt,
                    user=user_prompt,
                    max_tokens=self.extraction_config.max_tokens,
                )
                qa_list.extend(self._parse_qa_response(discovery_response.text))

            for seq_n, qa in enumerate(qa_list, start=1):
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if question and answer:
                    pair_id = f"compute-resources_{resource_id}_{seq_n}"

                    complexity = "simple"
                    if any(
                        term in question.lower()
                        for term in ["specifications", "how many", "performance", "compared"]
                    ):
                        complexity = "moderate"

                    pairs.append(
                        QAPair.create(
                            id=pair_id,
                            question=question,
                            answer=answer,
                            source_ref=f"mcp://compute-resources/resources/{resource_id}",
                            domain="compute-resources",
                            complexity=complexity,
                            source_data=source_data,
                        )
                    )

        except Exception as e:
            print(f"Error generating Q&A for {resource_id}: {e}")

        return pairs

    @staticmethod
    def _parse_qa_response(response_text: str) -> list[dict]:
        """Parse a JSON array of Q&A pairs from an LLM response."""
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            return json.loads(json_match.group())
        return []
