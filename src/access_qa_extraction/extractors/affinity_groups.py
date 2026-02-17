"""Extractor for affinity-groups MCP server.

Uses an LLM with fixed question categories to generate Q&A pairs based on
actual data returned from MCP tools. Fetches all groups via
search_affinity_groups({}) (list-all strategy), then fetches detail
(events + KB) per group.
"""

import json
import re

from ..generators.factoids import generate_factoid_pairs
from ..generators.incremental import compute_entity_hash
from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from ..question_categories import build_system_prompt, build_user_prompt
from .base import BaseExtractor, ExtractionOutput, ExtractionReport


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class AffinityGroupsExtractor(BaseExtractor):
    """Extract Q&A pairs from affinity-groups server using LLM."""

    server_name = "affinity-groups"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def report(self) -> ExtractionReport:
        """Fetch all affinity groups from MCP and return coverage stats."""
        result = await self.client.call_tool("search_affinity_groups", {})
        groups = result.get("items", result.get("groups", []))

        seen_ids: set[str] = set()
        for g in groups:
            gid = str(g.get("id", ""))
            if gid and g.get("name") and gid not in seen_ids:
                seen_ids.add(gid)

        return ExtractionReport(
            server_name=self.server_name,
            strategy="list-all",
            queries_used=[],
            total_fetched=len(groups),
            unique_entities=len(seen_ids),
            sample_ids=list(seen_ids)[:5],
        )

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all affinity groups."""
        pairs: ExtractionResult = []
        raw_data: dict = {}
        seen_ids: set[str] = set()

        # Fetch all affinity groups with a single call
        result = await self.client.call_tool("search_affinity_groups", {})
        groups = result.get("items", result.get("groups", []))

        system_prompt = build_system_prompt("affinity-groups")

        entity_count = 0
        for group in groups:
            group_id = str(group.get("id", ""))
            group_name = group.get("name", "")
            if not group_id or not group_name:
                continue

            if group_id in seen_ids:
                continue
            seen_ids.add(group_id)

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            # Fetch detail with events and knowledge base
            detail = await self._fetch_group_detail(group_id)

            # Clean data for LLM consumption
            clean_group = self._clean_group_data(group, detail)

            # Incremental: skip LLM + factoid if entity data unchanged
            entity_hash = compute_entity_hash(clean_group)
            used_cache = False
            if self.incremental_cache:
                if self.incremental_cache.is_unchanged(
                    "affinity-groups", group_id, entity_hash
                ):
                    cached_pairs = self.incremental_cache.get_cached_pairs(
                        "affinity-groups", group_id
                    )
                    if cached_pairs:
                        pairs.extend(cached_pairs)
                        used_cache = True

            if not used_cache:
                source_data = {"group": clean_group}

                # Send to LLM and get Q&A pairs back
                group_pairs = await self._generate_qa_pairs(
                    group_id, clean_group, source_data, system_prompt
                )
                pairs.extend(group_pairs)

                # Generate factoid Q&A pairs from templates (zero LLM)
                factoid_pairs = generate_factoid_pairs(
                    "affinity-groups", group_id, clean_group
                )
                pairs.extend(factoid_pairs)

                if self.incremental_cache:
                    self.incremental_cache.store(
                        "affinity-groups",
                        group_id,
                        entity_hash,
                        group_pairs + factoid_pairs,
                    )

            # Store normalized data for ComparisonGenerator
            raw_data[group_id] = {
                "name": group_name,
                "group_id": group_id,
                "category": group.get("category", ""),
                "coordinator": group.get("coordinator", ""),
                "has_events": bool(detail.get("events", {}).get("total", 0)),
                "has_knowledge_base": bool(detail.get("knowledge_base", {}).get("total", 0)),
            }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    async def _fetch_group_detail(self, group_id: str) -> dict:
        """Fetch detailed group info including events and knowledge base."""
        try:
            detail = await self.client.call_tool(
                "search_affinity_groups", {"id": group_id, "include": "all"}
            )
            return detail
        except Exception:
            return {}

    def _clean_group_data(self, group: dict, detail: dict) -> dict:
        """Clean group data for LLM consumption."""
        cleaned = {
            "name": group.get("name", ""),
            "description": strip_html(group.get("description", "")),
            "coordinator": group.get("coordinator", ""),
            "category": group.get("category", ""),
        }

        for field in ["slack_link", "support_url", "ask_ci_forum"]:
            value = group.get(field, "")
            if value:
                cleaned[field] = value

        events = detail.get("events", {})
        if events.get("total", 0) > 0:
            event_items = events.get("items", [])
            max_items = self.extraction_config.max_detail_items
            cleaned["upcoming_events"] = [
                {"title": e.get("title", ""), "date": e.get("date", "")}
                for e in event_items[:max_items]
                if e.get("title")
            ]

        kb = detail.get("knowledge_base", {})
        if kb.get("total", 0) > 0:
            kb_items = kb.get("items", [])
            max_items = self.extraction_config.max_detail_items
            cleaned["knowledge_base_topics"] = [
                e.get("title", "") for e in kb_items[:max_items] if e.get("title")
            ]

        return cleaned

    async def _generate_qa_pairs(
        self, group_id: str, group: dict, source_data: dict, system_prompt: str
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from group data."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(group, indent=2)
        user_prompt = build_user_prompt("affinity-groups", group_id, entity_json)

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
                        pair_id = f"affinity-groups_{group_id}_{category}"

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
                                source_ref=f"mcp://affinity-groups/groups/{group_id}",
                                domain="affinity-groups",
                                complexity=complexity,
                                source_data=source_data,
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for affinity group {group_id}: {e}")

        return pairs
