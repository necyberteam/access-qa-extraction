"""Extractor for affinity-groups MCP server.

Uses an LLM to generate Q&A pairs based on actual data returned from MCP tools.
The LLM analyzes each affinity group and generates contextually appropriate questions
based on what information is actually available.
"""

import json
import re

from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from .base import BaseExtractor, ExtractionOutput


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


# This prompt tells the LLM: "you are generating Q&A about ACCESS affinity groups."
# It's domain-specific — the compute-resources one talks about GPUs and nodes,
# this one talks about communities, coordinators, events, and how to join.
SYSTEM_PROMPT = """You are a Q&A pair generator for ACCESS-CI affinity groups. Your task is to generate high-quality question-answer pairs based on the provided affinity group data.

Guidelines:
1. Only generate questions that can be accurately answered from the provided data
2. Do not make up or infer information that isn't explicitly in the data
3. Generate a variety of question types (what is the group about, who coordinates it, how to join, what events are offered, etc.)
4. Questions should be natural - the kind a researcher or student might actually ask
5. Answers should be informative but concise
6. Each answer must end with the citation marker provided

Output format: Return a JSON array of objects with "question" and "answer" fields.
Example:
[
  {"question": "What is the GPU Computing affinity group?", "answer": "The GPU Computing affinity group is a community for researchers interested in GPU-accelerated computing on ACCESS resources. It is coordinated by Jane Smith and provides resources, events, and knowledge base articles about GPU computing best practices.\\n\\n<<SRC:affinity-groups:42>>"},
  {"question": "How can I join the GPU Computing affinity group?", "answer": "You can join the GPU Computing affinity group through their Slack channel or by visiting their support page.\\n\\n<<SRC:affinity-groups:42>>"}
]"""


# This template gets filled in once per affinity group. The LLM sees the actual
# data for that group and generates Q&A pairs from it.
USER_PROMPT_TEMPLATE = """Generate Q&A pairs for this ACCESS-CI affinity group.

Group ID: {group_id}
Citation marker to use: <<SRC:affinity-groups:{group_id}>>

Group Data:
{group_json}

Generate appropriate Q&A pairs based on what information is actually available. Return only the JSON array."""


class AffinityGroupsExtractor(BaseExtractor):
    """Extract Q&A pairs from affinity-groups server using LLM."""

    server_name = "affinity-groups"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        """Accept an optional LLM client.

        This is the same pattern as ComputeResourcesExtractor. The llm_client
        parameter lets tests inject a mock instead of calling a real LLM.
        If not provided, get_llm_client() picks the right backend from env vars.
        """
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all affinity groups.

        This is the main method. It:
        1. Fetches all groups from the MCP server
        2. For each group, fetches detail (events, knowledge base)
        3. Cleans the data
        4. Sends to LLM to generate Q&A pairs
        5. Returns pairs + raw_data (raw_data is used by ComparisonGenerator)
        """
        pairs: ExtractionResult = []
        raw_data: dict = {}
        seen_ids: set[str] = set()

        # Fetch all affinity groups with a single call
        result = await self.client.call_tool("search_affinity_groups", {})
        groups = result.get("items", result.get("groups", []))

        for group in groups:
            group_id = str(group.get("id", ""))
            group_name = group.get("name", "")
            if not group_id or not group_name:
                continue

            # Deduplicate — same pattern as seen_software in software_discovery.py
            if group_id in seen_ids:
                continue
            seen_ids.add(group_id)

            # Fetch detail with events and knowledge base
            detail = await self._fetch_group_detail(group_id)

            # Clean data for LLM consumption
            clean_group = self._clean_group_data(group, detail)

            # source_data is stored in the QAPair metadata so a human reviewer
            # can see exactly what data the LLM was given
            source_data = {"group": clean_group}

            # Send to LLM and get Q&A pairs back
            group_pairs = await self._generate_qa_pairs(
                group_id, clean_group, source_data
            )
            pairs.extend(group_pairs)

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
        """Fetch detailed group info including events and knowledge base.

        The affinity-groups MCP server supports include="all" to get
        events and KB in a single call. If it fails, we just return
        empty — the extractor still works, just with less data.
        """
        try:
            detail = await self.client.call_tool(
                "search_affinity_groups",
                {"id": group_id, "include": "all"}
            )
            return detail
        except Exception:
            return {}

    def _clean_group_data(self, group: dict, detail: dict) -> dict:
        """Clean group data for LLM consumption.

        We keep only the fields that are useful for generating Q&A.
        The LLM doesn't need internal IDs or empty fields.
        """
        cleaned = {
            "name": group.get("name", ""),
            "description": strip_html(group.get("description", "")),
            "coordinator": group.get("coordinator", ""),
            "category": group.get("category", ""),
        }

        # Only include contact/link fields if they have values
        for field in ["slack_link", "support_url", "ask_ci_forum"]:
            value = group.get(field, "")
            if value:
                cleaned[field] = value

        # Add event and KB summaries from detail response
        events = detail.get("events", {})
        if events.get("total", 0) > 0:
            event_items = events.get("items", [])
            cleaned["upcoming_events"] = [
                {"title": e.get("title", ""), "date": e.get("date", "")}
                for e in event_items[:5]  # Cap at 5 to keep prompt manageable
                if e.get("title")
            ]

        kb = detail.get("knowledge_base", {})
        if kb.get("total", 0) > 0:
            kb_items = kb.get("items", [])
            cleaned["knowledge_base_topics"] = [
                e.get("title", "") for e in kb_items[:5]
                if e.get("title")
            ]

        return cleaned

    async def _generate_qa_pairs(
        self, group_id: str, group: dict, source_data: dict
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from group data.

        This is where the LLM call happens. We:
        1. Format the cleaned data as JSON
        2. Fill in the prompt template
        3. Call the LLM
        4. Parse the JSON array from the response
        5. Wrap each Q&A in a QAPair object

        The try/except means if the LLM returns garbage for one group,
        we skip it and keep going with the next group.
        """
        pairs: ExtractionResult = []

        group_json = json.dumps(group, indent=2)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            group_id=group_id,
            group_json=group_json,
        )

        try:
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=2048,
            )

            response_text = response.text

            # The LLM might wrap JSON in markdown code blocks, so we
            # extract just the [...] array part
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                for qa in qa_list:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if question and answer:
                        # ID format: ag_{group_id}_{slug_of_question}
                        q_slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:30]
                        pair_id = f"ag_{group_id}_{q_slug}"

                        complexity = "simple"
                        if any(term in question.lower() for term in
                               ["how to", "steps", "process", "compare"]):
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
