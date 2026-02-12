"""Extractor for allocations MCP server.

Uses an LLM to generate Q&A pairs based on actual data returned from MCP tools.
The LLM analyzes each allocation project and generates contextually appropriate questions
based on what information is actually available.
"""

import json
import re

from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from .base import BaseExtractor, ExtractionOutput, ExtractionReport


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


SYSTEM_PROMPT = """You are a Q&A pair generator for ACCESS-CI allocation projects. Your task is to generate high-quality question-answer pairs based on the provided allocation project data.

Guidelines:
1. Only generate questions that can be accurately answered from the provided data
2. Do not make up or infer information that isn't explicitly in the data
3. Generate a variety of question types (what is the project about, who is the PI, what resources are allocated, what field of science, allocation dates, etc.)
4. Questions should be natural - the kind a researcher or student might actually ask
5. Answers should be informative but concise
6. Each answer must end with the citation marker provided

Output format: Return a JSON array of objects with "question" and "answer" fields.
Example:
[
  {"question": "What is allocation project TG-CIS210014?", "answer": "TG-CIS210014 is a research allocation titled 'Machine Learning for Climate Prediction' led by PI John Doe from MIT. It is in the field of Computer Science and runs from 2024-01-01 to 2025-12-31.\\n\\n<<SRC:allocations:TG-CIS210014>>"},
  {"question": "What resources are allocated to project TG-CIS210014?", "answer": "Project TG-CIS210014 has allocations on Delta (50,000 GPU hours) and Expanse (100,000 SUs).\\n\\n<<SRC:allocations:TG-CIS210014>>"}
]"""


USER_PROMPT_TEMPLATE = """Generate Q&A pairs for this ACCESS-CI allocation project.

Project ID: {project_id}
Citation marker to use: <<SRC:allocations:{project_id}>>

Project Data:
{project_json}

Generate appropriate Q&A pairs based on what information is actually available. Return only the JSON array."""


# The allocations server requires at least one search parameter (no list-all fallback).
# These queries are derived from the known enumerations in access-mcp's taxonomies.ts
# (FIELDS_OF_SCIENCE, ALLOCATION_TYPES, ACCESS_SYSTEMS) plus supplemental keywords.
# Deduplication by project ID happens across all queries.
# Use ExtractionConfig.max_queries to limit how many are used in a given run.
ALLOCATION_QUERIES = [
    # Fields of Science — the 9 NSF categories from taxonomies.ts
    "Computer Science",
    "Biological Sciences",
    "Physics",
    "Chemistry",
    "Engineering",
    "Earth Sciences",
    "Mathematics and Statistics",
    "Social Sciences",
    "Astronomy and Astrophysics",
    # Allocation Types — the 4 tiers from taxonomies.ts
    "Discover",
    "Explore",
    "Accelerate",
    "Maximize",
    # Major ACCESS systems — from taxonomies.ts ACCESS_SYSTEMS
    "Delta",
    "Bridges-2",
    "Anvil",
    "Expanse",
    "Stampede3",
    "Jetstream2",
    "Open Science Grid",
    # Supplemental keywords for topics that cross FoS boundaries
    "machine learning",
    "genomics",
    "climate",
    "molecular dynamics",
]


class AllocationsExtractor(BaseExtractor):
    """Extract Q&A pairs from allocations server using LLM."""

    server_name = "allocations"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def report(self) -> ExtractionReport:
        """Search MCP for allocation projects and return coverage stats."""
        queries = ALLOCATION_QUERIES[: self.extraction_config.max_queries]
        seen: set[str] = set()
        total_fetched = 0

        for query in queries:
            params = {"query": query, "limit": self.extraction_config.search_limit}
            result = await self.client.call_tool("search_projects", params)
            items = result.get("items", result.get("projects", []))
            total_fetched += len(items)
            for p in items:
                pid = str(p.get("projectId", "") or p.get("requestNumber", ""))
                if pid:
                    seen.add(pid)

        return ExtractionReport(
            server_name=self.server_name,
            strategy="broad-queries",
            queries_used=queries,
            total_fetched=total_fetched,
            unique_entities=len(seen),
            sample_ids=sorted(seen)[:5],
        )

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all allocation projects."""
        pairs: ExtractionResult = []
        raw_data: dict = {}
        seen_ids: set[str] = set()

        queries = ALLOCATION_QUERIES[: self.extraction_config.max_queries]

        projects = []
        for i, query in enumerate(queries):
            params = {"query": query, "limit": self.extraction_config.search_limit}
            result = await self.client.call_tool("search_projects", params)
            items = result.get("items", result.get("projects", []))
            new_count = sum(
                1 for p in items
                if (p.get("projectId") or p.get("requestNumber")) not in seen_ids
            )
            print(
                f"  [{i + 1}/{len(queries)}] "
                f"'{query}' → {len(items)} results, {new_count} new"
            )
            projects.extend(items)

        entity_count = 0
        for project in projects:
            project_id = project.get("projectId", "") or project.get("requestNumber", "")
            title = project.get("requestTitle", "")
            if not project_id or not title:
                continue

            if project_id in seen_ids:
                continue
            seen_ids.add(project_id)

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            clean_project = self._clean_project_data(project)
            source_data = {"project": clean_project}

            project_pairs = await self._generate_qa_pairs(project_id, clean_project, source_data)
            pairs.extend(project_pairs)

            raw_data[project_id] = {
                "name": title,
                "project_id": project_id,
                "pi": project.get("pi", ""),
                "institution": project.get("piInstitution", ""),
                "fos": project.get("fos", ""),
                "allocation_type": project.get("allocationType", ""),
                "resource_count": len(project.get("resources", [])),
            }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    def _clean_project_data(self, project: dict) -> dict:
        """Clean project data for LLM consumption."""
        cleaned = {
            "title": project.get("requestTitle", ""),
            "pi": project.get("pi", ""),
            "institution": project.get("piInstitution", ""),
            "field_of_science": project.get("fos", ""),
            "allocation_type": project.get("allocationType", ""),
        }

        for field in ["abstract", "beginDate", "endDate"]:
            value = project.get(field, "")
            if value:
                if field == "abstract":
                    cleaned[field] = strip_html(value)
                else:
                    cleaned[field] = value

        resources = project.get("resources", [])
        if resources:
            cleaned["resources"] = [
                {
                    "name": r.get("resourceName", ""),
                    "units": r.get("units", ""),
                    "allocation": r.get("allocation", ""),
                }
                for r in resources
                if r.get("resourceName")
            ]

        return cleaned

    async def _generate_qa_pairs(
        self, project_id: str, project: dict, source_data: dict
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from project data."""
        pairs: ExtractionResult = []

        project_json = json.dumps(project, indent=2)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            project_id=project_id,
            project_json=project_json,
        )

        try:
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )

            response_text = response.text

            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                for qa in qa_list:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if question and answer:
                        q_slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:30]
                        pair_id = f"alloc_{project_id}_{q_slug}"

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
                                source_ref=f"mcp://allocations/projects/{project_id}",
                                domain="allocations",
                                complexity=complexity,
                                source_data=source_data,
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for allocation project {project_id}: {e}")

        return pairs
