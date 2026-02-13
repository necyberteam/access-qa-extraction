"""Extractor for allocations — direct API pagination.

Fetches ALL allocation projects by paginating the public API at
allocations.access-ci.org/current-projects.json?page=N, then uses
fixed question categories to generate Q&A pairs via LLM.
"""

import json
import re

import httpx

from ..generators.factoids import generate_factoid_pairs
from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from ..question_categories import build_system_prompt, build_user_prompt
from .base import BaseExtractor, ExtractionOutput, ExtractionReport

ALLOCATIONS_API_URL = "https://allocations.access-ci.org/current-projects.json"


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class AllocationsExtractor(BaseExtractor):
    """Extract Q&A pairs from allocations using direct API pagination."""

    server_name = "allocations"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def run(self) -> ExtractionOutput:
        """Run extraction — no MCPClient needed (uses direct API)."""
        # Overrides BaseExtractor.run() which creates an MCPClient context.
        # This extractor fetches from allocations.access-ci.org directly.
        return await self.extract()

    async def run_report(self) -> ExtractionReport:
        """Run report — no MCPClient needed (uses direct API)."""
        return await self.report()

    async def report(self) -> ExtractionReport:
        """Fetch page 1 to get total page count and a sample of projects."""
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(ALLOCATIONS_API_URL, params={"page": 1})
            resp.raise_for_status()
            data = resp.json()

        projects = data.get("projects", [])
        total_pages = data.get("pages", 1)
        items_per_page = len(projects)

        ids = [str(p.get("projectId", "")) for p in projects if p.get("projectId")]

        return ExtractionReport(
            server_name=self.server_name,
            strategy="list-all",
            queries_used=[],
            total_fetched=items_per_page * total_pages,  # estimate
            unique_entities=items_per_page * total_pages,  # estimate
            sample_ids=ids[:5],
        )

    async def _fetch_all_projects(self) -> list[dict]:
        """Paginate the allocations API and return all projects.

        Fetches page 1 to learn total page count, then iterates through
        remaining pages. Respects --max-entities to stop early.
        """
        all_projects: list[dict] = []
        max_entities = self.extraction_config.max_entities

        async with httpx.AsyncClient(timeout=30.0) as http:
            # Page 1: learn total pages
            resp = await http.get(ALLOCATIONS_API_URL, params={"page": 1})
            resp.raise_for_status()
            data = resp.json()

            page_projects = data.get("projects", [])
            total_pages = data.get("pages", 1)
            all_projects.extend(page_projects)

            print(f"  Allocations API: {total_pages} pages, {len(page_projects)} on page 1")

            if max_entities and len(all_projects) >= max_entities:
                return all_projects[:max_entities]

            # Pages 2..N
            for page_num in range(2, total_pages + 1):
                resp = await http.get(ALLOCATIONS_API_URL, params={"page": page_num})
                resp.raise_for_status()
                data = resp.json()

                page_projects = data.get("projects", [])
                all_projects.extend(page_projects)

                if page_num % 50 == 0 or page_num == total_pages:
                    print(f"  Page {page_num}/{total_pages}: {len(all_projects)} total projects")

                if max_entities and len(all_projects) >= max_entities:
                    return all_projects[:max_entities]

        return all_projects

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all allocation projects."""
        pairs: ExtractionResult = []
        raw_data: dict = {}

        projects = await self._fetch_all_projects()
        print(f"  Fetched {len(projects)} projects, generating Q&A pairs...")

        system_prompt = build_system_prompt("allocations")

        entity_count = 0
        for project in projects:
            project_id = str(project.get("projectId", "") or project.get("requestNumber", ""))
            title = project.get("requestTitle", "")
            if not project_id or not title:
                continue

            # Respect max_entities limit (may have fetched extra on last page)
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break
            entity_count += 1

            clean_project = self._clean_project_data(project)

            project_pairs = await self._generate_qa_pairs(project_id, clean_project, system_prompt)
            pairs.extend(project_pairs)

            # Generate factoid Q&A pairs from templates (zero LLM)
            factoid_pairs = generate_factoid_pairs("allocations", project_id, clean_project)
            pairs.extend(factoid_pairs)

            raw_data[project_id] = {
                "name": title,
                "project_id": project_id,
                "pi": project.get("pi", ""),
                "institution": project.get("piInstitution", ""),
                "fos": project.get("fos", ""),
                "allocation_type": project.get("allocationType", ""),
                "resource_count": len(project.get("resources", [])),
                "resource_names": [
                    r.get("resourceName", "")
                    for r in project.get("resources", [])
                    if r.get("resourceName")
                ],
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
        self, project_id: str, project: dict, system_prompt: str
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from project data."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(project, indent=2)
        user_prompt = build_user_prompt("allocations", project_id, entity_json)

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
                        pair_id = f"allocations_{project_id}_{category}"

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
                                source_data={"project": project},
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for allocation project {project_id}: {e}")

        return pairs
