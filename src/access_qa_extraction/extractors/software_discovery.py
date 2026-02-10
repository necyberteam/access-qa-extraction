"""Extractor for software-discovery MCP server.

Uses an LLM to generate Q&A pairs based on software catalog data.
The LLM analyzes each software package and generates contextually appropriate
questions based on available metadata including versions, resources, and AI-enhanced data.
"""

import json
import re

from ..llm_client import BaseLLMClient, get_llm_client
from ..models import ExtractionResult, QAPair
from .base import BaseExtractor, ExtractionOutput

SYSTEM_PROMPT = """You are a Q&A pair generator for ACCESS-CI software catalog. Your task is to generate high-quality question-answer pairs based on the provided software data.

Guidelines:
1. Only generate questions that can be accurately answered from the provided data
2. Do not make up or infer information that isn't explicitly in the data
3. Generate a variety of question types:
   - What is [software]?
   - What resources have [software] available?
   - What versions of [software] are available on [resource]?
   - What is [software] used for?
   - How do I use [software]? (only if usage examples are provided)
4. Questions should be natural - the kind a researcher might actually ask
5. Answers should be informative but concise
6. If there's example usage code, include it in relevant answers
7. Each answer must end with the citation marker provided

Output format: Return a JSON array of objects with "question" and "answer" fields."""


USER_PROMPT_TEMPLATE = """Generate Q&A pairs for this ACCESS-CI software package.

Software Name: {software_name}
Citation marker to use: <<SRC:software-discovery:{software_name}>>

Software Data:
{software_json}

Generate appropriate Q&A pairs based on what information is actually available. Return only the JSON array."""


# Common software categories to search for comprehensive coverage.
# All terms are always available — use ExtractionConfig.max_queries to limit
# how many are used in a given run (e.g., max_queries=2 for cheap test runs).
SOFTWARE_SEARCH_TERMS = [
    "python",
    "cuda",
    "gcc",
    "tensorflow",
    "pytorch",
    "mpi",
    "openmpi",
    "r",
    "matlab",
    "julia",
    "gromacs",
    "namd",
    "lammps",
    "vasp",
    "gaussian",
    "ansys",
    "cmake",
    "git",
    "singularity",
    "apptainer",
    "conda",
    "java",
    "perl",
    "rust",
    "llvm",
    "boost",
    "hdf5",
    "netcdf",
    "fftw",
    "blas",
    "lapack",
    "openssl",
    "vim",
    "emacs",
    "slurm",
]


class SoftwareDiscoveryExtractor(BaseExtractor):
    """Extract Q&A pairs from software-discovery server using LLM."""

    server_name = "software-discovery"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for software packages."""
        pairs: ExtractionResult = []
        raw_data: dict = {}  # Normalized data for comparison generation
        seen_software: set[str] = set()

        terms = SOFTWARE_SEARCH_TERMS[: self.extraction_config.max_queries]

        # Search for each category of software
        for search_term in terms:
            try:
                result = await self.client.call_tool(
                    "search_software",
                    {
                        "query": search_term,
                        "limit": self.extraction_config.search_limit,
                        "include_ai_metadata": True,
                    },
                )

                software_list = result.get("items", result.get("software", []))

                for software in software_list:
                    name = software.get("name", "").lower()

                    # Skip if we've already processed this software
                    if name in seen_software:
                        continue
                    seen_software.add(name)

                    # Clean up data for LLM (also serves as source_data for review)
                    clean_software = self._clean_software_data(software)

                    # Generate Q&A pairs using LLM
                    software_pairs = await self._generate_qa_pairs(
                        name, clean_software, source_data=clean_software
                    )
                    pairs.extend(software_pairs)

                    # Store normalized data for comparison generation
                    raw_data[name] = {
                        "name": software.get("name", name),
                        "software_id": name,
                        "resources": self._extract_resource_ids(software),
                        "tags": clean_software.get("tags", []),
                        "research_area": clean_software.get("research_area"),
                        "software_type": clean_software.get("software_type"),
                    }

            except Exception as e:
                print(f"Error searching for '{search_term}': {e}")
                continue

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    def _extract_resource_ids(self, software: dict) -> list[str]:
        """Extract resource IDs where software is available."""
        resources = software.get("available_on_resources", [])
        if isinstance(resources, list):
            # Handle different formats - could be list of dicts or list of strings
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
                # Truncate very long examples
                example = ai_meta["example_use"]
                if len(example) > 1500:
                    example = example[:1500] + "..."
                cleaned["example_use"] = example

        return cleaned

    async def _generate_qa_pairs(
        self, software_name: str, software: dict, source_data: dict
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from software data."""
        pairs: ExtractionResult = []

        software_json = json.dumps(software, indent=2)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            software_name=software_name,
            software_json=software_json
        )

        try:
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )

            response_text = response.text

            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                for qa in qa_list:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if question and answer:
                        # Generate unique ID
                        q_slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:30]
                        pair_id = f"sw_{software_name}_{q_slug}".replace("-", "_")

                        # Determine complexity
                        complexity = "simple"
                        if any(term in question.lower() for term in
                               ["how do i", "how to", "example", "versions"]):
                            complexity = "moderate"

                        # Domain is just the server name - sub-categories can be derived from source_data if needed
                        domain = "software-discovery"

                        pairs.append(
                            QAPair.create(
                                id=pair_id,
                                question=question,
                                answer=answer,
                                source_ref=f"mcp://software-discovery/software/{software_name}",
                                domain=domain,
                                complexity=complexity,
                                source_data=source_data,
                            )
                        )

        except Exception as e:
            print(f"Error generating Q&A for {software_name}: {e}")

        return pairs
