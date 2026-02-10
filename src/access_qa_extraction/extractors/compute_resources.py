"""Extractor for compute-resources MCP server.

Uses an LLM to generate Q&A pairs based on actual data returned from MCP tools.
The LLM analyzes each resource and generates contextually appropriate questions
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


SYSTEM_PROMPT = """You are a Q&A pair generator for ACCESS-CI compute resources. Your task is to generate high-quality question-answer pairs based on the provided resource data.

Guidelines:
1. Only generate questions that can be accurately answered from the provided data
2. Do not make up or infer information that isn't explicitly in the data
3. Generate a variety of question types (what is, who operates, technical specs, capabilities, etc.)
4. Questions should be natural - the kind a researcher might actually ask
5. Answers should be informative but concise
6. If hardware specs are just generic descriptions without actual specs (core counts, memory sizes, GPU models), don't generate hardware spec questions
7. Each answer must end with the citation marker provided

Output format: Return a JSON array of objects with "question" and "answer" fields.
Example:
[
  {"question": "What is Delta?", "answer": "Delta is a dedicated, ACCESS-allocated resource designed by HPE and NCSA, delivering a highly capable GPU-focused compute environment for GPU and CPU workloads.\\n\\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"},
  {"question": "Who operates Delta?", "answer": "Delta is operated by the National Center for Supercomputing Applications (NCSA).\\n\\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"}
]"""


USER_PROMPT_TEMPLATE = """Generate Q&A pairs for this ACCESS-CI compute resource.

Resource ID: {resource_id}
Citation marker to use: <<SRC:compute-resources:{resource_id}>>

Resource Data:
{resource_json}

Hardware Data:
{hardware_json}

Generate appropriate Q&A pairs based on what information is actually available. Return only the JSON array."""


class ComputeResourcesExtractor(BaseExtractor):
    """Extract Q&A pairs from compute-resources server using LLM."""

    server_name = "compute-resources"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs for all compute resources."""
        pairs: ExtractionResult = []
        raw_data: dict = {}  # Normalized data for comparison generation

        # Fetch all resources
        result = await self.client.call_tool("search_resources", {"query": ""})
        resources = result.get("resources", result.get("items", []))

        for resource in resources:
            resource_id = resource.get("id", "")
            resource_name = resource.get("name", "")
            if not resource_id or not resource_name:
                continue

            # Skip "COMING SOON" resources with no real data
            if "COMING SOON" in resource_name and not resource.get("description"):
                continue

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

            # Combine into source_data for reviewer verification
            source_data = {
                "resource": clean_resource,
                "hardware": clean_hardware if clean_hardware else None,
            }

            # Generate Q&A pairs using LLM
            resource_pairs = await self._generate_qa_pairs(
                resource_id, clean_resource, clean_hardware, source_data
            )
            pairs.extend(resource_pairs)

            # Store normalized data for comparison generation
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
        # Remove "- COMING SOON" and similar suffixes
        name = re.sub(r'\s*-\s*(COMING SOON|RETIRED|BETA).*$', '', name, flags=re.IGNORECASE)
        return name.strip()

    def _extract_gpu_types(self, hardware: dict) -> list[str]:
        """Extract GPU type names from hardware data."""
        gpu_types = []
        if not hardware:
            return gpu_types

        # Look in gpus section
        for gpu in hardware.get("gpus", []):
            gpu_name = gpu.get("name", "")
            if gpu_name:
                gpu_types.append(gpu_name)

        # Also check compute_nodes for GPU mentions
        for node in hardware.get("compute_nodes", []):
            details = node.get("details", "")
            # Extract common GPU model patterns
            gpu_patterns = [
                r'NVIDIA\s+([A-Z]\d+\s*\d*\s*(?:GB)?)',  # NVIDIA A100, V100, etc.
                r'(A100|V100|H100|A40|A30|RTX\s*\d+)',
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
        # Remove internal IDs and keep useful fields
        useful_fields = [
            "name", "description", "organization_names", "feature_names",
            "hasGpu", "resourceType", "accessAllocated"
        ]
        cleaned = {k: v for k, v in resource.items() if k in useful_fields}

        # Filter out "Unknown Feature" entries
        if "feature_names" in cleaned:
            cleaned["feature_names"] = [
                f for f in cleaned["feature_names"]
                if not f.startswith("Unknown")
            ]

        # Strip HTML from description
        if "description" in cleaned:
            cleaned["description"] = strip_html(cleaned["description"])

        return cleaned

    def _clean_hardware_data(self, hardware: dict) -> dict:
        """Clean hardware data for LLM consumption."""
        if not hardware:
            return {}

        result = {}
        hw = hardware.get("hardware", hardware)

        # Process each hardware category
        for category in ["gpus", "compute_nodes", "storage", "memory"]:
            if category in hw and hw[category]:
                items = []
                for item in hw[category]:
                    cleaned_item = {
                        "name": item.get("name", ""),
                        "type": item.get("type", ""),
                        "details": strip_html(item.get("details", ""))
                    }
                    # Only include if details has meaningful content
                    if cleaned_item["details"] and len(cleaned_item["details"]) > 50:
                        items.append(cleaned_item)
                if items:
                    result[category] = items

        return result

    async def _generate_qa_pairs(
        self, resource_id: str, resource: dict, hardware: dict, source_data: dict
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from resource data."""
        pairs: ExtractionResult = []

        # Format data for LLM
        resource_json = json.dumps(resource, indent=2)
        hardware_json = json.dumps(hardware, indent=2) if hardware else "No hardware details available"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            resource_id=resource_id,
            resource_json=resource_json,
            hardware_json=hardware_json
        )

        try:
            response = self.llm.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )

            # Parse LLM response
            response_text = response.text

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                qa_list = json.loads(json_match.group())

                # Convert to QAPair objects
                for i, qa in enumerate(qa_list):
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    if question and answer:
                        # Generate unique ID based on question
                        q_slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:30]
                        pair_id = f"cr_{resource_id}_{q_slug}".replace(".", "_")

                        # Determine complexity based on question type
                        complexity = "simple"
                        if any(term in question.lower() for term in
                               ["specifications", "how many", "performance", "compared"]):
                            complexity = "moderate"

                        # Domain is just the server name - sub-categories can be derived from source_data if needed
                        domain = "compute-resources"

                        pairs.append(
                            QAPair.create(
                                id=pair_id,
                                question=question,
                                answer=answer,
                                source_ref=f"mcp://compute-resources/resources/{resource_id}",
                                domain=domain,
                                complexity=complexity,
                                source_data=source_data,
                            )
                        )

        except Exception as e:
            # Log error but continue with other resources
            print(f"Error generating Q&A for {resource_id}: {e}")

        return pairs
