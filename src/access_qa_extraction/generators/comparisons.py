"""Generate cross-resource comparison Q&A pairs.

This module generates comparison questions that span multiple resources,
such as "Which ACCESS resources have A100 GPUs?" or "Which GPU resources
have PyTorch installed?"

All answers are built programmatically from extracted data to avoid hallucination.
"""

import re
from collections import defaultdict

from ..models import ExtractionResult, QAPair


class ComparisonGenerator:
    """Generate cross-resource comparison Q&A pairs."""

    def __init__(self):
        self._resource_name_to_id: dict[str, str] = {}

    def _build_resource_lookup(self, compute_data: dict) -> None:
        """Build a lookup from short resource names to full IDs."""
        for resource_id, data in compute_data.items():
            name = data.get("name", "").lower()
            # Map various forms of the name to the full ID
            self._resource_name_to_id[name] = resource_id
            # Also map the short form (first part of domain)
            short_name = resource_id.split(".")[0].lower()
            self._resource_name_to_id[short_name] = resource_id
            # Handle hyphenated names like "bridges-2"
            if "-" in name:
                self._resource_name_to_id[name.replace("-", "")] = resource_id

    def _resolve_resource_id(self, short_name: str, compute_data: dict) -> str | None:
        """Resolve a short resource name to its full ID."""
        short_lower = short_name.lower().strip()

        # Direct match in lookup
        if short_lower in self._resource_name_to_id:
            return self._resource_name_to_id[short_lower]

        # Try without hyphens
        no_hyphen = short_lower.replace("-", "")
        if no_hyphen in self._resource_name_to_id:
            return self._resource_name_to_id[no_hyphen]

        # Fuzzy match - check if short_name is prefix of any resource ID
        for resource_id in compute_data.keys():
            if resource_id.lower().startswith(short_lower):
                return resource_id

        return None

    def generate(
        self,
        compute_data: dict,
        software_data: dict,
    ) -> ExtractionResult:
        """Generate comparison Q&A pairs from extracted data.

        Args:
            compute_data: Normalized compute resource data from extraction
            software_data: Normalized software data from extraction

        Returns:
            List of comparison Q&A pairs
        """
        pairs: ExtractionResult = []

        # Build resource name lookup for cross-referencing
        self._build_resource_lookup(compute_data)

        # Type 1: GPU availability questions
        pairs.extend(self._generate_gpu_availability_questions(compute_data))

        # Type 2: Feature availability questions
        pairs.extend(self._generate_feature_availability_questions(compute_data))

        # Type 3: Organization questions
        pairs.extend(self._generate_organization_questions(compute_data))

        # Type 4: Software availability questions
        pairs.extend(self._generate_software_availability_questions(software_data, compute_data))

        # Type 5: Cross-domain questions (GPU resources with specific software)
        pairs.extend(self._generate_cross_domain_questions(compute_data, software_data))

        return pairs

    def _generate_gpu_availability_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'which resources have X GPU?' questions."""
        pairs: ExtractionResult = []

        # Group resources by GPU type
        gpu_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for gpu_type in data.get("gpu_types", []):
                # Normalize GPU names
                normalized_gpu = self._normalize_gpu_name(gpu_type)
                if normalized_gpu:
                    gpu_resources[normalized_gpu].append({
                        "name": data["name"],
                        "resource_id": resource_id,
                    })

        # Generate questions for GPU types with 2+ resources
        for gpu_type, resources in gpu_resources.items():
            if len(resources) >= 2:
                question = f"Which ACCESS resources have {gpu_type} GPUs?"
                answer = self._format_resource_list_answer(
                    f"{gpu_type} GPUs are available on",
                    resources,
                    "compute-resources",
                )
                pairs.append(self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_gpu_{self._slugify(gpu_type)}",
                    domain="compute-resources",
                    resource_ids=[r["resource_id"] for r in resources],
                ))

        return pairs

    def _generate_feature_availability_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'which resources support X feature?' questions."""
        pairs: ExtractionResult = []

        # Features worth asking about
        interesting_features = [
            "GPU Acceleration",
            "GPU Computing",
            "Container Support",
            "Virtual Machine Support",
            "Cloud Computing Platform",
            "Interactive Computing",
            "AI/ML Optimized",
        ]

        # Group resources by feature
        feature_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for feature in data.get("features", []):
                if feature in interesting_features:
                    feature_resources[feature].append({
                        "name": data["name"],
                        "resource_id": resource_id,
                    })

        # Generate questions for features with 2+ resources
        for feature, resources in feature_resources.items():
            if len(resources) >= 2:
                question = f"Which ACCESS resources support {feature.lower()}?"
                answer = self._format_resource_list_answer(
                    f"{feature} is supported on",
                    resources,
                    "compute-resources",
                )
                pairs.append(self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_feat_{self._slugify(feature)}",
                    domain="compute-resources",
                    resource_ids=[r["resource_id"] for r in resources],
                ))

        return pairs

    def _generate_organization_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'what resources does X operate?' questions."""
        pairs: ExtractionResult = []

        # Group resources by organization
        org_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for org in data.get("organizations", []):
                if org:
                    org_resources[org].append({
                        "name": data["name"],
                        "resource_id": resource_id,
                    })

        # Generate questions for orgs with 2+ resources
        for org, resources in org_resources.items():
            if len(resources) >= 2:
                question = f"What ACCESS resources does {org} operate?"
                answer = self._format_resource_list_answer(
                    f"{org} operates",
                    resources,
                    "compute-resources",
                )
                pairs.append(self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_org_{self._slugify(org)}",
                    domain="compute-resources",
                    resource_ids=[r["resource_id"] for r in resources],
                ))

        return pairs

    def _generate_software_availability_questions(
        self, software_data: dict, compute_data: dict
    ) -> ExtractionResult:
        """Generate 'which resources have X software?' questions."""
        pairs: ExtractionResult = []

        for sw_name, sw_data in software_data.items():
            short_resource_names = sw_data.get("resources", [])
            if len(short_resource_names) < 2:
                continue

            # Map short resource names to full IDs and get display names
            resources = []
            for short_name in short_resource_names:
                full_id = self._resolve_resource_id(short_name, compute_data)
                if full_id and full_id in compute_data:
                    resources.append({
                        "name": compute_data[full_id]["name"],
                        "resource_id": full_id,
                    })
                else:
                    # Use short name as fallback
                    resources.append({
                        "name": short_name.title().replace("-", " "),
                        "resource_id": short_name,
                    })

            if len(resources) >= 2:
                display_name = sw_data.get("name", sw_name)
                question = f"Which ACCESS resources have {display_name} installed?"
                answer = self._format_resource_list_answer(
                    f"{display_name} is available on",
                    resources,
                    "compute-resources",
                    software_citation=sw_name,
                )
                pairs.append(self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_sw_{self._slugify(sw_name)}",
                    domain="software-discovery",
                    resource_ids=[r["resource_id"] for r in resources],
                    software_id=sw_name,
                ))

        return pairs

    def _generate_cross_domain_questions(
        self, compute_data: dict, software_data: dict
    ) -> ExtractionResult:
        """Generate questions combining compute + software data."""
        pairs: ExtractionResult = []

        # Find GPU resources (by full ID)
        gpu_resource_ids = {
            rid for rid, data in compute_data.items()
            if data.get("has_gpu")
        }

        if not gpu_resource_ids:
            return pairs

        # ML/AI software to check for cross-domain questions
        ml_software = ["pytorch", "tensorflow", "jax", "cuda", "cupy", "rapids"]

        for sw_name, sw_data in software_data.items():
            if sw_name.lower() not in ml_software:
                continue

            # Map short resource names to full IDs
            short_resources = sw_data.get("resources", [])
            sw_full_ids = set()
            for short_name in short_resources:
                full_id = self._resolve_resource_id(short_name, compute_data)
                if full_id:
                    sw_full_ids.add(full_id)

            # Find resources that have both GPU and this software
            gpu_with_software = gpu_resource_ids & sw_full_ids

            if len(gpu_with_software) < 2:
                continue

            # Build resource list
            resources = []
            for rid in gpu_with_software:
                if rid in compute_data:
                    resources.append({
                        "name": compute_data[rid]["name"],
                        "resource_id": rid,
                    })

            if len(resources) >= 2:
                display_name = sw_data.get("name", sw_name)
                question = f"Which GPU-enabled ACCESS resources have {display_name} installed?"
                answer = self._format_cross_domain_answer(
                    display_name, resources, sw_name
                )
                pairs.append(self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_gpu_sw_{self._slugify(sw_name)}",
                    domain="comparison",
                    resource_ids=[r["resource_id"] for r in resources],
                    software_id=sw_name,
                ))

        return pairs

    def _normalize_gpu_name(self, gpu_name: str) -> str:
        """Normalize GPU name for grouping."""
        if not gpu_name:
            return ""

        # Common normalizations
        name = gpu_name.upper().strip()

        # Extract model (A100, V100, H100, etc.)
        model_match = re.search(r'(A100|V100|H100|A40|A30|A10|RTX\s*\d+)', name)
        if model_match:
            return f"NVIDIA {model_match.group(1)}"

        return ""

    def _format_resource_list_answer(
        self,
        prefix: str,
        resources: list[dict],
        domain: str,
        software_citation: str | None = None,
    ) -> str:
        """Format an answer listing multiple resources with citations."""
        # Sort resources by name for consistency
        resources = sorted(resources, key=lambda r: r["name"])

        # Format resource names
        names = [r["name"] for r in resources]
        if len(names) == 2:
            resource_list = f"{names[0]} and {names[1]}"
        else:
            resource_list = ", ".join(names[:-1]) + f", and {names[-1]}"

        answer = f"{prefix} {resource_list}."

        # Add citations
        for r in resources:
            answer += f"\n\n<<SRC:{domain}:{r['resource_id']}>>"

        if software_citation:
            answer += f"\n\n<<SRC:software-discovery:{software_citation}>>"

        return answer

    def _format_cross_domain_answer(
        self,
        software_name: str,
        resources: list[dict],
        software_id: str,
    ) -> str:
        """Format a cross-domain answer with compute and software citations."""
        resources = sorted(resources, key=lambda r: r["name"])

        names = [r["name"] for r in resources]
        if len(names) == 2:
            resource_list = f"{names[0]} and {names[1]}"
        else:
            resource_list = ", ".join(names[:-1]) + f", and {names[-1]}"

        answer = (
            f"{software_name} is available on GPU-enabled ACCESS resources including "
            f"{resource_list}. These systems provide GPU acceleration for "
            f"{software_name} workloads."
        )

        # Add compute resource citations
        for r in resources:
            answer += f"\n\n<<SRC:compute-resources:{r['resource_id']}>>"

        # Add software citation
        answer += f"\n\n<<SRC:software-discovery:{software_id}>>"

        return answer

    def _create_pair(
        self,
        question: str,
        answer: str,
        pair_id: str,
        domain: str,
        resource_ids: list[str],
        software_id: str | None = None,
    ) -> QAPair:
        """Create a comparison Q&A pair."""
        # Build source_data for review
        source_data = {
            "comparison_type": "availability",
            "resources": resource_ids,
        }
        if software_id:
            source_data["software"] = software_id

        return QAPair.create(
            id=pair_id,
            question=question,
            answer=answer,
            source_ref=f"mcp://comparison/{pair_id}",
            domain=domain,
            complexity="moderate",
            source_data=source_data,
        )

    def _slugify(self, text: str) -> str:
        """Convert text to a URL-safe slug."""
        return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:30]
