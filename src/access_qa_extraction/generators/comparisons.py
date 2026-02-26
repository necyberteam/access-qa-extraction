"""Generate cross-entity comparison Q&A pairs.

This module generates comparison questions that span multiple entities,
such as "Which ACCESS resources have A100 GPUs?" or "What allocation
projects study Computer Science?"

All answers are built programmatically from extracted data to avoid hallucination.
Covers all 5 domains: compute-resources, software-discovery, allocations,
nsf-awards, and affinity-groups.
"""

import re
from collections import defaultdict

from ..models import ExtractionResult, QAPair

# Minimum group size to generate a comparison question
MIN_GROUP_SIZE_SMALL = 2  # small domains: compute-resources, affinity-groups
MIN_GROUP_SIZE_LARGE = 3  # large domains: allocations, nsf-awards


class ComparisonGenerator:
    """Generate cross-entity comparison Q&A pairs."""

    def __init__(self):
        self._resource_name_to_id: dict[str, str] = {}

    def _build_resource_lookup(self, compute_data: dict) -> None:
        """Build a lookup from short resource names to full IDs."""
        for resource_id, data in compute_data.items():
            name = data.get("name", "").lower()
            self._resource_name_to_id[name] = resource_id
            short_name = resource_id.split(".")[0].lower()
            self._resource_name_to_id[short_name] = resource_id
            if "-" in name:
                self._resource_name_to_id[name.replace("-", "")] = resource_id

    def _resolve_resource_id(self, short_name: str, compute_data: dict) -> str | None:
        """Resolve a short resource name to its full ID."""
        short_lower = short_name.lower().strip()

        if short_lower in self._resource_name_to_id:
            return self._resource_name_to_id[short_lower]

        no_hyphen = short_lower.replace("-", "")
        if no_hyphen in self._resource_name_to_id:
            return self._resource_name_to_id[no_hyphen]

        for resource_id in compute_data.keys():
            if resource_id.lower().startswith(short_lower):
                return resource_id

        return None

    def generate(
        self,
        compute_data: dict,
        software_data: dict,
        allocations_data: dict | None = None,
        nsf_awards_data: dict | None = None,
        affinity_groups_data: dict | None = None,
    ) -> ExtractionResult:
        """Generate comparison Q&A pairs from extracted data.

        Args:
            compute_data: Normalized compute resource data from extraction
            software_data: Normalized software data from extraction
            allocations_data: Normalized allocation project data
            nsf_awards_data: Normalized NSF award data
            affinity_groups_data: Normalized affinity group data

        Returns:
            List of comparison Q&A pairs
        """
        pairs: ExtractionResult = []

        self._build_resource_lookup(compute_data)

        # Compute-resources comparisons
        pairs.extend(self._generate_gpu_availability_questions(compute_data))
        pairs.extend(self._generate_feature_availability_questions(compute_data))
        pairs.extend(self._generate_organization_questions(compute_data))

        # Software-discovery comparisons
        pairs.extend(self._generate_software_availability_questions(software_data, compute_data))

        # Cross-domain: compute + software
        pairs.extend(self._generate_cross_domain_questions(compute_data, software_data))

        # Allocations comparisons
        if allocations_data:
            pairs.extend(self._generate_allocations_by_fos(allocations_data))
            pairs.extend(self._generate_allocations_by_institution(allocations_data))
            pairs.extend(self._generate_allocations_by_resource(allocations_data, compute_data))

        # NSF Awards comparisons
        if nsf_awards_data:
            pairs.extend(self._generate_nsf_by_program(nsf_awards_data))
            pairs.extend(self._generate_nsf_by_institution(nsf_awards_data))

        # Affinity Groups comparisons
        if affinity_groups_data:
            pairs.extend(self._generate_affinity_by_category(affinity_groups_data))

        return pairs

    # ── Compute-resources comparisons ────────────────────────────────

    def _generate_gpu_availability_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'which resources have X GPU?' questions."""
        pairs: ExtractionResult = []

        gpu_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for gpu_type in data.get("gpu_types", []):
                normalized_gpu = self._normalize_gpu_name(gpu_type)
                if normalized_gpu:
                    gpu_resources[normalized_gpu].append(
                        {
                            "name": data["name"],
                            "resource_id": resource_id,
                        }
                    )

        for gpu_type, resources in gpu_resources.items():
            if len(resources) >= MIN_GROUP_SIZE_SMALL:
                question = f"Which ACCESS resources have {gpu_type} GPUs?"
                answer = self._format_resource_list_answer(
                    f"{gpu_type} GPUs are available on",
                    resources,
                    "compute-resources",
                )
                pairs.append(
                    self._create_pair(
                        question=question,
                        answer=answer,
                        pair_id=f"cmp_gpu_{self._slugify(gpu_type)}",
                        domain="compute-resources",
                        resource_ids=[r["resource_id"] for r in resources],
                    )
                )

        return pairs

    def _generate_feature_availability_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'which resources support X feature?' questions."""
        pairs: ExtractionResult = []

        interesting_features = [
            "GPU Acceleration",
            "GPU Computing",
            "Container Support",
            "Virtual Machine Support",
            "Cloud Computing Platform",
            "Interactive Computing",
            "AI/ML Optimized",
        ]

        feature_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for feature in data.get("features", []):
                if feature in interesting_features:
                    feature_resources[feature].append(
                        {
                            "name": data["name"],
                            "resource_id": resource_id,
                        }
                    )

        for feature, resources in feature_resources.items():
            if len(resources) >= MIN_GROUP_SIZE_SMALL:
                question = f"Which ACCESS resources support {feature.lower()}?"
                answer = self._format_resource_list_answer(
                    f"{feature} is supported on",
                    resources,
                    "compute-resources",
                )
                pairs.append(
                    self._create_pair(
                        question=question,
                        answer=answer,
                        pair_id=f"cmp_feat_{self._slugify(feature)}",
                        domain="compute-resources",
                        resource_ids=[r["resource_id"] for r in resources],
                    )
                )

        return pairs

    def _generate_organization_questions(self, compute_data: dict) -> ExtractionResult:
        """Generate 'what resources does X operate?' questions."""
        pairs: ExtractionResult = []

        org_resources: dict[str, list[dict]] = defaultdict(list)
        for resource_id, data in compute_data.items():
            for org in data.get("organizations", []):
                if org:
                    org_resources[org].append(
                        {
                            "name": data["name"],
                            "resource_id": resource_id,
                        }
                    )

        for org, resources in org_resources.items():
            if len(resources) >= MIN_GROUP_SIZE_SMALL:
                question = f"What ACCESS resources does {org} operate?"
                answer = self._format_resource_list_answer(
                    f"{org} operates",
                    resources,
                    "compute-resources",
                )
                pairs.append(
                    self._create_pair(
                        question=question,
                        answer=answer,
                        pair_id=f"cmp_org_{self._slugify(org)}",
                        domain="compute-resources",
                        resource_ids=[r["resource_id"] for r in resources],
                    )
                )

        return pairs

    # ── Software-discovery comparisons ───────────────────────────────

    def _generate_software_availability_questions(
        self, software_data: dict, compute_data: dict
    ) -> ExtractionResult:
        """Generate 'which resources have X software?' questions."""
        pairs: ExtractionResult = []

        for sw_name, sw_data in software_data.items():
            short_resource_names = sw_data.get("resources", [])
            if len(short_resource_names) < 2:
                continue

            resources = []
            for short_name in short_resource_names:
                full_id = self._resolve_resource_id(short_name, compute_data)
                if full_id and full_id in compute_data:
                    resources.append(
                        {
                            "name": compute_data[full_id]["name"],
                            "resource_id": full_id,
                        }
                    )
                else:
                    resources.append(
                        {
                            "name": short_name.title().replace("-", " "),
                            "resource_id": short_name,
                        }
                    )

            if len(resources) >= MIN_GROUP_SIZE_SMALL:
                display_name = sw_data.get("name", sw_name)
                question = f"Which ACCESS resources have {display_name} installed?"
                answer = self._format_resource_list_answer(
                    f"{display_name} is available on",
                    resources,
                    "compute-resources",
                    software_citation=sw_name,
                )
                pairs.append(
                    self._create_pair(
                        question=question,
                        answer=answer,
                        pair_id=f"cmp_sw_{self._slugify(sw_name)}",
                        domain="software-discovery",
                        resource_ids=[r["resource_id"] for r in resources],
                        software_id=sw_name,
                    )
                )

        return pairs

    def _generate_cross_domain_questions(
        self, compute_data: dict, software_data: dict
    ) -> ExtractionResult:
        """Generate questions combining compute + software data."""
        pairs: ExtractionResult = []

        gpu_resource_ids = {rid for rid, data in compute_data.items() if data.get("has_gpu")}

        if not gpu_resource_ids:
            return pairs

        ml_software = ["pytorch", "tensorflow", "jax", "cuda", "cupy", "rapids"]

        for sw_name, sw_data in software_data.items():
            if sw_name.lower() not in ml_software:
                continue

            short_resources = sw_data.get("resources", [])
            sw_full_ids = set()
            for short_name in short_resources:
                full_id = self._resolve_resource_id(short_name, compute_data)
                if full_id:
                    sw_full_ids.add(full_id)

            gpu_with_software = gpu_resource_ids & sw_full_ids

            if len(gpu_with_software) < MIN_GROUP_SIZE_SMALL:
                continue

            resources = []
            for rid in gpu_with_software:
                if rid in compute_data:
                    resources.append(
                        {
                            "name": compute_data[rid]["name"],
                            "resource_id": rid,
                        }
                    )

            if len(resources) >= MIN_GROUP_SIZE_SMALL:
                display_name = sw_data.get("name", sw_name)
                question = f"Which GPU-enabled ACCESS resources have {display_name} installed?"
                answer = self._format_cross_domain_answer(display_name, resources, sw_name)
                pairs.append(
                    self._create_pair(
                        question=question,
                        answer=answer,
                        pair_id=f"cmp_gpu_sw_{self._slugify(sw_name)}",
                        domain="comparison",
                        resource_ids=[r["resource_id"] for r in resources],
                        software_id=sw_name,
                    )
                )

        return pairs

    # ── Allocations comparisons ──────────────────────────────────────

    def _generate_allocations_by_fos(self, allocations_data: dict) -> ExtractionResult:
        """Generate 'what projects study X field of science?' questions."""
        return self._generate_grouped_comparison(
            data=allocations_data,
            group_field="fos",
            question_template="What ACCESS allocation projects study {group}?",
            answer_prefix="There are {count} allocation projects in {group}",
            pair_id_prefix="cmp_alloc_fos",
            domain="allocations",
            min_group_size=MIN_GROUP_SIZE_LARGE,
        )

    def _generate_allocations_by_institution(self, allocations_data: dict) -> ExtractionResult:
        """Generate 'what projects are at X institution?' questions."""
        return self._generate_grouped_comparison(
            data=allocations_data,
            group_field="institution",
            question_template=(
                "What ACCESS allocation projects are led by researchers at {group}?"
            ),
            answer_prefix="There are {count} allocation projects at {group}",
            pair_id_prefix="cmp_alloc_inst",
            domain="allocations",
            min_group_size=MIN_GROUP_SIZE_LARGE,
        )

    def _generate_allocations_by_resource(
        self, allocations_data: dict, compute_data: dict
    ) -> ExtractionResult:
        """Generate 'what projects use X resource?' questions."""
        pairs: ExtractionResult = []

        resource_projects: dict[str, list[dict]] = defaultdict(list)
        for project_id, data in allocations_data.items():
            for resource_name in data.get("resource_names", []):
                if resource_name:
                    resource_projects[resource_name].append(
                        {
                            "name": data.get("name", ""),
                            "entity_id": project_id,
                        }
                    )

        for resource_name, projects in resource_projects.items():
            if len(projects) < MIN_GROUP_SIZE_LARGE:
                continue

            question = f"What allocation projects use {resource_name}?"
            answer = self._format_count_sample_answer(
                f"There are {len(projects)} allocation projects using {resource_name}",
                projects,
                "allocations",
            )

            pairs.append(
                self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_alloc_res_{self._slugify(resource_name)}",
                    domain="allocations",
                    resource_ids=[p["entity_id"] for p in projects[:5]],
                )
            )

        return pairs

    # ── NSF Awards comparisons ───────────────────────────────────────

    def _generate_nsf_by_program(self, nsf_awards_data: dict) -> ExtractionResult:
        """Generate 'what awards are funded by X program?' questions."""
        return self._generate_grouped_comparison(
            data=nsf_awards_data,
            group_field="fund_program_name",
            question_template="What NSF awards are funded by the {group} program?",
            answer_prefix="There are {count} NSF awards funded by {group}",
            pair_id_prefix="cmp_nsf_prog",
            domain="nsf-awards",
            min_group_size=MIN_GROUP_SIZE_LARGE,
        )

    def _generate_nsf_by_institution(self, nsf_awards_data: dict) -> ExtractionResult:
        """Generate 'what awards are at X institution?' questions."""
        return self._generate_grouped_comparison(
            data=nsf_awards_data,
            group_field="institution",
            question_template="What NSF awards are at {group}?",
            answer_prefix="There are {count} NSF awards at {group}",
            pair_id_prefix="cmp_nsf_inst",
            domain="nsf-awards",
            min_group_size=MIN_GROUP_SIZE_LARGE,
        )

    # ── Affinity Groups comparisons ──────────────────────────────────

    def _generate_affinity_by_category(self, affinity_groups_data: dict) -> ExtractionResult:
        """Generate 'what groups are in X category?' questions."""
        pairs: ExtractionResult = []

        category_groups: dict[str, list[dict]] = defaultdict(list)
        for group_id, data in affinity_groups_data.items():
            category = data.get("category", "")
            if category:
                category_groups[category].append(
                    {
                        "name": data.get("name", ""),
                        "entity_id": group_id,
                    }
                )

        for category, groups in category_groups.items():
            if len(groups) < MIN_GROUP_SIZE_SMALL:
                continue

            groups_sorted = sorted(groups, key=lambda g: g["name"])
            names = [g["name"] for g in groups_sorted]
            if len(names) == 2:
                name_list = f"{names[0]} and {names[1]}"
            else:
                name_list = ", ".join(names[:-1]) + f", and {names[-1]}"

            question = f"What ACCESS affinity groups are in the {category} category?"
            answer = f"The {category} affinity groups include {name_list}."
            for g in groups_sorted:
                answer += f"\n\n<<SRC:affinity-groups:{g['entity_id']}>>"

            pairs.append(
                self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"cmp_ag_cat_{self._slugify(category)}",
                    domain="affinity-groups",
                    resource_ids=[g["entity_id"] for g in groups_sorted],
                )
            )

        return pairs

    # ── Generic grouping helper ──────────────────────────────────────

    def _generate_grouped_comparison(
        self,
        data: dict,
        group_field: str,
        question_template: str,
        answer_prefix: str,
        pair_id_prefix: str,
        domain: str,
        min_group_size: int = MIN_GROUP_SIZE_LARGE,
    ) -> ExtractionResult:
        """Generate comparison Q&A by grouping entities on a shared field.

        Uses count+sample format for large groups.
        """
        pairs: ExtractionResult = []

        groups: dict[str, list[dict]] = defaultdict(list)
        for entity_id, entity_data in data.items():
            value = entity_data.get(group_field, "")
            if value:
                groups[value].append(
                    {
                        "name": entity_data.get("name", ""),
                        "entity_id": entity_id,
                    }
                )

        for group_value, entities in groups.items():
            if len(entities) < min_group_size:
                continue

            question = question_template.format(group=group_value)
            answer = self._format_count_sample_answer(
                answer_prefix.format(count=len(entities), group=group_value),
                entities,
                domain,
            )

            pairs.append(
                self._create_pair(
                    question=question,
                    answer=answer,
                    pair_id=f"{pair_id_prefix}_{self._slugify(group_value)}",
                    domain=domain,
                    resource_ids=[e["entity_id"] for e in entities[:5]],
                )
            )

        return pairs

    # ── Answer formatting ────────────────────────────────────────────

    def _normalize_gpu_name(self, gpu_name: str) -> str:
        """Normalize GPU name for grouping."""
        if not gpu_name:
            return ""
        name = gpu_name.upper().strip()
        model_match = re.search(r"(A100|V100|H100|A40|A30|A10|RTX\s*\d+)", name)
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
        resources = sorted(resources, key=lambda r: r["name"])
        names = [r["name"] for r in resources]
        if len(names) == 2:
            resource_list = f"{names[0]} and {names[1]}"
        else:
            resource_list = ", ".join(names[:-1]) + f", and {names[-1]}"

        answer = f"{prefix} {resource_list}."
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
        for r in resources:
            answer += f"\n\n<<SRC:compute-resources:{r['resource_id']}>>"
        answer += f"\n\n<<SRC:software-discovery:{software_id}>>"
        return answer

    def _format_count_sample_answer(
        self,
        prefix: str,
        entities: list[dict],
        domain: str,
        max_sample: int = 5,
    ) -> str:
        """Format 'There are N items, including A, B, C.' with citations for sample."""
        sorted_entities = sorted(entities, key=lambda e: e["name"])
        sample = sorted_entities[:max_sample]

        sample_names = [e["name"] for e in sample]
        if len(sample_names) == 1:
            name_list = sample_names[0]
        elif len(sample_names) == 2:
            name_list = f"{sample_names[0]} and {sample_names[1]}"
        else:
            name_list = ", ".join(sample_names[:-1]) + f", and {sample_names[-1]}"

        if len(entities) > max_sample:
            answer = f"{prefix}, including {name_list}."
        else:
            answer = f"{prefix}: {name_list}."

        for e in sample:
            answer += f"\n\n<<SRC:{domain}:{e['entity_id']}>>"
        return answer

    # ── Pair creation ────────────────────────────────────────────────

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
            granularity="comparison",
            source_data=source_data,
        )

    def _slugify(self, text: str) -> str:
        """Convert text to a URL-safe slug."""
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30]
