"""Per-entity factoid Q&A generation from templates.

Generates single-fact Q&A pairs using Python string templates applied to
cleaned entity data. Zero LLM cost, zero hallucination risk, deterministic
output. Complements the LLM-driven comprehensive pairs from question_categories.py.

Each template has:
  - id: stable identifier (fq_ prefix)
  - question/answer: format strings with {field} placeholders
  - required_fields: fields that must be present and truthy for the template to fire
  - For boolean templates: bool_field + answer_yes/answer_no instead of answer
"""

import re

from ..models import ExtractionResult, QAPair

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _filter_strings(
    items: list, exclude_prefixes: list[str] | None = None
) -> list[str]:
    """Filter a list of strings, removing empty/whitespace-only entries.

    Optionally exclude entries starting with any of the given prefixes.
    """
    result = []
    for item in items:
        s = str(item).strip() if item else ""
        if not s:
            continue
        if exclude_prefixes and any(s.startswith(p) for p in exclude_prefixes):
            continue
        result.append(s)
    return result


_QUALITY_DEFECT_PATTERNS = [
    re.compile(r"\b(is|by|at|for|uses|has|in|on|the)\s*[.,;:]"),  # dangling preposition + punct
    re.compile(r",\s*[.,;:]"),  # dangling comma before punctuation
    re.compile(r"\(\s*\)"),  # empty parenthetical
    re.compile(r"  "),  # double space (sign of empty interpolation)
]


def _has_quality_defect(text: str) -> bool:
    """Check if formatted text has quality defects from empty interpolation."""
    # Strip citation before checking (citations are always appended)
    clean = re.sub(r"\n\n<<SRC:[^>]+>>$", "", text).strip()
    if len(clean) < 10:
        return True
    for pattern in _QUALITY_DEFECT_PATTERNS:
        if pattern.search(clean):
            return True
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-domain field preparers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These compute derived fields from the cleaned entity data so templates
# can reference them. Each returns a new dict with the original + derived fields.


def _prepare_compute_resources(data: dict) -> dict:
    d = dict(data)
    orgs = _filter_strings(data.get("organization_names", []))
    d["organization_names_str"] = ", ".join(orgs) if orgs else ""

    features = _filter_strings(data.get("feature_names", []), exclude_prefixes=["Unknown"])
    d["feature_names_str"] = ", ".join(features) if features else ""

    hw = data.get("hardware", {})
    gpu_names = _filter_strings([g.get("name", "") for g in hw.get("gpus", [])])
    d["gpu_types_str"] = ", ".join(gpu_names) if gpu_names else ""

    desc = data.get("description", "")
    if desc and desc.strip():
        first_sentence = desc.strip().split(". ")[0]
        if not first_sentence.endswith("."):
            first_sentence += "."
        d["description_short"] = first_sentence
    else:
        d["description_short"] = ""

    return d


def _prepare_software_discovery(data: dict) -> dict:
    d = dict(data)
    resources = data.get("available_on_resources", [])
    if isinstance(resources, list):
        names = []
        for r in resources:
            if isinstance(r, dict):
                names.append(r.get("name", r.get("resource_id", "")))
            else:
                names.append(str(r))
        names = _filter_strings(names)
        d["resource_count"] = len(names)
        d["resource_names_str"] = ", ".join(names) if names else ""
    else:
        d["resource_count"] = 0
        d["resource_names_str"] = ""

    versions = data.get("versions", [])
    if versions:
        if isinstance(versions[0], dict):
            ver = versions[0].get("version", "")
        else:
            ver = str(versions[0])
        d["latest_version"] = ver.strip() if ver and ver.strip() else ""
        d["version_count"] = len(versions)
    else:
        d["latest_version"] = ""
        d["version_count"] = 0

    return d


def _prepare_allocations(data: dict) -> dict:
    d = dict(data)
    resources = data.get("resources", [])
    resource_names = _filter_strings([r.get("name", "") for r in resources])
    d["resource_count"] = len(resource_names)
    d["resource_names_str"] = ", ".join(resource_names) if resource_names else ""
    return d


def _prepare_nsf_awards(data: dict) -> dict:
    d = dict(data)
    co_pis = _filter_strings(data.get("co_pis", []))
    d["co_pis"] = co_pis  # overwrite so bool_field check uses filtered list
    d["copi_count"] = len(co_pis)
    d["copis_str"] = ", ".join(co_pis) if co_pis else ""
    return d


def _prepare_affinity_groups(data: dict) -> dict:
    return dict(data)


_FIELD_PREPARERS = {
    "compute-resources": _prepare_compute_resources,
    "software-discovery": _prepare_software_discovery,
    "allocations": _prepare_allocations,
    "nsf-awards": _prepare_nsf_awards,
    "affinity-groups": _prepare_affinity_groups,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factoid templates per domain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FACTOID_TEMPLATES: dict[str, list[dict]] = {
    # ── Compute Resources (7 templates) ──────────────────────────────
    "compute-resources": [
        {
            "id": "fq_resource_type",
            "question": "What type of resource is {name}?",
            "answer": "{name} is a {resourceType} resource.",
            "required_fields": ["name", "resourceType"],
        },
        {
            "id": "fq_operator",
            "question": "Who operates {name}?",
            "answer": "{name} is operated by {organization_names_str}.",
            "required_fields": ["name", "organization_names_str"],
        },
        {
            "id": "fq_has_gpu",
            "question": "Does {name} have GPUs?",
            "bool_field": "hasGpu",
            "answer_yes": "Yes, {name} has GPUs available.",
            "answer_no": "No, {name} does not have GPUs.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_gpu_model",
            "question": "What GPU models does {name} have?",
            "answer": "{name} has {gpu_types_str}.",
            "required_fields": ["name", "gpu_types_str"],
        },
        {
            "id": "fq_allocated",
            "question": "Is {name} ACCESS-allocated?",
            "bool_field": "accessAllocated",
            "answer_yes": "Yes, {name} is an ACCESS-allocated resource.",
            "answer_no": "No, {name} is not ACCESS-allocated.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_features",
            "question": "What features does {name} support?",
            "answer": "{name} supports {feature_names_str}.",
            "required_fields": ["name", "feature_names_str"],
        },
        {
            "id": "fq_description",
            "question": "What is {name}?",
            "answer": "{description_short}",
            "required_fields": ["name", "description_short"],
        },
    ],
    # ── Software Discovery (7 templates) ─────────────────────────────
    "software-discovery": [
        {
            "id": "fq_software_type",
            "question": "What type of software is {name}?",
            "answer": "{name} is a {software_type}.",
            "required_fields": ["name", "software_type"],
        },
        {
            "id": "fq_resource_count",
            "question": "How many ACCESS resources have {name} installed?",
            "answer": "{name} is available on {resource_count} ACCESS resources.",
            "required_fields": ["name", "resource_count"],
        },
        {
            "id": "fq_resource_list",
            "question": "Which ACCESS systems have {name}?",
            "answer": "{name} is available on {resource_names_str}.",
            "required_fields": ["name", "resource_names_str"],
        },
        {
            "id": "fq_latest_version",
            "question": "What is the latest version of {name} on ACCESS?",
            "answer": "The latest version of {name} on ACCESS is {latest_version}.",
            "required_fields": ["name", "latest_version"],
        },
        {
            "id": "fq_version_count",
            "question": "How many versions of {name} are available on ACCESS?",
            "answer": "There are {version_count} versions of {name} available on ACCESS.",
            "required_fields": ["name", "version_count"],
        },
        {
            "id": "fq_has_example",
            "question": "Is there a usage example for {name} on ACCESS?",
            "bool_field": "example_use",
            "answer_yes": "Yes, there is a usage example available for {name} on ACCESS.",
            "answer_no": "No, there is no usage example available for {name} on ACCESS.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_description",
            "question": "What is {name}?",
            "answer": "{description}",
            "required_fields": ["name", "description"],
        },
    ],
    # ── Allocations (8 templates) ────────────────────────────────────
    "allocations": [
        {
            "id": "fq_pi_name",
            "question": "Who is the PI for {title}?",
            "answer": "The PI for {title} is {pi}.",
            "required_fields": ["title", "pi"],
        },
        {
            "id": "fq_institution",
            "question": "What institution leads {title}?",
            "answer": "{title} is led by researchers at {institution}.",
            "required_fields": ["title", "institution"],
        },
        {
            "id": "fq_field",
            "question": "What field of science is {title} in?",
            "answer": "{title} is in the field of {field_of_science}.",
            "required_fields": ["title", "field_of_science"],
        },
        {
            "id": "fq_start_date",
            "question": "When does {title} start?",
            "answer": "{title} starts on {beginDate}.",
            "required_fields": ["title", "beginDate"],
        },
        {
            "id": "fq_end_date",
            "question": "When does {title} end?",
            "answer": "{title} ends on {endDate}.",
            "required_fields": ["title", "endDate"],
        },
        {
            "id": "fq_alloc_type",
            "question": "What type of allocation is {title}?",
            "answer": "{title} is a {allocation_type} allocation.",
            "required_fields": ["title", "allocation_type"],
        },
        {
            "id": "fq_resource_count",
            "question": "How many resources does {title} use?",
            "answer": "{title} uses {resource_count} resources.",
            "required_fields": ["title", "resource_count"],
        },
        {
            "id": "fq_resource_list",
            "question": "What resources does {title} use?",
            "answer": "{title} uses {resource_names_str}.",
            "required_fields": ["title", "resource_names_str"],
        },
    ],
    # ── NSF Awards (8 templates) ─────────────────────────────────────
    "nsf-awards": [
        {
            "id": "fq_pi_name",
            "question": 'Who is the PI for the NSF award "{title}"?',
            "answer": 'The PI for "{title}" is {principal_investigator}.',
            "required_fields": ["title", "principal_investigator"],
        },
        {
            "id": "fq_institution",
            "question": 'What institution is the NSF award "{title}" at?',
            "answer": '"{title}" is at {institution}.',
            "required_fields": ["title", "institution"],
        },
        {
            "id": "fq_amount",
            "question": 'How much funding was awarded for "{title}"?',
            "answer": '"{title}" was awarded {total_intended_award}.',
            "required_fields": ["title", "total_intended_award"],
        },
        {
            "id": "fq_program",
            "question": 'What NSF program funds "{title}"?',
            "answer": '"{title}" is funded by the {primary_program} program.',
            "required_fields": ["title", "primary_program"],
        },
        {
            "id": "fq_start_date",
            "question": 'When does the NSF award "{title}" start?',
            "answer": '"{title}" starts on {startDate}.',
            "required_fields": ["title", "startDate"],
        },
        {
            "id": "fq_end_date",
            "question": 'When does the NSF award "{title}" end?',
            "answer": '"{title}" ends on {endDate}.',
            "required_fields": ["title", "endDate"],
        },
        {
            "id": "fq_has_copis",
            "question": 'Does "{title}" have co-PIs?',
            "bool_field": "co_pis",
            "answer_yes": 'Yes, "{title}" has {copi_count} co-PI(s): {copis_str}.',
            "answer_no": 'No, "{title}" does not have any co-PIs.',
            "required_fields": ["title"],
        },
        {
            "id": "fq_award_number",
            "question": 'What is the award number for "{title}"?',
            "answer": 'The award number for "{title}" is {award_number}.',
            "required_fields": ["title", "award_number"],
        },
    ],
    # ── Affinity Groups (6 templates) ────────────────────────────────
    "affinity-groups": [
        {
            "id": "fq_coordinator",
            "question": "Who coordinates the {name} affinity group?",
            "answer": "The {name} affinity group is coordinated by {coordinator}.",
            "required_fields": ["name", "coordinator"],
        },
        {
            "id": "fq_category",
            "question": "What category is the {name} affinity group in?",
            "answer": "The {name} affinity group is in the {category} category.",
            "required_fields": ["name", "category"],
        },
        {
            "id": "fq_has_slack",
            "question": "Does the {name} affinity group have a Slack channel?",
            "bool_field": "slack_link",
            "answer_yes": "Yes, the {name} affinity group has a Slack channel.",
            "answer_no": "No, the {name} affinity group does not have a Slack channel.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_has_events",
            "question": "Does the {name} affinity group host events?",
            "bool_field": "upcoming_events",
            "answer_yes": "Yes, the {name} affinity group hosts events.",
            "answer_no": "No, the {name} affinity group does not currently have events listed.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_has_kb",
            "question": "Does the {name} affinity group have a knowledge base?",
            "bool_field": "knowledge_base_topics",
            "answer_yes": "Yes, the {name} affinity group maintains a knowledge base.",
            "answer_no": "No, the {name} affinity group does not have a knowledge base.",
            "required_fields": ["name"],
        },
        {
            "id": "fq_support",
            "question": "Where can I get support from the {name} affinity group?",
            "answer": "You can get support from the {name} affinity group at {support_url}.",
            "required_fields": ["name", "support_url"],
        },
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_factoid_pairs(
    domain: str, entity_id: str, entity_data: dict
) -> ExtractionResult:
    """Generate factoid Q&A pairs from templates for a single entity.

    Args:
        domain: Domain name (e.g., "compute-resources")
        entity_id: Entity identifier (e.g., "delta.ncsa.access-ci.org")
        entity_data: Cleaned entity data dict (same data the LLM sees)

    Returns:
        List of QAPair objects with granularity="factoid"
    """
    templates = FACTOID_TEMPLATES.get(domain, [])
    if not templates:
        return []

    preparer = _FIELD_PREPARERS.get(domain, lambda d: dict(d))
    prepared = preparer(entity_data)

    citation = f"\n\n<<SRC:{domain}:{entity_id}>>"
    source_ref = _build_source_ref(domain, entity_id)

    pairs: ExtractionResult = []

    for template in templates:
        pair = _apply_template(template, prepared, domain, entity_id, citation, source_ref)
        if pair is not None:
            pairs.append(pair)

    return pairs


def _apply_template(
    template: dict,
    data: dict,
    domain: str,
    entity_id: str,
    citation: str,
    source_ref: str,
) -> QAPair | None:
    """Apply a single template to entity data. Returns None if required fields missing."""
    # Check required fields
    for field in template["required_fields"]:
        value = data.get(field)
        if not value and value != 0 and value is not False:
            return None

    try:
        question = template["question"].format(**data)

        if "bool_field" in template:
            # Boolean template
            field_value = data.get(template["bool_field"])
            # Truthy check: non-empty list, non-empty string, True
            is_true = bool(field_value)
            answer_template = template["answer_yes"] if is_true else template["answer_no"]
            answer = answer_template.format(**data)
        else:
            answer = template["answer"].format(**data)

        # Post-format quality check — catch empty interpolation artifacts
        if _has_quality_defect(answer):
            return None

        answer += citation

        pair_id = f"{domain}_{entity_id}_{template['id']}"

        return QAPair.create(
            id=pair_id,
            question=question,
            answer=answer,
            source_ref=source_ref,
            domain=domain,
            complexity="simple",
            granularity="factoid",
            source_data=None,
        )
    except (KeyError, IndexError):
        return None


def _build_source_ref(domain: str, entity_id: str) -> str:
    """Build the source_ref URI for a given domain and entity."""
    entity_type = {
        "compute-resources": "resources",
        "software-discovery": "software",
        "allocations": "projects",
        "nsf-awards": "awards",
        "affinity-groups": "groups",
    }.get(domain, "entities")
    return f"mcp://{domain}/{entity_type}/{entity_id}"
