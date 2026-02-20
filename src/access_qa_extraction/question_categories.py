"""Prompt templates and per-domain guidance for the two-shot extraction pipeline.

Two-shot pipeline:
  1. Battery prompt — one Q&A pair per field group (guaranteed coverage)
  2. Discovery prompt — additional pairs for what the battery missed

Supporting data:
  - DOMAIN_LABELS: display names and entity types per domain
  - DOMAIN_NOTES: optional per-domain LLM guidance on data quirks
  - FIELD_GUIDANCE: per-domain field groups mapping data fields to required Q&A pairs
"""

DOMAIN_LABELS = {
    "compute-resources": {"display": "compute resources", "entity_type": "HPC system"},
    "software-discovery": {"display": "software catalog", "entity_type": "software package"},
    "allocations": {"display": "allocation projects", "entity_type": "allocation project"},
    "nsf-awards": {"display": "NSF awards", "entity_type": "NSF award"},
    "affinity-groups": {"display": "affinity groups", "entity_type": "community group"},
}

# Optional per-domain notes appended to the system prompt.
# Use these to guide the LLM on domain-specific data quirks.
DOMAIN_NOTES: dict[str, str] = {
    "nsf-awards": (
        "- If a `primary_program_budget_code` field is present, it is an internal NSF budget "
        "code (e.g., '01002526DB NSF RESEARCH & RELATED ACTIVIT') with limited meaning. "
        "Use `fund_program_name` for the human-readable program name instead."
    ),
    "affinity-groups": (
        "- Contact emails may appear in obfuscated form (e.g., '[at]', '[dot]'). "
        "Present them as standard email addresses (e.g., support@example.edu)."
    ),
}

FIELD_GUIDANCE: dict[str, list[dict[str, str]]] = {
    "compute-resources": [
        {
            "fields": "name, description, resourceType",
            "instruction": "Overview — what is this resource and what is it designed for?",
        },
        {
            "fields": "organization_names",
            "instruction": "Organization — who operates this resource?",
        },
        {
            "fields": "hardware.gpus",
            "instruction": "GPU hardware — what GPU models, counts, and memory are available?",
            "condition": "only if hasGpu is true and hardware.gpus is non-empty",
        },
        {
            "fields": "hardware.compute_nodes",
            "instruction": "CPU/compute hardware — what compute nodes are available?",
            "condition": "only if hardware.compute_nodes is non-empty",
        },
        {
            "fields": "hardware.storage",
            "instruction": "Storage — what storage systems and capacities are available?",
            "condition": "only if hardware.storage is non-empty",
        },
        {
            "fields": "feature_names",
            "instruction": "Features & capabilities — what does this resource support?",
        },
        {
            "fields": "accessAllocated",
            "instruction": "Access — how can a researcher get access to this resource?",
        },
    ],
    "software-discovery": [
        {
            "fields": "name, description, software_type",
            "instruction": "Overview — what is this software and what type of tool is it?",
        },
        {
            "fields": "available_on_resources",
            "instruction": "Availability — which ACCESS resources have this software installed?",
        },
        {
            "fields": "versions",
            "instruction": "Versions — what versions are available?",
        },
        {
            "fields": "example_use",
            "instruction": "Usage — how do I load and use this software?",
            "condition": "only if example_use is present and non-empty",
        },
        {
            "fields": "research_area, research_field, tags",
            "instruction": "Research use — what disciplines or research areas use this software?",
        },
        {
            "fields": "core_features",
            "instruction": "Core features — what are the key capabilities of this software?",
            "condition": "only if core_features is present and non-empty",
        },
    ],
    "allocations": [
        {
            "fields": "title, abstract, allocation_type",
            "instruction": "Overview — what is this project about?",
        },
        {
            "fields": "pi, institution",
            "instruction": "PI & institution — who leads this project and where?",
        },
        {
            "fields": "resources",
            "instruction": (
                "Resources — what compute resources are allocated? "
                "Include resource names, units, and allocation amounts."
            ),
        },
        {
            "fields": "field_of_science",
            "instruction": "Field of science — what discipline does this project fall under?",
        },
        {
            "fields": "beginDate, endDate",
            "instruction": "Timeline — when does this allocation start and end?",
        },
    ],
    "nsf-awards": [
        {
            "fields": "title, abstract",
            "instruction": "Overview — what is this award about?",
        },
        {
            "fields": "principal_investigator, institution, co_pis",
            "instruction": "People — who is the PI, what institution, and who are the co-PIs?",
            "condition": "include co_pis only if present",
        },
        {
            "fields": "total_intended_award, totalAwardedToDate",
            "instruction": "Funding — how much was awarded?",
        },
        {
            "fields": "fund_program_name",
            "instruction": "Program — what NSF program funds this award?",
        },
        {
            "fields": "startDate, endDate",
            "instruction": "Timeline — when does this award start and end?",
        },
    ],
    "affinity-groups": [
        {
            "fields": "name, description, category",
            "instruction": "Overview — what is this group and what community does it serve?",
        },
        {
            "fields": "coordinator",
            "instruction": "Coordinator — who coordinates this group?",
        },
        {
            "fields": "slack_link, support_url, ask_ci_forum",
            "instruction": "How to engage — how can someone join or contact this group?",
        },
        {
            "fields": "upcoming_events",
            "instruction": "Events — what events does this group offer?",
            "condition": "only if upcoming_events is present and non-empty",
        },
        {
            "fields": "knowledge_base_topics",
            "instruction": "Knowledge base — what resources or articles does this group maintain?",
            "condition": "only if knowledge_base_topics is present and non-empty",
        },
    ],
}


def format_field_guidance_block(domain: str) -> str:
    """Format field guidance into a numbered list for the system prompt."""
    guidance = FIELD_GUIDANCE[domain]
    lines = []
    n = 1
    for g in guidance:
        condition = ""
        if "condition" in g:
            condition = f" *(skip if not applicable: {g['condition']})*"
        lines.append(f"{n}. **{g['instruction']}**{condition}")
        lines.append(f"   Data fields: `{g['fields']}`")
        n += 1
    return "\n".join(lines)


# --- Battery prompt (call 1): one pair per field group ---

BATTERY_SYSTEM_PROMPT_TEMPLATE = """You are a Q&A pair generator for ACCESS-CI {domain_display_name}.

You will receive structured data about a single {entity_type}. Generate exactly one
Q&A pair for each of the field groups listed below. Skip a group ONLY if the data
genuinely does not contain information for it.

## Required field groups

{field_guidance_block}

## Rules

1. Output a JSON array. Each element has two fields: "question", "answer".
2. Only use information present in the provided data. Do not infer or fabricate facts.
3. Questions should be natural — the kind a researcher would actually type into a search box.
4. Answers should be concise but complete. Include specific numbers, names, and dates
   when the data provides them.
5. Every answer MUST end with the citation marker provided in the user message.
6. Generate exactly one pair per field group — no more, no less (unless skipping).
{domain_notes}
## Output format

```json
[
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}},
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}}
]
```"""


# --- Discovery prompt (call 2): find what the battery missed ---

DISCOVERY_SYSTEM_PROMPT_TEMPLATE = """You are a Q&A pair generator for ACCESS-CI {domain_display_name}.

You will receive structured data about a single {entity_type}, along with Q&A pairs
that have already been generated for this entity. Your job is to find what's
**missing or interesting** that the existing pairs didn't capture.

## Already covered

The following Q&A pairs have already been generated:

{existing_pairs_block}

## Your task

Look at the source data and identify additional details worth surfacing that the
existing pairs missed:
- Notable partnerships, collaborations, or multi-institution arrangements
- Unique technologies, architectures, or methodologies
- Interdisciplinary applications or unusual use cases
- Specific numbers, capacities, or performance characteristics
- Anything distinctive about this entity that a researcher would want to know

Focus on what's genuinely new or interesting. If a topic is already well-covered
in the existing pairs, skip it rather than rephrasing the same information.

Generate additional Q&A pairs for these discoveries. If the existing pairs already
cover everything interesting, output an empty array `[]`.

## Rules

1. Output a JSON array. Each element has two fields: "question", "answer".
2. Only use information present in the provided data. Do not infer or fabricate facts.
3. Do NOT duplicate topics already covered by the existing pairs.
4. Questions should be natural — the kind a researcher would actually type into a search box.
5. Every answer MUST end with the citation marker provided in the user message.
{domain_notes}
## Output format

```json
[
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}},
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}}
]
```"""


# --- User prompt (shared by both calls) ---

USER_PROMPT_TEMPLATE = """Entity ID: {entity_id}
Citation marker: <<SRC:{domain}:{entity_id}>>

Data:
{entity_json}"""


def build_battery_system_prompt(domain: str) -> str:
    """Build the battery system prompt (call 1).

    Strictly one pair per field group.
    """
    labels = DOMAIN_LABELS[domain]
    notes = DOMAIN_NOTES.get(domain)
    domain_notes = f"\n## Data notes\n\n{notes}\n" if notes else ""
    return BATTERY_SYSTEM_PROMPT_TEMPLATE.format(
        domain_display_name=labels["display"],
        entity_type=labels["entity_type"],
        field_guidance_block=format_field_guidance_block(domain),
        domain_notes=domain_notes,
    )


def build_discovery_system_prompt(domain: str, existing_pairs: list[dict[str, str]]) -> str:
    """Build the discovery system prompt (call 2).

    Receives existing pairs so it knows what's already covered.
    """
    labels = DOMAIN_LABELS[domain]
    notes = DOMAIN_NOTES.get(domain)
    domain_notes = f"\n## Data notes\n\n{notes}\n" if notes else ""
    # Format existing pairs as a readable list
    lines = []
    for i, pair in enumerate(existing_pairs, 1):
        lines.append(f"{i}. **Q:** {pair['question']}")
        # Truncate long answers for the prompt — the LLM just needs to know the topic
        answer_preview = pair["answer"][:150]
        if len(pair["answer"]) > 150:
            answer_preview += "..."
        lines.append(f"   **A:** {answer_preview}")
    existing_pairs_block = "\n".join(lines) if lines else "(none)"
    return DISCOVERY_SYSTEM_PROMPT_TEMPLATE.format(
        domain_display_name=labels["display"],
        entity_type=labels["entity_type"],
        existing_pairs_block=existing_pairs_block,
        domain_notes=domain_notes,
    )


def build_user_prompt(domain: str, entity_id: str, entity_json: str) -> str:
    """Build the user prompt for a single entity."""
    return USER_PROMPT_TEMPLATE.format(
        entity_id=entity_id,
        domain=domain,
        entity_json=entity_json,
    )
