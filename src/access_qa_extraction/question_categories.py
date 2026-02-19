"""Question categories and prompt templates for all 5 extractors.

Each domain has 5-6 key areas. These serve two roles:
  1. (Legacy) Fixed category menu: LLM generates exactly one pair per category
  2. (New) Freeform guidance: LLM uses categories as a floor, not a ceiling

Design principles:
  - Categories are derived from the cleaned data fields the LLM sees
  - "overview" and "people" are universal across all domains
  - No "comparison" category — that's ComparisonGenerator's job
  - 5-6 categories per domain: enough variety, not excessive
  - Categories marked with condition != "always" → LLM skips if data isn't there
"""

QUESTION_CATEGORIES = {
    "compute-resources": [
        {
            "id": "overview",
            "description": "What is this resource and what is it designed for?",
            "condition": "always",
        },
        {
            "id": "organization",
            "description": "What organization operates this resource?",
            "condition": "always",
        },
        {
            "id": "gpu_hardware",
            "description": "What GPUs does this resource have? (models, counts, memory)",
            "condition": "only if hasGpu is true and hardware.gpus is non-empty",
        },
        {
            "id": "cpu_hardware",
            "description": "What CPUs/compute nodes does this resource have?",
            "condition": "only if hardware.compute_nodes is non-empty",
        },
        {
            "id": "capabilities",
            "description": "What features and capabilities does this resource support?",
            "condition": "always",
        },
        {
            "id": "access",
            "description": "How do I get access to or start using this resource?",
            "condition": "always",
        },
    ],
    "software-discovery": [
        {
            "id": "overview",
            "description": "What is this software and what type of tool is it?",
            "condition": "always",
        },
        {
            "id": "availability",
            "description": "Which ACCESS resources have this software installed?",
            "condition": "always",
        },
        {
            "id": "versions",
            "description": "What versions of this software are available?",
            "condition": "always",
        },
        {
            "id": "usage",
            "description": "How do I load and use this software on ACCESS systems?",
            "condition": "only if example_use is present",
        },
        {
            "id": "research_use",
            "description": "What research areas or disciplines use this software?",
            "condition": "always",
        },
    ],
    "allocations": [
        {
            "id": "overview",
            "description": "What is this project about? (title, abstract, allocation type)",
            "condition": "always",
        },
        {
            "id": "people",
            "description": "Who is the PI and what institution are they from?",
            "condition": "always",
        },
        {
            "id": "resources",
            "description": "What compute resources are allocated and how much?",
            "condition": "always",
        },
        {
            "id": "field_of_science",
            "description": "What field of science does this project fall under?",
            "condition": "always",
        },
        {
            "id": "timeline",
            "description": "When does this allocation start and end?",
            "condition": "always",
        },
    ],
    "nsf-awards": [
        {
            "id": "overview",
            "description": "What is this award about? (title and abstract)",
            "condition": "always",
        },
        {
            "id": "people",
            "description": "Who is the PI (and co-PIs if any), and what institution?",
            "condition": "always",
        },
        {
            "id": "funding",
            "description": "How much funding was awarded?",
            "condition": "always",
        },
        {
            "id": "program",
            "description": "What NSF program funds this award?",
            "condition": "always",
        },
        {
            "id": "timeline",
            "description": "When does this award start and end?",
            "condition": "always",
        },
    ],
    "affinity-groups": [
        {
            "id": "overview",
            "description": "What is this group and what community does it serve?",
            "condition": "always",
        },
        {
            "id": "people",
            "description": "Who coordinates this group?",
            "condition": "always",
        },
        {
            "id": "access",
            "description": "How can I join or contact this group?",
            "condition": "always",
        },
        {
            "id": "events",
            "description": "What events does this group offer?",
            "condition": "only if upcoming_events is present and non-empty",
        },
        {
            "id": "knowledge_base",
            "description": "What resources or articles does this group maintain?",
            "condition": "only if knowledge_base_topics is present and non-empty",
        },
    ],
}

DOMAIN_LABELS = {
    "compute-resources": {"display": "compute resources", "entity_type": "HPC system"},
    "software-discovery": {"display": "software catalog", "entity_type": "software package"},
    "allocations": {"display": "allocation projects", "entity_type": "allocation project"},
    "nsf-awards": {"display": "NSF awards", "entity_type": "NSF award"},
    "affinity-groups": {"display": "affinity groups", "entity_type": "community group"},
}

SYSTEM_PROMPT_TEMPLATE = """You are a Q&A pair generator for ACCESS-CI {domain_display_name}.

You will receive structured data about a single {entity_type}. Generate exactly one
Q&A pair for each applicable category listed below. Skip a category ONLY if the data
genuinely does not contain information for it.

## Categories

{categories_block}

## Rules

1. Output a JSON array. Each element has three fields: "category", "question", "answer".
2. The "category" field MUST be one of the category IDs listed above.
3. Only use information present in the provided data. Do not infer or fabricate facts.
4. Questions should be natural — the kind a researcher would actually type into a search box.
5. Answers should be concise but complete. Include specific numbers, names, and dates
   when the data provides them.
6. Every answer MUST end with the citation marker provided in the user message.
7. Do not generate multiple Q&A pairs for the same category.

## Output format

```json
[
  {{"category": "overview", "question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}},
  {{"category": "people",   "question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}}
]
```"""

USER_PROMPT_TEMPLATE = """Entity ID: {entity_id}
Citation marker: <<SRC:{domain}:{entity_id}>>

Data:
{entity_json}"""


def format_categories_block(domain: str) -> str:
    """Format the categories for a domain into a markdown list for the system prompt."""
    categories = QUESTION_CATEGORIES[domain]
    lines = []
    for cat in categories:
        condition_note = ""
        if cat["condition"] != "always":
            condition_note = " *(skip if not applicable)*"
        lines.append(f"- **{cat['id']}**: {cat['description']}{condition_note}")
    return "\n".join(lines)


def build_system_prompt(domain: str) -> str:
    """Build the full system prompt for a domain."""
    labels = DOMAIN_LABELS[domain]
    return SYSTEM_PROMPT_TEMPLATE.format(
        domain_display_name=labels["display"],
        entity_type=labels["entity_type"],
        categories_block=format_categories_block(domain),
    )


def build_user_prompt(domain: str, entity_id: str, entity_json: str) -> str:
    """Build the user prompt for a single entity."""
    return USER_PROMPT_TEMPLATE.format(
        entity_id=entity_id,
        domain=domain,
        entity_json=entity_json,
    )


# --- Freeform extraction (single LLM pass, categories as guidance) ---

FREEFORM_SYSTEM_PROMPT_TEMPLATE = """You are a Q&A pair generator for ACCESS-CI {domain_display_name}.

You will receive structured data about a single {entity_type}. Generate Q&A pairs that
capture all useful information a researcher might want to know. Generate as many pairs
as the data warrants — data-rich entities should produce more pairs than simple ones.

## Key areas to cover

At minimum, make sure you address these topics if the data supports them:

{categories_block}

## Go beyond the basics

After covering the key areas above, look for additional details worth surfacing:
- Specific hardware specs, storage systems, networking capabilities
- Notable partnerships, collaborations, or multi-institution arrangements
- Unique technologies, architectures, or methodologies
- Interdisciplinary applications or unusual use cases
- Specific numbers, capacities, or performance characteristics

Generate a separate Q&A pair for each distinct piece of information. Do NOT cram
multiple topics into one answer. A data-rich entity might warrant 10-15 pairs.
A simple entity might only need 4-5. Let the data drive the count.

## Rules

1. Output a JSON array. Each element has two fields: "question", "answer".
2. Only use information present in the provided data. Do not infer or fabricate facts.
3. Questions should be natural — the kind a researcher would actually type into a search box.
4. Answers should be concise but complete. Include specific numbers, names, and dates
   when the data provides them.
5. Every answer MUST end with the citation marker provided in the user message.
6. Each pair should cover a distinct topic — no duplicate or overlapping questions.

## Output format

```json
[
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}},
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}}
]
```"""


def format_freeform_categories_block(domain: str) -> str:
    """Format categories as guidance (floor, not ceiling) for freeform prompt."""
    categories = QUESTION_CATEGORIES[domain]
    lines = []
    for cat in categories:
        condition_note = ""
        if cat["condition"] != "always":
            condition_note = " *(only if data is present)*"
        lines.append(f"- {cat['description']}{condition_note}")
    return "\n".join(lines)


def build_freeform_system_prompt(domain: str) -> str:
    """Build the freeform system prompt for a domain.

    Uses categories as guidance (floor) rather than constraints (ceiling).
    The LLM generates as many pairs as the data warrants.
    """
    labels = DOMAIN_LABELS[domain]
    return FREEFORM_SYSTEM_PROMPT_TEMPLATE.format(
        domain_display_name=labels["display"],
        entity_type=labels["entity_type"],
        categories_block=format_freeform_categories_block(domain),
    )
