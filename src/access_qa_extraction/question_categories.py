"""Fixed question categories for all 5 extractors.

Each domain has 5-6 categories. The LLM picks from this menu instead of
freestyling. IDs become {domain}_{entity_id}_{category} — stable and predictable.

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


# --- Bonus (exploratory) question generation ---

BONUS_SYSTEM_PROMPT_TEMPLATE = """You are a Q&A pair generator for ACCESS-CI {domain_display_name}.

You will receive structured data about a single {entity_type}, along with a list of
categories that have ALREADY been covered by standard Q&A generation. Your job is to
find 1-3 ADDITIONAL questions that surface unique, interesting, or specific information
from this entity that is NOT adequately captured by the standard categories.

## Already covered categories

{covered_categories}

## Rules

1. Output a JSON array. Each element has two fields: "question", "answer".
2. Only generate questions whose answers are NOT already covered by the categories above.
3. Focus on entity-specific details: notable collaborations, unusual capabilities,
   specific methodologies, interdisciplinary aspects, named technologies, etc.
4. Only use information present in the provided data. Do not infer or fabricate facts.
5. If the entity data contains nothing beyond what the standard categories cover,
   return an empty array: []
6. Every answer MUST end with the citation marker provided in the user message.
7. Generate at most 3 additional questions.

## Output format

```json
[
  {{"question": "...", "answer": "...\\n\\n<<SRC:domain:id>>"}},
]
```"""

RICH_TEXT_FIELDS = {
    "compute-resources": ["description"],
    "software-discovery": ["description"],
    "allocations": ["abstract"],
    "nsf-awards": ["abstract"],
    "affinity-groups": ["description"],
}

MIN_RICH_TEXT_LENGTH = 100

SOURCE_REF_PATTERNS = {
    "compute-resources": "mcp://compute-resources/resources/{entity_id}",
    "software-discovery": "mcp://software-discovery/software/{entity_id}",
    "allocations": "mcp://allocations/projects/{entity_id}",
    "nsf-awards": "mcp://nsf-awards/awards/{entity_id}",
    "affinity-groups": "mcp://affinity-groups/groups/{entity_id}",
}


def has_rich_text(domain: str, entity_data: dict) -> bool:
    """Check if entity has rich text fields long enough to warrant bonus questions."""
    fields = RICH_TEXT_FIELDS.get(domain, [])
    for field_name in fields:
        val = entity_data.get(field_name, "")
        if isinstance(val, str) and len(val.strip()) >= MIN_RICH_TEXT_LENGTH:
            return True
    return False


def build_bonus_system_prompt(domain: str) -> str:
    """Build the bonus (exploratory) system prompt for a domain."""
    labels = DOMAIN_LABELS[domain]
    categories = QUESTION_CATEGORIES[domain]
    covered = "\n".join(f"- **{cat['id']}**: {cat['description']}" for cat in categories)
    return BONUS_SYSTEM_PROMPT_TEMPLATE.format(
        domain_display_name=labels["display"],
        entity_type=labels["entity_type"],
        covered_categories=covered,
    )


def generate_bonus_pairs(
    domain: str,
    entity_id: str,
    entity_data: dict,
    llm_client,
    max_tokens: int = 1024,
) -> list:
    """Generate 0-3 exploratory bonus Q&A pairs for entity-unique information.

    Only fires when the entity has rich text (description/abstract >= 100 chars).
    Returns an empty list if the entity has no rich text or the LLM finds nothing
    beyond standard categories.
    """
    import json
    import re

    from .models import QAPair

    if not has_rich_text(domain, entity_data):
        return []

    system_prompt = build_bonus_system_prompt(domain)
    entity_json = json.dumps(entity_data, indent=2)
    user_prompt = build_user_prompt(domain, entity_id, entity_json)

    try:
        response = llm_client.generate(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
        )

        json_match = re.search(r"\[[\s\S]*\]", response.text)
        if not json_match:
            return []

        qa_list = json.loads(json_match.group())
        pairs = []
        bonus_num = 0
        for qa in qa_list:
            if bonus_num >= 3:
                break
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if question and answer:
                bonus_num += 1
                pair_id = f"{domain}_{entity_id}_bonus_{bonus_num}"
                pattern = SOURCE_REF_PATTERNS.get(
                    domain, f"mcp://{domain}/{entity_id}"
                )
                source_ref = pattern.format(entity_id=entity_id)

                pairs.append(
                    QAPair.create(
                        id=pair_id,
                        question=question,
                        answer=answer,
                        source_ref=source_ref,
                        domain=domain,
                        granularity="exploratory",
                    )
                )
        return pairs

    except Exception as e:
        print(f"Error generating bonus Q&A for {domain}/{entity_id}: {e}")
        return []
