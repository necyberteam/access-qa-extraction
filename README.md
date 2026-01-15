# ACCESS Q&A Extraction

Extract Q&A pairs from ACCESS-CI MCP servers for fine-tuning language models.

## Overview

This tool generates training data by:
1. Calling MCP server endpoints to fetch structured data
2. Applying Q&A templates to generate question-answer pairs
3. Outputting JSONL files compatible with the training pipeline
4. (Optional) Pushing to Argilla for human review

## Data Flow

```
MCP Servers                    Extraction                      Output
┌─────────────────┐           ┌──────────────┐           ┌─────────────────┐
│compute-resources│──────────▶│              │──────────▶│ data/output/    │
│software-discovery│──────────▶│  Extractors  │           │ *.jsonl         │
│allocations      │──────────▶│      +       │           └─────────────────┘
│affinity-groups  │──────────▶│  Generators  │                   │
│nsf-awards       │──────────▶│              │                   ▼
└─────────────────┘           └──────────────┘           ┌─────────────────┐
                                                         │ Argilla (later) │
                                                         └─────────────────┘
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# With Argilla support (later)
pip install -e ".[dev,argilla]"
```

## Configuration

Set MCP server URLs via environment variables:

```bash
export MCP_COMPUTE_RESOURCES_URL=http://localhost:3002
export MCP_SOFTWARE_DISCOVERY_URL=http://localhost:3003
export MCP_ALLOCATIONS_URL=http://localhost:3004
export MCP_AFFINITY_GROUPS_URL=http://localhost:3005
export MCP_NSF_AWARDS_URL=http://localhost:3006
```

## Usage

```bash
# Extract from all servers
qa-extract all

# Extract from specific server
qa-extract compute-resources
qa-extract software-discovery

# Specify output directory
qa-extract all --output ./my-output

# Dry run (show what would be generated)
qa-extract all --dry-run
```

## Output Format

JSONL files following the schema in `access-qa-planning/02-training-data.md`:

```json
{
  "id": "qa_00001",
  "source": "mcp_extraction",
  "source_ref": "mcp://compute-resources/resources/delta.ncsa.access-ci.org",
  "domain": "compute:resource_specs",
  "messages": [
    {"role": "user", "content": "What GPUs does Delta have?"},
    {"role": "assistant", "content": "Delta at NCSA has NVIDIA A100 GPUs...\n\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"}
  ],
  "metadata": {
    "complexity": "simple",
    "has_citation": true,
    "created_at": "2025-01-05T00:00:00Z"
  }
}
```

## Q&A Templates

### Factual (per entity)
- "What GPUs does {resource} have?"
- "How many nodes does {resource} have?"
- "Is {software} available on {resource}?"

### Comparison
- "Compare {resource_a} and {resource_b}"
- "Which has more GPUs, {resource_a} or {resource_b}?"

### Recommendation
- "What resource should I use for machine learning?"
- "Which system is best for large memory jobs?"

### Deferral (negative examples)
- "Is Delta currently down?" → Defer to live MCP
- "What events are this week?" → Defer to live MCP

## Development

```bash
# Run tests
pytest

# Format and lint
ruff format .
ruff check --fix .
```
