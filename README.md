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

## Local Development Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for MCP servers)
- An LLM API key (OpenAI, Anthropic) or Ollama for local models

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. MCP servers

The MCP servers live in the sibling `access-mcp/` repo and run via Docker:

```bash
cd ../access-mcp
docker compose up -d
```

This starts all servers on their assigned ports:

| Server | Port |
|---|---|
| compute-resources | 3002 |
| software-discovery | 3004 |
| allocations | 3006 |
| nsf-awards | 3007 |
| affinity-groups | 3011 |

### 3. LLM backend

Extraction requires an LLM to generate Q&A pairs. Choose one:

**Option A: OpenAI API key (easiest for dev)**
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (LLM_BACKEND=openai is already the default)
```

**Option B: Anthropic API key (production)**
```bash
cp .env.example .env
# Edit .env: comment out the OpenAI lines, uncomment ANTHROPIC_API_KEY, set LLM_BACKEND=anthropic
```

**Option C: Ollama (local models, no API key needed)**
```bash
brew install ollama
brew services start ollama
ollama pull qwen3:8b
# Then set LLM_BACKEND=local, LOCAL_LLM_URL=http://localhost:11434/v1, LOCAL_LLM_MODEL=qwen3:8b in .env
```

### 4. Verify setup

```bash
# Tests (no LLM or MCP servers needed)
pytest

# Check server config
qa-extract list-servers

# Smoke test (MCP servers must be running, .env configured)
qa-extract extract compute-resources --dry-run
```

## Configuration

MCP server URLs can be overridden via environment variables (defaults shown):

```bash
export MCP_COMPUTE_RESOURCES_URL=http://localhost:3002
export MCP_SOFTWARE_DISCOVERY_URL=http://localhost:3004
export MCP_ALLOCATIONS_URL=http://localhost:3006
export MCP_AFFINITY_GROUPS_URL=http://localhost:3011
export MCP_NSF_AWARDS_URL=http://localhost:3007
```

## Usage

```bash
# Extract from specific server
qa-extract extract compute-resources
qa-extract extract software-discovery

# Extract from multiple servers
qa-extract extract compute-resources software-discovery

# Combine output into a single file
qa-extract extract compute-resources software-discovery --combined

# Specify output directory
qa-extract extract compute-resources --output ./my-output

# Dry run (show what would be generated)
qa-extract extract compute-resources --dry-run
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
