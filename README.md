# ACCESS Q&A Extraction

Extract Q&A pairs from ACCESS-CI MCP servers for fine-tuning language models.

## Overview

This tool generates training data by running a 2-pass pipeline per entity:
1. **Freeform Q&A** — LLM generates variable-count question-answer pairs from cleaned entity data (categories as guidance, not constraint)
2. **Judge evaluation** — Cheaper LLM scores each pair on faithfulness, relevance, and completeness (skippable with `--no-judge`)

Output is JSONL files with 2 granularity levels (comprehensive, comparison), plus an incremental cache so unchanged entities are skipped on re-runs. Q&A pairs can be pushed to Argilla for human review.

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
                                                         │     Argilla     │
                                                         │  (human review) │
                                                         └─────────────────┘
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

> **Note (macOS):** Use `python3` (not `python`) when creating the virtual environment.
> On a fresh Mac, `python` may not exist or may point to an old system Python.
> Once the venv is activated, `python` and `python3` both work — the venv
> creates its own `python` symlink that points to the correct Python 3.

## Local Development Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for MCP servers)
- An LLM API key (OpenAI, Anthropic) or Ollama for local models

### 1. Python environment

```bash
python3 -m venv .venv
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

### 4. Argilla (human review)

Argilla is used for human review of extracted Q&A pairs. The Docker stack lives in the sibling `access-argilla/` repo:

```bash
cd ../access-argilla
docker compose up -d
```

This starts Argilla on `http://localhost:6900` (default login: `argilla` / `12345678`).

### 5. Verify setup

```bash
# Tests (no LLM or MCP servers needed)
pytest

# Check server config
qa-extract list-servers

# Smoke test (MCP servers must be running, .env configured)
qa-extract extract compute-resources --dry-run
```

## Configuration

LLM backend and MCP server URLs are configured via environment variables. Copy `.env.example` to `.env` and edit:

```bash
# LLM backend (choose one)
LLM_BACKEND=openai              # or: anthropic, local, transformers
OPENAI_API_KEY=sk-...           # for openai backend
OPENAI_MODEL=gpt-4o             # default model

# Judge evaluation (optional, defaults to main LLM backend)
LLM_JUDGE_BACKEND=openai        # cheaper model recommended
LLM_JUDGE_MODEL=gpt-4o-mini

# MCP server URLs (defaults shown)
MCP_COMPUTE_RESOURCES_URL=http://localhost:3002
MCP_SOFTWARE_DISCOVERY_URL=http://localhost:3004
MCP_ALLOCATIONS_URL=http://localhost:3006
MCP_NSF_AWARDS_URL=http://localhost:3007
MCP_AFFINITY_GROUPS_URL=http://localhost:3011

# Argilla (human review)
ARGILLA_URL=http://localhost:6900
ARGILLA_API_KEY=argilla.apikey
```

## Usage

```bash
# Extract from specific server
qa-extract extract compute-resources
qa-extract extract software-discovery

# Extract from multiple servers
qa-extract extract compute-resources software-discovery

# Cheap test run (2 entities, fast feedback)
qa-extract extract compute-resources --max-entities 2

# Skip judge pass (faster, cheaper)
qa-extract extract allocations --no-judge

# Incremental mode (skip unchanged entities on re-runs)
qa-extract extract compute-resources --incremental

# Combine output into a single file
qa-extract extract compute-resources software-discovery --combined

# Specify output directory
qa-extract extract compute-resources --output ./my-output

# Dry run (show what would be generated)
qa-extract extract compute-resources --dry-run

# Control search scope for broad-query extractors
qa-extract extract allocations --max-queries 3 --search-limit 50

# Extract and push directly to Argilla for review
qa-extract extract compute-resources --push-to-argilla

# Push an existing JSONL file to Argilla
qa-extract push data/output/compute-resources_qa_pairs.jsonl

```

## Output Format

JSONL files with one JSON object per line. Each pair includes judge evaluation scores when available:

```json
{
  "id": "compute-resources_delta.ncsa.access-ci.org_1",
  "source": "mcp_extraction",
  "source_ref": "mcp://compute-resources/resources/delta.ncsa.access-ci.org",
  "domain": "compute-resources",
  "messages": [
    {"role": "user", "content": "What is Delta and what is it designed for?"},
    {"role": "assistant", "content": "Delta is a GPU-focused HPC system at NCSA...\n\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"}
  ],
  "metadata": {
    "complexity": "simple",
    "granularity": "comprehensive",
    "has_citation": true,
    "faithfulness_score": 0.95,
    "relevance_score": 0.90,
    "completeness_score": 0.85,
    "confidence_score": 0.85,
    "suggested_decision": "approved",
    "source_data": {"name": "Delta", "...": "..."}
  }
}
```

### Two granularity levels

| Granularity | Generator | LLM? | Example |
|---|---|---|---|
| **comprehensive** | Freeform LLM pass (variable count) | Yes | "What is Delta and what is it designed for?" |
| **comparison** | ComparisonGenerator | No | "Which ACCESS resources support interactive computing?" |

## Development

```bash
# Run tests
pytest

# Format and lint
ruff format .
ruff check --fix .
```
