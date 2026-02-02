# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Extract Q&A pairs from ACCESS-CI MCP (Model Context Protocol) servers to generate JSONL training data for RAG-based Q&A retrieval on HPC resource discovery. Part of a larger ACCESS-CI intelligent Q&A system.

## Related Repositories

These sibling repos live under the same parent directory (`access-ci/`):

- **`access-mcp/`** — The MCP servers (Node.js/TypeScript). Contains all server implementations including compute-resources, software-discovery, allocations, nsf-awards, affinity-groups, and others. Run via `docker compose up` from that repo.
- **`access-qa-planning/`** — System design docs, specs, and architecture decisions for the whole Q&A pipeline.

## Commands

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Run tests (uses pytest-asyncio with asyncio_mode="auto")
pytest

# Run a single test
pytest tests/test_citation_validator.py::test_name

# Lint and format
ruff check --fix .
ruff format .

# CLI entry point
qa-extract extract compute-resources
qa-extract extract software-discovery
qa-extract list-servers
qa-extract stats data/output/file.jsonl
qa-extract validate data/output/file.jsonl
```

## Architecture

**Pipeline flow**: MCP Servers → Extractors (LLM-powered) → Generators (programmatic) → JSONL output

### Key layers

- **`cli.py`** — Typer CLI with commands: `extract`, `list-servers`, `stats`, `validate`. Orchestrates the full extraction pipeline. Contains `EXTRACTORS` registry dict mapping server names to extractor classes.
- **`mcp_client.py`** — Async HTTP client (httpx) for invoking MCP server tool endpoints. `call_tool(tool_name, args)` POSTs to `{url}/tools/{tool_name}` and parses the MCP response format `{"content": [{"type": "text", "text": "..."}]}`, returning the parsed JSON dict.
- **`llm_client.py`** — Abstract `BaseLLMClient` with three backends: `AnthropicClient`, `LocalLLMClient` (vLLM/ollama via OpenAI-compatible API), `TransformersClient`. Selected via `LLM_BACKEND` env var through `get_llm_client()` factory.
- **`models.py`** — Pydantic models: `QAPair`, `Message`, `QAMetadata`. `QAPair.create()` factory auto-detects citations and sets metadata. `ExtractionResult = list[QAPair]`.
- **`extractors/`** — Per-domain extractors inheriting `BaseExtractor`. Each fetches data from an MCP server, cleans it, and uses LLM prompts to generate Q&A pairs.
- **`generators/comparisons.py`** — `ComparisonGenerator` produces cross-resource comparison Q&As programmatically from extractor output (no LLM, zero hallucination risk).
- **`citation_validator.py`** — Validates `<<SRC:domain:entity_id>>` citations against real MCP entities. Used by `validate` CLI command and for hallucination detection.
- **`output/jsonl_writer.py`** — Writes QAPair lists to JSONL files (single, multi-server, or combined).

### Extractor pattern

Every extractor follows the same recipe (see `compute_resources.py` and `software_discovery.py` as references):

1. Subclass `BaseExtractor`, set `server_name` class attribute
2. Accept optional `llm_client: BaseLLMClient | None = None` in `__init__`
3. Implement `async def extract() -> ExtractionOutput`
4. Fetch data via `self.client.call_tool(tool_name, params)` — returns parsed JSON dicts
5. Clean raw data (strip HTML, filter junk fields)
6. Define `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` as module-level constants
7. Send cleaned data to LLM, parse JSON array of `{"question", "answer"}` from response
8. Wrap each in `QAPair.create(id, question, answer, source_ref, domain, complexity, source_data)`
9. Return `ExtractionOutput(pairs=pairs, raw_data=raw_data)`
   - `raw_data` is keyed by entity ID with normalized fields for `ComparisonGenerator`

### Citation format

All assistant answers use `<<SRC:domain:entity_id>>` citations (e.g., `<<SRC:compute-resources:delta.ncsa.access-ci.org>>`). The citation validator checks these against known MCP entities to detect hallucinated references.

## MCP Servers — Ports and Tools

These are the actual ports from `access-mcp/docker-compose.yml`. The extraction config in `config.py` must match these.

| Server | Port | MCP Tools | Extractor Status |
|---|---|---|---|
| compute-resources | 3002 | `search_resources`, `get_resource_hardware` | Implemented |
| software-discovery | 3004 | `search_software` | Implemented |
| allocations | 3006 | `search_projects`, `get_allocation_statistics`, `analyze_funding` | Not yet |
| nsf-awards | 3007 | `search_nsf_awards` | Not yet |
| affinity-groups | 3011 | `search_affinity_groups` | Not yet |

### MCP tool response shapes (for new extractors)

**allocations `search_projects`** returns `{total, items}` where each item has: `projectId`, `requestNumber`, `requestTitle`, `pi`, `piInstitution`, `fos` (field of science), `abstract`, `allocationType`, `beginDate`, `endDate`, `resources[{resourceName, units, allocation, resourceId}]`

**nsf-awards `search_nsf_awards`** returns `{total, items}` where each item has: `awardNumber`, `title`, `institution`, `principalInvestigator`, `coPIs[]`, `totalIntendedAward` (pre-formatted "$X,XXX"), `totalAwardedToDate`, `startDate`, `endDate`, `abstract`, `primaryProgram`, `programOfficer`, `ueiNumber`

**affinity-groups `search_affinity_groups`** returns `{total, items}` where each item has: `id`, `name`, `description`, `coordinator`, `category`, `slack_link`, `support_url`, `ask_ci_forum`. With `include="all"` and a specific `id`: returns `{group, events{total,items}, knowledge_base{total,items}}`

## Local Development Setup

Full step-by-step instructions are in README.md. Key details for Claude context:

### Step 1: Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Step 2: MCP servers (Docker)
From sibling `access-mcp/` repo:
```bash
cd ../access-mcp
docker compose up -d
```
Starts all servers on assigned ports (see table above). Some servers need env vars (e.g., `SDS_API_KEY` for software-discovery). Verify with `docker compose ps`.

### Step 3: LLM backend
The `LLM_BACKEND` env var selects the LLM client (see `llm_client.py`). Four options:

- **`anthropic`** (default) — Needs `ANTHROPIC_API_KEY` in `.env`. Ask Andrew (apasquale) about the key.
- **`openai`** — Calls OpenAI's cloud API. Needs `OPENAI_API_KEY` in `.env`. Set `OPENAI_MODEL` to choose the model (default `gpt-4o`).
- **`local`** — Connects to any OpenAI-compatible API running locally (vLLM, Ollama). For local dev with Ollama:
  ```bash
  brew install ollama
  brew services start ollama
  ollama pull qwen3:8b
  ```
  Then run extractions with:
  ```bash
  LLM_BACKEND=local LOCAL_LLM_URL=http://localhost:11434/v1 LOCAL_LLM_MODEL=qwen3:8b \
    qa-extract extract compute-resources --dry-run
  ```
- **`transformers`** — Loads model directly into GPU memory. Requires serious hardware (GH200). See `scripts/run_extraction_gh200.py`.

For local development, OpenAI or Ollama with a small model are the easiest ways to validate the full pipeline.

### Step 4: Verify
```bash
pytest                        # Tests (no LLM/MCP needed)
qa-extract list-servers       # Check config
# Then with MCP servers + LLM running:
qa-extract extract compute-resources --dry-run
```

## Environment Variables

- `LLM_BACKEND` — `anthropic` (default), `openai`, `local`, or `transformers`
- `ANTHROPIC_API_KEY` — Required when using Anthropic backend
- `OPENAI_API_KEY` — Required when using OpenAI backend
- `OPENAI_MODEL` — Model for OpenAI backend (default `gpt-4o`)
- `LOCAL_LLM_URL` — URL for local backend (default `http://localhost:8000/v1`)
- `LOCAL_LLM_MODEL` — Model name for local backend (default `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`)
- `MCP_COMPUTE_RESOURCES_URL` — default `http://localhost:3002`
- `MCP_SOFTWARE_DISCOVERY_URL` — default `http://localhost:3004`
- `MCP_ALLOCATIONS_URL` — default `http://localhost:3006`
- `MCP_NSF_AWARDS_URL` — default `http://localhost:3007`
- `MCP_AFFINITY_GROUPS_URL` — default `http://localhost:3011`

## Code Style

- Python 3.11+, Pydantic v2 models
- Ruff: line-length=100, lint rules `E,F,I,UP`
- Async-first design (httpx async, pytest-asyncio)
- Source layout: `src/access_qa_extraction/`

## Current Work

Implementation plan for three new extractors is at `.claude/plans/stateless-coalescing-metcalfe.md`. Summary:
1. Fix config port mismatches (config.py currently has wrong ports for allocations, nsf-awards, affinity-groups)
2. Build `AllocationsExtractor`, `NSFAwardsExtractor`, `AffinityGroupsExtractor`
3. Register in CLI and update citation validator
4. Add tests for each
