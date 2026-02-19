# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Extract Q&A pairs from ACCESS-CI MCP (Model Context Protocol) servers to generate JSONL training data for RAG-based Q&A retrieval on HPC resource discovery. Part of a larger ACCESS-CI intelligent Q&A system.

## Related Repositories

These sibling repos live under the same parent directory (`access-ci/`):

- **`access-mcp/`** — The MCP servers (Node.js/TypeScript). Contains all server implementations including compute-resources, software-discovery, allocations, nsf-awards, affinity-groups, and others. Run via `docker compose up` from that repo.
- **`access-qa-planning/`** — System design docs, specs, and architecture decisions for the whole Q&A pipeline.
- **`access-argilla/`** — Docker Compose stack for Argilla (human review platform). Runs Argilla server on port 6900 with Elasticsearch, Redis, and PostgreSQL. Local dev credentials in its `.env`.

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

# CLI entry point (all 5 extractors available)
qa-extract extract compute-resources
qa-extract extract software-discovery
qa-extract extract allocations
qa-extract extract nsf-awards
qa-extract extract affinity-groups
qa-extract list-servers
qa-extract stats data/output/file.jsonl
qa-extract validate data/output/file.jsonl

# Extraction control flags
qa-extract extract compute-resources --max-entities 2    # cap entities sent to LLM
qa-extract extract allocations --max-queries 3           # cap search queries used
qa-extract extract nsf-awards --search-limit 50          # cap results per MCP query
qa-extract extract compute-resources --no-judge            # skip judge evaluation
qa-extract extract compute-resources --incremental         # skip unchanged entities (hash cache)
qa-extract extract compute-resources --push-to-argilla     # push to Argilla after extraction
qa-extract push data/output/file.jsonl                    # push existing JSONL to Argilla
```

## Architecture

**Pipeline flow**: MCP Servers → Extractors (2-pass per entity) → Generators (programmatic) → JSONL output

Each entity goes through up to 2 passes: (1) freeform Q&A via LLM (variable pair count, categories as guidance not constraint), (2) judge evaluation via cheaper LLM. Pass 2 is skippable with `--no-judge`. An incremental cache (`--incremental`) skips unchanged entities entirely.

### Key layers

- **`cli.py`** — Typer CLI with commands: `extract`, `list-servers`, `stats`, `validate`. Orchestrates the full extraction pipeline. Contains `EXTRACTORS` registry dict mapping server names to extractor classes.
- **`mcp_client.py`** — Async HTTP client (httpx) for invoking MCP server tool endpoints. `call_tool(tool_name, args)` POSTs to `{url}/tools/{tool_name}` and parses the MCP response format `{"content": [{"type": "text", "text": "..."}]}`, returning the parsed JSON dict.
- **`llm_client.py`** — Abstract `BaseLLMClient` with four backends: `AnthropicClient`, `OpenAIClient`, `LocalLLMClient` (vLLM/ollama via OpenAI-compatible API), `TransformersClient`. Selected via `LLM_BACKEND` env var through `get_llm_client()` factory. Also has `get_judge_client()` for the cheaper judge model (`LLM_JUDGE_BACKEND`/`LLM_JUDGE_MODEL` env vars).
- **`models.py`** — Pydantic models: `QAPair`, `Message`, `QAMetadata`. `QAPair.create()` factory auto-detects citations and sets metadata. `ExtractionResult = list[QAPair]`.
- **`extractors/`** — Per-domain extractors inheriting `BaseExtractor`. Each fetches data from an MCP server, cleans it, and uses LLM prompts to generate Q&A pairs.
- **`generators/comparisons.py`** — `ComparisonGenerator` produces cross-resource comparison Q&As programmatically from extractor output (no LLM, zero hallucination risk).
- **`generators/incremental.py`** — `IncrementalCache` with `compute_entity_hash()` for hash-based change detection. Stores pairs + judge scores so unchanged entities are skipped on re-runs.
- **`generators/judge.py`** — `evaluate_pairs()` sends all pairs for one entity to a cheaper judge LLM. Scores faithfulness, relevance, completeness (0.0-1.0). Confidence = min(three scores). Threshold 0.8 → `suggested_decision`.
- **`question_categories.py`** — Shared module defining 5-6 categories per domain (used as guidance, not constraint), prompt builders (`build_freeform_system_prompt`, `build_user_prompt`).
- **`citation_validator.py`** — Validates `<<SRC:domain:entity_id>>` citations against real MCP entities. Used by `validate` CLI command and for hallucination detection.
- **`argilla_client.py`** — `ArgillaClient` for pushing Q&A pairs to Argilla for human review. Dataset creation with full metadata schema (judge scores, granularity, eval_issues, source_ref), embedding generation (all-MiniLM-L6-v2), batch pushing. **Needs update:** current dedup-by-embedding logic to be replaced with entity-replace-by-`source_ref`.
- **`output/jsonl_writer.py`** — Writes QAPair lists to JSONL files (single, multi-server, or combined).

### Extractor pattern

Every extractor follows the same recipe (see `compute_resources.py` and `software_discovery.py` as references):

1. Subclass `BaseExtractor`, set `server_name` class attribute
2. Accept optional `llm_client: BaseLLMClient | None = None` in `__init__`
3. Implement `async def extract() -> ExtractionOutput`
4. Fetch data via `self.client.call_tool(tool_name, params)` — returns parsed JSON dicts
5. Clean raw data (strip HTML, filter junk fields)
6. Use `build_freeform_system_prompt(domain)` for the system prompt
7. Send cleaned data to LLM, parse JSON array of `{"question", "answer"}` from response
8. Wrap each in `QAPair.create(...)` with sequential IDs (`{domain}_{entity_id}_{seq_n}`)
9. Return `ExtractionOutput(pairs=pairs, raw_data=raw_data)`
   - `raw_data` is keyed by entity ID with normalized fields for `ComparisonGenerator`

### Citation format

All assistant answers use `<<SRC:domain:entity_id>>` citations (e.g., `<<SRC:compute-resources:delta.ncsa.access-ci.org>>`). The citation validator checks these against known MCP entities to detect hallucinated references.

## MCP Servers — Ports and Tools

These are the actual ports from `access-mcp/docker-compose.yml`. The extraction config in `config.py` must match these.

| Server | Port | MCP Tools | Strategy |
|---|---|---|---|
| 🖥️ compute-resources | 3002 | `search_resources`, `get_resource_hardware` | list-all (~26 records) |
| 📦 software-discovery | 3004 | `search_software` | search-terms (~34 curated terms) |
| 📊 allocations | 3006 | `search_projects`, `get_allocation_statistics`, `analyze_funding` | broad-queries (~5,360 projects) |
| 💰 nsf-awards | 3007 | `search_nsf_awards` | broad-queries (10K cap) |
| 👥 affinity-groups | 3011 | `search_affinity_groups` | list-all (~54 groups) |

All 5 extractors are implemented and registered in the CLI.

### Extraction strategies

Each extractor uses one of three strategies depending on dataset size and server API behavior:

- **list-all**: Server returns all entities with an empty/broad query. Used for small datasets: 🖥️ compute-resources (~26 HPC systems) and 👥 affinity-groups (~54 community groups).
- **search-terms**: A curated list of domain-specific search terms. Used by 📦 software-discovery, which has a `SOFTWARE_SEARCH_TERMS` constant with ~34 terms (python, gcc, cuda, tensorflow, etc.).
- **broad-queries**: For large datasets where the MCP server requires search parameters. The extractor sends targeted keyword queries organized by dimension (fields of science, resource names, institutions, etc.) with deduplication via `seen_ids`. Used by 📊 allocations and 💰 nsf-awards. **Team decision**: keep search logic in the Python extractors rather than adding list-all fallbacks to the MCP servers — Andrew's concern is that MCP endpoints should encourage agents to use targeted parameters, not dump entire datasets into context.

### Software-discovery API key

The 📦 software-discovery extractor requires `SDS_API_KEY` to be set in the `access-mcp` repo's `.env` file (not this repo). Docker Compose passes it to the software-discovery container. Without it, the extractor returns 0 results.

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
- `LLM_JUDGE_BACKEND` — Backend for judge evaluation (defaults to main `LLM_BACKEND` value)
- `LLM_JUDGE_MODEL` — Model for judge (e.g., `gpt-4o-mini`, `claude-haiku`). Cheaper model recommended
- `ARGILLA_URL` — Argilla server URL (default `http://localhost:6900`)
- `ARGILLA_API_KEY` — Argilla API key (default `argilla.apikey`)

## Code Style

- Python 3.11+, Pydantic v2 models
- Ruff: line-length=100, lint rules `E,F,I,UP`
- Async-first design (httpx async, pytest-asyncio)
- Source layout: `src/access_qa_extraction/`

## Current Work

All 5 extractors are implemented with the 2-pass pipeline (freeform LLM + judge). 124 tests passing on branch `spike/quality-incremental-bonus`.

### What's done

- **2-pass pipeline** — freeform LLM extraction (variable pair count, categories as guidance) + judge evaluation (cheaper LLM). Factoid templates were removed after analysis showed 72% overlap with freeform and 100% of data quality issues were factoid-only (see `docs/TO_FACTOID_OR_NOT.md`).
- **Incremental cache** — hash-based change detection in `data/cache/{domain}/`. Unchanged entities are skipped on re-runs. Cache stores pairs + judge scores.
- **Argilla client** — `ArgillaClient` with full metadata schema: judge scores, suggested_decision, granularity, eval_issues, source_ref. Dataset creation, embedding generation, batch pushing. **Known issue:** current dedup-by-embedding approach doesn't work (only compares questions, blocks updated answers). Will be replaced with entity-replace-by-`source_ref`. See design doc.

### Current priorities

1. **Implement entity-replace in ArgillaClient** — Decided approach: delete all Argilla records by `source_ref` before pushing fresh extraction. Semantic dedup ruled out (only compares questions, blocks updated answers). Need `delete_records_by_source_ref()` and modified `push_pairs()` flow. See `docs/design-extraction-rethink-2026-02-18.md` Parts 1.5 and 2.
2. **Per-domain cleanup** — allocations (singular/plural grammar), nsf-awards (dirty primary_program field, co-PI emails), affinity-groups (thin data overlap, obfuscated emails).
3. **Software-discovery testing** — Needs `SDS_API_KEY` in access-mcp `.env` to return results.
