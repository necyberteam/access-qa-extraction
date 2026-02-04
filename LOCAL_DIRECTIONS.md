# Running Extractors Locally

A step-by-step guide for running the Q&A extraction pipeline on your own machine. This assumes you're using an **OpenAI API key** as the LLM backend and have Docker Desktop installed.

---

## What's happening under the hood

This project generates training data for a Q&A system about ACCESS-CI (a national network of shared computing resources for researchers). The pipeline has three moving parts:

1. **MCP servers** — Small web services that wrap ACCESS-CI databases. Each one serves a specific kind of data (compute resources, software catalogs, NSF awards, etc.) over HTTP. They run locally in Docker containers.

2. **An LLM** — The extraction code sends cleaned-up data from each MCP server to a language model and asks it to generate natural question-answer pairs. You're using OpenAI's API for this.

3. **The extractor CLI** (`qa-extract`) — A Python command-line tool that orchestrates everything: calls the MCP servers, sends data to the LLM, collects the Q&A pairs, and writes them out as JSONL files.

4. **Argilla** — A human review platform where extracted Q&A pairs are pushed for quality control. Reviewers can approve, reject, or edit pairs. Runs locally via Docker.

```
Docker containers          Your machine             OpenAI API
┌──────────────────┐      ┌──────────────────┐      ┌─────────────┐
│ compute-resources │─────▶│                  │─────▶│             │
│ software-discovery│─────▶│   qa-extract     │      │  gpt-4o     │
│ allocations       │─────▶│   (Python CLI)   │◀─────│             │
│ nsf-awards        │─────▶│                  │      └─────────────┘
│ affinity-groups   │─────▶│                  │
└──────────────────┘      └─────────┬────────┘
     port 3002-3011           │            │
                        writes .jsonl   pushes to
                         files         Argilla
                                    (port 6900)
```

---

## Step 1: Set up Python

```bash
cd access-qa-extraction
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You'll need to activate the virtualenv (`source .venv/bin/activate`) every time you open a new terminal.

---

## Step 2: Start the MCP servers

The MCP servers live in the sibling `access-mcp/` repo. They run as Docker containers:

```bash
cd ../access-mcp
docker compose up -d
```

Verify they're running:

```bash
docker compose ps
```

You should see containers for each server. The ports that matter:

| Server              | Port | What it serves                       |
|---------------------|------|--------------------------------------|
| compute-resources   | 3002 | HPC systems (Delta, Expanse, etc.)   |
| software-discovery  | 3004 | Software installed on each system    |
| allocations         | 3006 | Research project allocations          |
| nsf-awards          | 3007 | NSF funding awards                   |
| affinity-groups     | 3011 | Community groups and events          |

> **Note:** The software-discovery server needs an `SDS_API_KEY` to work. Check `access-mcp/.env` or ask the team if that extractor fails.

---

## Step 2.5: Start Argilla (human review)

Argilla is used for reviewing extracted Q&A pairs. The Docker stack lives in the sibling `access-argilla/` repo:

```bash
cd ../access-argilla
docker compose up -d
```

This starts Argilla at `http://localhost:6900`. Default login: `argilla` / `12345678`.

> **Note:** Argilla is optional for extraction — you can generate JSONL files without it. But you'll need it running to use `--push-to-argilla` or `qa-extract push`.

---

## Step 3: Configure your `.env` file

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real OpenAI key:

```
LLM_BACKEND=openai
OPENAI_API_KEY=sk-your-actual-key-here
```

That's it for configuration. The CLI loads `.env` automatically on startup, so you never need to type these on the command line. The default model is `gpt-4o` — you can change it by adding `OPENAI_MODEL=gpt-4o-mini` (or whatever model you want) to `.env`.

---

## Step 4: Verify everything works

Run the test suite first — this needs no servers or API keys:

```bash
pytest
```

All 59 tests should pass. Then check that the CLI sees all the servers:

```bash
qa-extract list-servers
```

You should see all five servers listed with "Implemented" status.

---

## Step 5: Run an extractor (dry run)

A dry run calls the MCP servers and the LLM, generates Q&A pairs, and prints a summary table — but doesn't write any files. This is the best way to test that things are working:

```bash
qa-extract extract compute-resources --dry-run
```

Try each one individually:

```bash
# Compute resources (HPC systems like Delta, Expanse, Bridges-2)
qa-extract extract compute-resources --dry-run

# Software catalog (what's installed on each system)
qa-extract extract software-discovery --dry-run

# Allocation projects (research grants and their resource allocations)
qa-extract extract allocations --dry-run

# NSF awards (funding details, PIs, institutions)
qa-extract extract nsf-awards --dry-run

# Affinity groups (community groups, events, knowledge base)
qa-extract extract affinity-groups --dry-run
```

---

## Step 6: Write real output

Drop the `--dry-run` flag and the CLI writes `.jsonl` files to `data/output/`:

```bash
qa-extract extract allocations
```

You can specify a custom output directory with `-o`:

```bash
qa-extract extract allocations -o my-output/
```

Run all five extractors at once:

```bash
qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups
```

Add `-c` to combine everything into a single `.jsonl` file instead of one per server:

```bash
qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups -c
```

### Running in the background

A full extraction across all servers can take a while (each project/award gets sent to the LLM individually). To run it in the background so you can close the terminal or do other work:

```bash
qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups > extraction.log 2>&1 &
```

What this does:
- `> extraction.log` — sends all normal output to a file instead of the screen
- `2>&1` — also sends errors to the same file
- `&` — runs the whole thing in the background

You can check on progress while it runs:

```bash
# Follow the log in real time (Ctrl+C to stop watching)
tail -f extraction.log

# Check if the job is still running
jobs
```

When `jobs` shows nothing (or says "Done"), it's finished. The full results will be in `extraction.log` and the JSONL files will be in `data/output/`.

> **Note:** `extraction.log` is in `.gitignore` — it won't be committed.

---

## Step 6.5: Push to Argilla for review

Once you have JSONL output, you can push it to Argilla for human review:

```bash
# Push an existing JSONL file
qa-extract push data/output/compute-resources_qa_pairs.jsonl

# Push without duplicate checking
qa-extract push data/output/allocations_qa_pairs.jsonl --no-dedup
```

Or extract and push in one step:

```bash
qa-extract extract compute-resources --push-to-argilla
```

After pushing, open `http://localhost:6900` in your browser to see the `qa-review` dataset. Reviewers can approve, reject, or edit each Q&A pair.

---

## Step 7: Inspect the output

After writing output, you can look at what was generated:

```bash
# Show stats (domain breakdown, complexity, citation coverage)
qa-extract stats data/output/allocations.jsonl

# Validate that citations reference real entities
# (requires MCP servers to still be running)
qa-extract validate data/output/allocations.jsonl
```

The JSONL files are plain text — one JSON object per line. You can peek at a single record:

```bash
head -1 data/output/allocations.jsonl | python -m json.tool
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on an extractor | MCP server isn't running. Run `docker compose up -d` in `access-mcp/` |
| `ANTHROPIC_API_KEY required` | Your `.env` is missing `LLM_BACKEND=openai` |
| `401 Unauthorized` from OpenAI | Check your `OPENAI_API_KEY` in `.env` |
| software-discovery returns empty | Needs `SDS_API_KEY` in the MCP server's environment |
| `ModuleNotFoundError` | Activate your virtualenv: `source .venv/bin/activate` |

---

## Overriding variables on the fly

Everything is configured through your `.env` file, but sometimes you want to change a setting for just one command without editing the file. You can do that by putting variables before the command:

```bash
OPENAI_MODEL=gpt-4o-mini qa-extract extract allocations --dry-run
```

This uses `gpt-4o-mini` for this one run, while everything else still comes from `.env`. You can override any variable this way — useful for trying a different model, pointing at a different server, etc.

```bash
# Try a cheaper model
OPENAI_MODEL=gpt-4o-mini qa-extract extract nsf-awards --dry-run

# Point at a different MCP server URL
MCP_ALLOCATIONS_URL=http://some-other-host:3006 qa-extract extract allocations --dry-run
```

The full list of variables you can override:

| Variable | What it controls | Default (from `.env`) |
|----------|------------------|-----------------------|
| `LLM_BACKEND` | Which LLM client to use | `openai` |
| `OPENAI_API_KEY` | API key for OpenAI | your key |
| `OPENAI_MODEL` | Which OpenAI model to use | `gpt-4o` |
| `MCP_COMPUTE_RESOURCES_URL` | compute-resources server | `http://localhost:3002` |
| `MCP_SOFTWARE_DISCOVERY_URL` | software-discovery server | `http://localhost:3004` |
| `MCP_ALLOCATIONS_URL` | allocations server | `http://localhost:3006` |
| `MCP_NSF_AWARDS_URL` | nsf-awards server | `http://localhost:3007` |
| `ARGILLA_URL` | Argilla server | `http://localhost:6900` |
| `ARGILLA_API_KEY` | Argilla API key | `argilla.apikey` |
| `MCP_AFFINITY_GROUPS_URL` | affinity-groups server | `http://localhost:3011` |
