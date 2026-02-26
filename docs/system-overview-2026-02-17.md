# ACCESS-CI Q&A Extraction Pipeline — System Overview

**Updated**: 2026-02-26
**Branch**: `feat/two-shot`
**Tests**: 144/144 passing

## What This System Does

Extracts Q&A training pairs from 5 ACCESS-CI data domains (HPC resources, software catalog, allocations, NSF awards, community groups) for use in a RAG-based Q&A retrieval system. Produces structured JSONL files with 2 granularity levels (comprehensive + comparison).

## Pipeline Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        MCP1["compute-resources<br/>MCP :3002<br/>23 entities"]
        MCP2["software-discovery<br/>MCP :3004<br/>1,404 entities"]
        MCP3["allocations<br/>Direct API<br/>5,440 entities"]
        MCP4["nsf-awards<br/>Direct API<br/>10,000+ entities"]
        MCP5["affinity-groups<br/>MCP :3011<br/>55 entities"]
    end

    subgraph "Extraction Layer"
        E1["ComputeResourcesExtractor"]
        E2["SoftwareDiscoveryExtractor"]
        E3["AllocationsExtractor"]
        E4["NsfAwardsExtractor"]
        E5["AffinityGroupsExtractor"]
    end

    subgraph "Generation Layer (per entity)"
        direction TB
        HASH["compute_entity_hash()"]
        CACHE{"IncrementalCache<br/>unchanged?"}
        BAT["LLM Pass 1: Battery<br/>Guaranteed field coverage<br/>→ comprehensive"]
        DISC["LLM Pass 2: Discovery<br/>What did battery miss?<br/>→ comprehensive"]
        JUDGE["LLM Pass 3: Judge<br/>Score each pair<br/>→ scores + decisions"]
        CACHED["Replay<br/>cached pairs"]
    end

    subgraph "Cross-Entity"
        COMP["ComparisonGenerator<br/>Programmatic<br/>→ comparison"]
    end

    subgraph "Output + Review"
        JSONL["JSONL files<br/>per domain + comparisons"]
        ARGILLA["Argilla<br/>entity-replace push"]
    end

    MCP1 --> E1
    MCP2 --> E2
    MCP3 --> E3
    MCP4 --> E4
    MCP5 --> E5

    E1 & E2 & E3 & E4 & E5 --> HASH
    HASH --> CACHE
    CACHE -->|yes| CACHED
    CACHE -->|no| BAT
    BAT --> DISC
    DISC --> JUDGE
    CACHED --> JSONL
    JUDGE --> JSONL
    JSONL --> ARGILLA

    E1 & E2 & E3 & E4 & E5 -->|raw_data| COMP
    COMP --> JSONL
```

## Per-Entity Extraction Sequence

```mermaid
sequenceDiagram
    participant CLI as CLI (qa-extract)
    participant EXT as Extractor
    participant CACHE as IncrementalCache
    participant MCP as MCP Server / API
    participant LLM as LLM
    participant JUDGE as Judge LLM (cheap)
    participant OUT as JSONL Writer

    CLI->>EXT: extract(domain, max_entities)
    EXT->>MCP: fetch entity list
    MCP-->>EXT: [entity_1, entity_2, ...]

    loop Each entity
        EXT->>EXT: clean data, compute hash

        alt --incremental enabled
            EXT->>CACHE: is_unchanged(domain, id, hash)?
            alt unchanged
                CACHE-->>EXT: cached pairs + scores
                EXT->>OUT: write cached pairs
            else changed or new
                CACHE-->>EXT: false
                Note over EXT,LLM: Proceed to generation
            end
        end

        rect rgb(230, 245, 255)
            Note over EXT,LLM: Pass 1 — Battery (guaranteed coverage)
            EXT->>LLM: battery prompt (per-domain field groups) + entity JSON
            LLM-->>EXT: JSON array [{question, answer}]
        end

        rect rgb(235, 250, 240)
            Note over EXT,LLM: Pass 2 — Discovery (what did battery miss?)
            EXT->>LLM: discovery prompt (existing pairs + entity JSON)
            LLM-->>EXT: JSON array [{question, answer}] — additional pairs
        end

        rect rgb(255, 230, 230)
            Note over EXT,JUDGE: Pass 3 — Judge Evaluation
            EXT->>JUDGE: all pairs + source_data
            JUDGE-->>EXT: faithfulness, relevance, completeness per pair
            EXT->>EXT: confidence = min(3 scores), suggested_decision
        end

        EXT->>CACHE: store(domain, id, hash, all_pairs + scores)
        EXT->>OUT: write all pairs
    end

    rect rgb(245, 235, 255)
        Note over EXT,OUT: Cross-Entity — Comparisons (programmatic, no LLM)
        EXT->>EXT: group entities by shared attributes
        EXT->>OUT: comparison QAPairs (granularity=comparison)
    end
```

## Two Granularity Levels

| Granularity | Generator | LLM? | Purpose | Example Q |
|---|---|---|---|---|
| **comprehensive** | Two-shot LLM (battery + discovery) | Yes | Entity-specific questions, variable count driven by data richness | "What is Delta and what is it designed for?" |
| **comparison** | ComparisonGenerator | No | Cross-entity questions | "Which ACCESS resources support interactive computing?" |

## Two-Shot Extraction Strategy

Each entity gets two LLM passes:

1. **Battery** — per-domain field groups guarantee coverage of key areas (hardware specs, organizations, access methods, etc.). Produces one Q&A pair per applicable field group.
2. **Discovery** — sees the battery output and asks "what interesting, useful Q&A pairs can you find that aren't already covered?" Produces additional pairs for entity-specific details the battery template didn't anticipate.

This replaced the original fixed-category approach, which constrained the LLM to a rigid menu. Two-shot produces more pairs for data-rich entities and fewer for thin ones — pair count scales with data richness.

## Argilla Entity-Replace

When pushing pairs to Argilla for human review:

1. Group pairs by `source_ref` (e.g., `mcp://compute-resources/resources/delta.ncsa.access-ci.org`)
2. For each source_ref, delete existing Argilla records
3. Before deleting, archive any records with human annotations to `qa-review-archive-superseded`
4. Push fresh records

This means the incremental cache is the decision boundary: unchanged entity → skip entirely (human annotations survive). Changed entity → re-extract, replace in Argilla (stale annotations archived).

## LLM Judge Evaluation

After both LLM passes for an entity, all pairs are sent as a batch to a **judge LLM** (cheaper model: gpt-4o-mini or claude-haiku). Scores each pair on three dimensions (0.0–1.0):

- **Faithfulness** — does the answer match the source data?
- **Relevance** — does the answer address the question?
- **Completeness** — does the answer cover the key facts?

**Confidence** = min(faithfulness, relevance, completeness). Threshold 0.8 → `suggested_decision = "approved"` or `"needs_review"`. Skip with `--no-judge`.

## Full-Scale Cost Estimate

Costs assume **gpt-4o-mini** for all three passes (battery, discovery, judge). Two-shot means 2 extraction calls + 1 judge call per entity (3 LLM calls total).

| Domain | Entities | Est. Pairs | Est. Cost (gpt-4o-mini) |
|---|---|---|---|
| Compute Resources | 23 | ~345 | ~$0.28 |
| Software Discovery | 1,404 | ~19,000 | ~$16.80 |
| Affinity Groups | 55 | ~670 | ~$0.66 |
| Allocations | 5,440 | ~79,000 | ~$65.20 |
| NSF Awards | 10,000+ | ~145,000 | ~$120.00 |
| **Total** | **~17K** | **~244K** | **~$203** |

With `--incremental`, re-runs cost ~$0 for unchanged entities. Judge scores are cached alongside pairs.

**Model alternatives:** Using `claude-haiku` for judge only while keeping `gpt-4o-mini` for extraction would have minimal cost impact (~$7 for judge across all entities). Using `gpt-4o` or `claude-sonnet` for extraction would increase costs ~10-20x.

## Key Files

```
src/access_qa_extraction/
├── cli.py                          # Typer CLI: extract, list-servers, stats, validate, push
├── config.py                       # ExtractionConfig, MCPServerConfig
├── models.py                       # QAPair, QAMetadata
├── mcp_client.py                   # Async HTTP client for MCP servers
├── llm_client.py                   # Anthropic / OpenAI / Local / Transformers
├── question_categories.py          # Per-domain field groups, battery/discovery prompt builders
├── citation_validator.py           # Validates <<SRC:domain:id>> citations
├── argilla_client.py               # Entity-replace push to Argilla for human review
├── extractors/
│   ├── base.py                     # BaseExtractor (incremental cache slot)
│   ├── compute_resources.py        # MCP, search_resources({})
│   ├── software_discovery.py       # MCP, list_all_software
│   ├── allocations.py              # Direct API pagination (httpx)
│   ├── nsf_awards.py               # Direct API pagination (httpx)
│   └── affinity_groups.py          # MCP, search_affinity_groups({})
├── generators/
│   ├── comparisons.py              # ComparisonGenerator (programmatic, all 5 domains)
│   ├── incremental.py              # IncrementalCache + compute_entity_hash()
│   └── judge.py                    # LLM judge evaluation (faithfulness/relevance/completeness)
└── output/
    └── jsonl_writer.py             # JSONL file writer
```

## CLI Quick Reference

```bash
# Full extraction (all domains)
qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups

# Cheap test run (2 entities per domain, with judge + incremental + Argilla push)
qa-extract extract compute-resources --max-entities 2 --incremental --push-to-argilla

# Skip judge evaluation
qa-extract extract allocations --no-judge

# Incremental only (skip unchanged entities)
qa-extract extract allocations --incremental

# Push existing JSONL to Argilla
qa-extract push data/output/compute-resources_qa_pairs.jsonl

# Inspect output
qa-extract stats data/output/compute-resources_qa_pairs.jsonl
qa-extract validate data/output/compute-resources_qa_pairs.jsonl
```
