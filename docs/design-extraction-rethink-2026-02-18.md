# Design: Extraction Approach Rethink + Argilla-as-Cache

**Date**: 2026-02-18 (updated 2026-02-19)
**Branch**: `spike/freeform-extraction` (implemented)
**Status**: Implemented. Freeform extraction running across all 5 domains. Argilla push strategy decided: entity-replace (see Part 1.5 and Part 2).

---

## Motivation

After running the full 4-pass pipeline against 2 compute-resource entities (Nexus, PNRP) and reviewing the output in Argilla, two problems became clear:

1. **Fixed categories leave good data on the table.** Compute-resources has 6 categories (overview, organization, gpu_hardware, cpu_hardware, capabilities, access). For data-rich entities like PNRP — which has multi-institution partnerships, 100Gbps+ networking, FPGA accelerators, and tape-backed archival storage — the LLM can only generate one Q&A pair per category. The bonus pass adds up to 3 more, but that's not enough. Meanwhile, simpler entities get the same rigid treatment.

2. **The ID scheme no longer serves its purpose.** The `{domain}_{entity_id}_{category}` IDs were designed for Argilla upsert — update specific records without creating duplicates. But: (a) the hash-based incremental cache makes partial entity updates unnecessary (it's binary: skip or re-extract everything for an entity), and (b) upsert would actually be harmful once humans start annotating in Argilla, because it would overwrite their work.

These two problems connect to a third, larger question: **how should Argilla fit into the system lifecycle?**

---

## Part 1: Extraction Approach

### Current approach (baseline)

Each entity goes through 4 passes:

| Pass | What | LLM? | Output |
|------|------|------|--------|
| 1. Comprehensive | Fixed category menu (5-6 per domain) | Yes | 1 pair per applicable category |
| 2. Factoid | Templates with field interpolation | No | 6-8 factoid pairs |
| 3. Bonus | "Find what the categories missed" | Yes | 0-3 exploratory pairs |
| 4. Judge | Score each pair on faithfulness/relevance/completeness | Yes (cheap) | Scores + suggested_decision |

**Problems observed:**
- Pass 1 constrains the LLM to a fixed menu. Data-rich entities get the same 6 categories as data-poor ones.
- Pass 2 factoid templates are rigid — they can only see structured fields, but much of the richness lives in free-text descriptions. They also inherit data quality issues (e.g., "COMING SOON" in entity names).
- Pass 3 is capped at 3, and is framed as "what did the categories miss?" — a mop-up pass instead of a first-class extraction.
- Two LLM calls (Pass 1 + Pass 3) for extraction when one could do it.

### Proposed approach

Replace Passes 1 and 3 with a single, freer LLM pass. Keep Pass 2 (factoid templates) and Pass 4 (judge).

| Pass | What | LLM? | Output |
|------|------|------|--------|
| 1. LLM extraction | "Generate all useful Q&A pairs from this entity" | Yes | Variable count, driven by data richness |
| 2. Factoid | Templates (unchanged, maybe improved) | No | Deterministic single-fact pairs |
| 3. Judge | Score each pair | Yes (cheap) | Scores + suggested_decision |

**Key changes in the LLM pass:**
- No fixed category menu. Instead, give the LLM the entity data and ask it to generate as many useful Q&A pairs as the data warrants.
- Provide guidance about what "useful" means (researcher-facing questions, specific facts, no vague generalities) but don't constrain the topics.
- Let the count vary: a data-rich entity like PNRP might produce 12-15 pairs. A simple entity might produce 4-5.
- The prompt should still require citation markers and use-only-provided-data rules.

**What stays the same:**
- Factoid templates (Pass 2) — cheap, deterministic, zero hallucination risk. Good for precise single-fact retrieval. May improve templates to handle data quality issues better.
- Judge evaluation (Pass 3/4) — scores every pair regardless of how it was generated.
- Incremental cache — hash-based, binary skip/re-extract per entity.
- Comparison generator — programmatic cross-entity pairs, no LLM.
- Citation validation.

**What this gains:**
- Data-rich entities get proportionally more coverage.
- One LLM call instead of two per entity (saves cost and latency).
- Removes the awkward "mop-up" framing of the bonus pass.
- The LLM isn't fighting a category constraint — it can surface entity-specific details naturally.

**What this loses:**
- Stable IDs per category (but we've argued these are no longer needed).
- Guaranteed coverage of specific topics (but the judge + factoid templates partially fill this gap).

### Experiment plan

1. Create new branch off `spike/quality-incremental-bonus`.
2. Write the new single-pass LLM prompt.
3. Run against the same 2 entities (Nexus, PNRP) with `--max-entities 2`.
4. Compare output against the baseline saved in `data/output/baseline-categories-2entity/`.
5. Evaluate: more coverage? Better quality? Does the judge still work well?

### Experiment results (2026-02-18)

Ran freeform vs baseline on compute-resources with 2 entities (PNRP, Ranch). Then rolled out to all 5 domains.

**Compute-resources comparison (PNRP + Ranch):**

| Metric | Baseline (categories+bonus) | Freeform |
|---|---|---|
| PNRP comprehensive | 6 pairs | 10 pairs |
| PNRP bonus | 3 pairs | — (merged into freeform) |
| PNRP factoid | 6 pairs | 6 pairs |
| PNRP total | 15 | 16 |
| PNRP avg confidence | 0.88 | 0.97 |
| LLM calls per entity | 3 (category + bonus + judge) | 2 (freeform + judge) |

Freeform produced more pairs, higher judge scores, and better coverage of entity-specific details. One fewer LLM call per entity.

**Full 5-domain run (2 entities per domain, `--push-to-argilla`):**

| Domain | Entities | Comprehensive | Factoid | Total |
|---|---|---|---|---|
| compute-resources | Ranch, PNRP | 16 | 12 | 28 |
| software-discovery | Abaqus, ABINIT | 20 | 14 | 34 |
| allocations | 72097, 72096 | 23 | 16 | 39 |
| nsf-awards | 2529183, 2449122 | 22 | 16 | 38 |
| affinity-groups | REPACSS, Voyager | 10 | 10 | 20 |
| comparisons | (programmatic) | — | — | 3 |
| **Total** | | | | **162** |

All 162 pairs scored `suggested_decision: "approved"`. Output in `data/output/all-domains-freeform/`.

**Cleanup needs identified per domain:**
- **allocations**: Factoid template grammar — "uses 1 resources" (singular/plural)
- **nsf-awards**: Raw `primary_program` field leaks codes like `"01002526DB NSF RESEARCH & RELATED ACTIVIT"`; co-PI factoid includes raw email addresses
- **affinity-groups**: Thin source data → comprehensive/factoid overlap; obfuscated email (`[at]`) passed through to answers
- **compute-resources**: `fq_description` factoid truncates Ranch description (describes the problem, not Ranch itself)
- **software-discovery**: Cleanest domain, rich source data

---

## Part 1.5: Argilla Dedup Discovery

The `--push-to-argilla` flag on the full 5-domain run resulted in **0 records pushed, 162 skipped as duplicates**. Investigation revealed a bug in the dedup logic:

### How dedup currently works

`ArgillaClient.push_pairs()` (argilla_client.py:286) iterates each pair and calls `find_duplicate()` before pushing:

```python
# argilla_client.py:211-229
def find_duplicate(self, question_embedding: list[float]) -> bool:
    similar_query = rg.Query(
        similar=rg.Similar(
            name="question_embedding",
            value=question_embedding,
        )
    )
    similar_records = dataset.records(similar_query).to_list(flatten=True)
    return len(similar_records) > 0  # ← BUG: no threshold check
```

1. Each question gets an embedding via `all-MiniLM-L6-v2` (384-dim)
2. Argilla's vector similarity query returns nearest neighbors
3. **If ANY record comes back, it's treated as a duplicate** — regardless of distance
4. `SIMILARITY_THRESHOLD = 0.85` is defined on line 20 but **never used in the query**

### Why everything matched

Old records from prior (category-based) extraction runs are still in Argilla. New freeform questions like "What is Abaqus used for?" are semantically close to existing category-based questions like "What is Abaqus and what is it designed for?" — so Argilla returns them as neighbors, and the code flags them as duplicates.

### Why semantic dedup won't work for us

The fundamental problem isn't the missing threshold — it's that Argilla's semantic dedup **only compares questions, not answers**. When upstream data changes (e.g., a PI changes on an award), the question "Who is the principal investigator on award X?" is still semantically identical. Argilla's dedup would block the updated answer from ever getting pushed, which is the opposite of what we want.

Semantic dedup is designed for workflows where you're accumulating records from overlapping sources (CSVs, scrapers) and want to avoid duplicates organically. Our use case is different: we have structured entities with known `source_ref` values and need to replace stale answers when data changes.

### Decision: Entity-replace (not semantic dedup)

**Entity-replace** is the right approach: when source data changes, delete all Argilla records for that entity by `source_ref` and push fresh. This makes dedup moot for re-extractions. See Part 2 for the full design.

---

## Part 2: Argilla as the Canonical Store

### The question

Right now the data flow is:

```
MCP servers → extractors → JSONL files → (push) → Argilla
```

Argilla is a downstream consumer. But the planning spec (03-review-system.md) envisions Argilla as the canonical system of record — the place where approved Q&A pairs live and get synced to the RAG database. That creates a lifecycle question: **what happens when upstream data changes?**

### The hard problem

Two scenarios conflict:

1. **Upstream data changes** (MCP server returns updated entity data). We want to re-extract Q&A pairs for that entity, because the old answers may be stale.

2. **A human has annotated records in Argilla** (approved, edited, rated). We don't want to destroy their work.

These conflict when both happen for the same entity: the data changed AND a human already reviewed the old extraction.

### Decided model: Replace-by-entity

When re-extracting an entity whose source data has changed:

1. **Delete all Argilla records for that entity** — identified by `source_ref` (e.g., `mcp://compute-resources/resources/delta.ncsa.access-ci.org`). This includes all granularity levels (comprehensive, factoid, exploratory).
2. **Push the new extraction** — fresh records with new IDs, new scores, no annotations.
3. **Human annotations on the deleted records are lost.** This is intentional: the underlying data changed, so the old answers (and therefore the old annotations) are stale.

**Why this works:**
- The incremental cache already implements the decision boundary: if the entity hash is unchanged, skip it entirely (human annotations survive). If the hash changed, re-extract everything (stale annotations are discarded).
- This is simpler than trying to diff and merge. No upsert logic, no conflict resolution.
- Comparisons are always regenerated (they're programmatic and free).

**What Argilla is for:**
- **Quality gate**, not upstream data correction. Reviewers approve/reject/edit Q&A pairs.
- If a reviewer edits an answer, that edit improves the Q&A pair for RAG, but it doesn't change the source data.
- If source data is wrong, the fix happens upstream (in the MCP server's data source), not in Argilla.

**Human edits survive when:**
- The entity's source data hasn't changed. The cache skips it, Argilla records untouched.

**Human edits are lost when:**
- The entity's source data changed. The old pairs are replaced. The reviewer will need to re-review the new extraction — but the answers are based on updated data, so re-review is appropriate.

### Annotation preservation at push time

Before entity-replace destroys records, we should check whether any have human annotations (approvals, edits, ratings). If they do, **save them before deleting**. This is push-layer logic — extraction stays a pure function, unaware of Argilla.

**Why this matters:** A reviewer may have spent real time annotating pairs. If the entity's source data changes, the old answers are stale and replacement is correct. But the annotations themselves — editorial judgments, corrected phrasing, quality ratings — have value. They can inform re-review of the new extraction or serve as an audit trail.

**Proposed flow:**
1. Before deleting records for a `source_ref`, query Argilla for existing records with that `source_ref`
2. Filter for records with human annotations (`response.status == "submitted"`)
3. If annotated records exist, copy them to an archive dataset in Argilla (see naming below)
4. Log a warning: "Replacing N annotated records for {source_ref} — archived to {archive_dataset_name}"
5. Proceed with delete + push fresh to the live dataset

**Archive dataset:** A single Argilla dataset named `qa-review-archive-superseded` (mirrors the live `qa-review` dataset). The `domain` metadata field distinguishes domains, same as in the live dataset. The dataset description should state: "Archived Q&A pairs replaced by re-extraction. Not used for RAG. Preserved for reference during re-review."

**Why an Argilla dataset (not local JSON):** Reviewers work in Argilla. When re-reviewing a fresh extraction, they can open the archive dataset side-by-side to see their prior annotations — what they approved, what they edited, what they flagged. This keeps the workflow in one tool.

**Archive record metadata** should include:
- `archived_at` — timestamp of when the record was superseded
- `replaced_reason` — "source_data_changed" (the entity hash changed upstream)
- `annotation_depth` — "approved_only" (rubber-stamp, no edits) vs "has_edits" (human rewrote question/answer or left rejection notes). Determined by checking whether `edited_question`, `edited_answer`, or `rejection_notes` response fields are non-empty. This lets reviewers filter the archive for records where a human actually did substantive work, vs. ones that were just clicked through.
- All original metadata preserved (judge scores, source_data, etc.)

**What this does NOT do:**
- Does not merge old annotations into new records (the data changed, so old annotations may not apply)
- Does not block replacement (archiving is informational, not a gate)
- Does not add Argilla as a dependency for extraction (this is purely push-layer)
- Does not feed into RAG — the archive dataset is strictly for reviewer reference

### Implementation needed

The deletion key is `source_ref` — every record for a given entity shares the same `source_ref` (e.g., `mcp://compute-resources/resources/delta.ncsa.access-ci.org`) regardless of granularity level or ID scheme. This makes entity-level replacement straightforward.

To support replace-by-entity, `ArgillaClient` needs:

1. **`delete_records_by_source_ref(source_ref: str)`** — Delete all records matching a specific `source_ref` metadata value. Argilla SDK supports filtering records by metadata and deleting them.

2. **Modified `push_pairs()` flow** — When pushing pairs for an entity that was re-extracted (not cached), first delete existing records for that `source_ref`, then push new ones.

3. **Comparison replacement** — Comparisons are programmatic (no LLM cost), so the simplest approach is to delete all comparison records for the domain and regenerate them all. No need to track which specific comparisons reference which entities.

### Not needed yet

- Argilla webhook sync to RAG (Phase 6 in planning spec) — that's a separate piece of work.
- CILogon auth — not needed for local dev.
- Post-deployment feedback dataset — separate concern.

---

## Part 3: ID Strategy

### Current

IDs follow `{domain}_{entity_id}_{category}` for comprehensive pairs, `{domain}_{entity_id}_bonus_{n}` for exploratory, `{domain}_{entity_id}_{template_name}` for factoid.

### Proposed

Since we no longer need stable category-based IDs:

- **LLM-generated pairs**: `{domain}_{entity_id}_{seq_n}` where `seq_n` is a simple sequence number (1, 2, 3...). These IDs are ephemeral — they change on every re-extraction.
- **Factoid pairs**: Keep `{domain}_{entity_id}_{template_name}` since these are deterministic.
- **Comparison pairs**: Keep current scheme (`cmp_feat_{feature_slug}`).

IDs exist for Argilla record identity and JSONL deduplication, not for stable cross-run tracking. The incremental cache handles cross-run continuity at the entity level.

---

## Summary of Changes

| Aspect | Current | Proposed |
|--------|---------|----------|
| LLM extraction | 2 passes (categories + bonus) | 1 pass (open-ended) |
| Pairs per entity | Fixed (categories) + 0-3 bonus | Variable, driven by data richness |
| Factoid templates | Kept | Kept (improve data quality handling) |
| Judge | Kept | Kept |
| IDs | Stable per category | Ephemeral sequence numbers |
| Argilla updates | Push (append) | Replace-by-entity (delete + push) |
| Incremental cache | Hash-based skip/re-extract | Unchanged |
| Comparisons | Programmatic | Unchanged (always regenerated) |

---

## Open Questions for Andrew

1. **Variable pair count** — Is it OK that data-rich entities produce more Q&A pairs than data-poor ones? Or do we want rough parity across entities?

2. **Factoid value** — The factoid templates produce precise single-fact pairs (e.g., "What type of resource is Delta?" → "Compute"). Are these still valuable for RAG retrieval, or does the LLM pass cover them better? Some factoid templates have data quality issues (upstream data with "COMING SOON", raw program codes, PI emails) that need cleanup either way.

3. **PI emails in training data** — NSF co-PI entries include inline emails ("Jane Doe jane@mit.edu"). Strip before sending to LLM, or leave as-is? Might actually be useful for a researcher-facing system.

4. **Comparison scope** — Comparisons are currently cross-entity within a domain. Should they be replaced when *any* entity in the domain changes, or only regenerated on full domain re-extraction?

### Answered / Decided

- **Replace vs. preserve** — Replace-by-entity. Human annotations on changed entities are lost intentionally (stale data). Semantic dedup doesn't work for us because it only compares questions, not answers.
- **Stable IDs** — Not needed. Entity-level replacement makes per-category IDs moot.
