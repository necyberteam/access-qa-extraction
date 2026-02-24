# QA Review Observations

Issues, patterns, and anomalies noticed during manual Argilla review.
Used to inform prompt improvements, extractor fixes, and Andrew conversations.

**How to use:**
- Add a row per issue (or per pattern if it repeats across entities)
- `Status`: `open` → `fix-applied` → `verified` (or `wont-fix` with a note)
- Link to a commit or PR when a fix lands

---

## Issue Log

| # | Domain | Entity / Pair ID | Issue Type | Description | Possible Solutions | Status | Notes |
|---|--------|-----------------|------------|-------------|--------------------|--------|-------|
| 1a | affinity-groups | affinity-groups:cloudbank.access-ci.org_3 | `temporal-assumption` | LLM described past events as "upcoming" — it cannot know whether event dates are in the future or past. | DOMAIN_NOTES instruction added (commit `28be413`): don't describe events as "upcoming/scheduled", use neutral language. Field guidance instruction also changed from "offer" to "host or organize". | `fix-applied` | Verify on next extraction run. |
| 1b | compute-resources (comparisons) | cmp_feat_gpu_computing + 4 others | `stale-status` | Distinct root cause from 1a: "(coming soon)" status label baked into programmatically-generated comparison pair text via `raw_data["name"]`. Not a temporal-language issue — a stale status indicator. | `_clean_name()` in compute_resources extractor now strips `\(coming soon\)` so comparison pairs get clean resource names (commit `28be413`). | `fix-applied` | Verify on next extraction run. |
| 2 | all domains (worst: nsf-awards) | nsf-awards_2529183_1 and 26 others | `decontextualized-question` | Questions use "this project / this award / this resource" with no entity name. **Full corpus scan: 27/176 pairs (15.3%).** NSF awards worst: 17/38 pairs. At RAG retrieval time: (1) weak semantic signal vs. a user query containing the actual name/ID; (2) at inference time, "this" is ambiguous across multiple retrieved chunks. Root cause: FIELD_GUIDANCE instructions themselves use "this [X]" framing, seeding the pattern. | See P1 preferred fix — entity name interpolation in user prompt. | `open` | Confirmed pervasive. P1 fix (entity name in user prompt header) is the right remedy. |
| 3 | compute-resources | compute-resources_nexus.gatech.access-ci.org_2 | `other` | Zero-confidence pair (faithfulness 0.5, completeness 0.0, marked `needs_review`). Q: "Who operates the Nexus platform?" A: "Operated by Georgia Institute of Technology." **Judge is correct** — the source description explicitly states Nexus is "jointly developed by Georgia Tech and NCSA." The answer omits NCSA entirely, likely because `organization_names` only lists Georgia Tech and the LLM answered from that field rather than reading the full description. | The P1 entity-name fix may help indirectly (name in prompt = LLM reads more carefully). Longer term: ensure battery prompt instruction for the "organization" field group directs the LLM to use the full description, not just `organization_names`. | `open` | Judge calibration is fine. This is a real content gap. |

---

## Issue Types (reference)

| Type | Meaning |
|------|---------|
| `hallucination` | Answer contains a claim not in source data |
| `bad-citation` | `<<SRC:...>>` references wrong or missing entity |
| `empty-answer` | Answer is vague, evasive, or essentially blank |
| `wrong-entity` | Question/answer is about a different entity than the source_ref |
| `bad-grammar` | Singular/plural mismatch, awkward phrasing |
| `thin-data` | Source entity has too little data for useful Q&A — structural fix needed |
| `prompt-drift` | LLM ignored instructions (e.g., generated factoids instead of freeform) |
| `scope-creep` | Pair answers something outside the entity's actual data |
| `duplicate` | Near-identical pair within same entity (battery + discovery overlap) |
| `formatting` | Answer has markdown artifacts, excessive bullets, etc. |
| `temporal-assumption` | LLM uses time-relative language ("upcoming", "recent", "current") for data that may be stale |
| `decontextualized-question` | Question uses pronouns or deictic references ("this project", "this resource") instead of naming the entity — breaks semantic retrieval and confuses consuming models |
| `other` | Anything else — describe in Notes |

---

## Patterns / Themes

### Positive findings

- **Corpus-level health** (full scan, 176 pairs): 100% citation accuracy, 99.4% confidence at or above acceptable threshold (≥0.75). No hallucinations detected.
- **Long abstract extraction works** (allocations:72170): LLM correctly surfaced both team members from a ~600-word abstract where the Team section was buried at the bottom. "Dan Jurafsky, a renowned professor" — verified against source data, fair summary of his listed credentials. Judge approved.

### P1 — Questions must be self-contained

A Q&A pair should work as a standalone unit with no surrounding context. At retrieval time the question is matched against a user query; at inference time it is served as context to a consuming model. Both scenarios require that the question unambiguously identifies the entity it is about.

**Failing patterns:**
- Deictic references: "this project", "this resource", "this group" (see issue #2)
- Implicit temporal framing: "upcoming events", "recent awards" (see issue #1)
- Bare pronouns: "What does it offer?", "Who leads it?"

**Desired pattern:** "What events does the [Group Name] affinity group host?" / "The [Group Name] affinity group hosts..." — entity named in both Q and A, temporally neutral.

**Why name it in both Q and A?** Traditional Q&A lets the reader hold the question in mind while reading the answer, so the A can say "its funding is $X." RAG breaks this assumption in two ways: (1) retrieval matches the question against a user query by embedding — a generic "this project" has weak signal vs. the actual project name; (2) at inference time the consuming model reasons over multiple retrieved pairs simultaneously, so answers referencing "this project" become ambiguous across chunks. The redundancy (entity named in both) is intentional in RAG training data.

**Self-contained answers also broaden retrieval surface area.** Most RAG systems embed the full Q+A chunk (not just the Q), so retrieval can match on either half. A well-named answer serves multiple query shapes simultaneously:
- "Who is the PI for the Foo Project?" → matches the Q
- "Who is Bob Jones?" → matches the A (his name is there)
- "What projects is Bob Jones involved in?" → also matches the A

A thin answer ("Bob Jones.") only serves the first shape. This means naming the entity in the answer isn't just about clarity at inference time — it also increases the chance of retrieval for reverse-lookup and entity-centric queries that don't mirror the Q's phrasing.

**Fix implemented (commit `28be413`):** Entity name interpolation in user prompt + explicit rule in battery and discovery prompts.

`build_user_prompt()` now accepts `entity_name` and surfaces it at the top of every user message:

```
Entity ID: 2543343
Entity name: Advanced Computational Methods for Climate Science
Citation marker: <<SRC:nsf-awards:2543343>>

Data:
{ ... }
```

All 5 extractors pass the human-readable name (`resource_name`, project `title`, award `title`, group `name`, software display `name`). Battery and discovery system prompts both have rule 7: "Always refer to the entity by the name given in 'Entity name:' — never use 'this project', 'this resource', etc." Combination of positive signal (here's the name to use) + prohibition (don't use deictic references) is more robust than either alone.

---

## Resolved Issues

*Move rows here (with resolution notes) once `verified`.*

