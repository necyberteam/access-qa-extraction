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
| 1 | affinity-groups + comparisons | affinity-groups:cloudbank.access-ci.org_3; cmp_feat_gpu_computing + 4 other comparison pairs | `temporal-assumption` | LLM described past events as "upcoming" — it has no way to know whether event dates in the source data are future or past. **Full corpus scan: 6 affected pairs total.** Notably 5 of the 6 are in programmatically-generated comparison pairs containing "(coming soon)" for PNRP/Nexus — different root cause from the LLM issue. | (a) DOMAIN_NOTES prompt instruction: don't use temporal language for events. (b) Fix comparison generator to strip "(coming soon)" suffixes from resource names. (c) Strip event dates from source data. | `open` | Two distinct root causes: LLM temporal framing (affinity-groups) vs. "(coming soon)" baked into comparison pair text by `ComparisonGenerator`. Fix (b) targets the comparison issue specifically. |
| 2 | all domains (worst: nsf-awards) | nsf-awards_2529183_1 and 26 others | `decontextualized-question` | Questions use "this project / this award / this resource" with no entity name. **Full corpus scan: 27/176 pairs (15.3%).** NSF awards worst: 17/38 pairs. At RAG retrieval time: (1) weak semantic signal vs. a user query containing the actual name/ID; (2) at inference time, "this" is ambiguous across multiple retrieved chunks. Root cause: FIELD_GUIDANCE instructions themselves use "this [X]" framing, seeding the pattern. | See P1 preferred fix — entity name interpolation in user prompt. | `open` | Confirmed pervasive. P1 fix (entity name in user prompt header) is the right remedy. |
| 3 | compute-resources | compute-resources_nexus.gatech.access-ci.org_2 | `other` | Zero-confidence pair (faithfulness 0.5, completeness 0.0, marked `needs_review`). Q: "Who operates the Nexus platform?" A: "Operated by Georgia Institute of Technology." Answer is factually correct but judge gave completeness 0.0. Either the judge expected more context (NCSA partnership, operational details) or the answer is genuinely thin for this question. | (a) Expand answer with additional operational context from source data. (b) Investigate whether judge model is systematically over-strict on completeness for short factual answers. | `open` | Only 1 zero-confidence pair in 176 (0.6%). May be a judge calibration issue rather than a content problem. |

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

**Preferred fix — entity name interpolation in the user prompt (promising, not yet implemented):**

`build_user_prompt()` in `question_categories.py` is called per-entity inside every extractor's loop — it already receives `entity_id` and `entity_json`. The proposal is to also pass `entity_name` (the human-readable name, e.g. the project title or resource name) and surface it prominently at the top of the user message:

```
Entity ID: 2543343
Entity name: Advanced Computational Methods for Climate Science
Citation marker: <<SRC:nsf-awards:2543343>>

Data:
{ ... }
```

With the name right there alongside the citation marker, the LLM will reach for it naturally rather than defaulting to "this project." The system prompt (built once per domain run) doesn't need to change; only the per-entity user prompt does. Each extractor already has the name in scope (`resource_name`, `title`, `name` depending on domain), so the wiring cost is low. This is preferred over a rule-based instruction ("don't say 'this project'") because it gives the LLM the correct value to use rather than just a prohibition.

---

## Resolved Issues

*Move rows here (with resolution notes) once `verified`.*

