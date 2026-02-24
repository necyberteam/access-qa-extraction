# QA Review Observations

Issues, patterns, and anomalies noticed during manual Argilla review.
Used to inform prompt improvements, extractor fixes, and Andrew conversations.

**How to use:**
- Add a row per issue (or per pattern if it repeats across entities)
- `Status`: `open` → `fix-applied` → `verified` (or `wont-fix` with a note)
- Link to a commit or PR when a fix lands

---

## Issue Log

| # | Domain | Entity / Pair ID | Issue Type | Description | Status | Notes |
|---|--------|-----------------|------------|-------------|--------|-------|
| — | — | — | — | *Start logging here* | — | — |

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
| `other` | Anything else — describe in Notes |

---

## Patterns / Themes

*Use this section for cross-entity patterns that don't fit cleanly into a single row.*

---

## Resolved Issues

*Move rows here (with resolution notes) once `verified`.*

