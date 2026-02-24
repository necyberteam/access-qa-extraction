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
| 1 | affinity-groups | unknown | `temporal-assumption` | LLM described past events as "upcoming" — it has no way to know whether event dates in the source data are in the future or past, so it defaults to assuming they're upcoming. Factually correct data, wrong temporal framing; could mislead the consuming model. | (a) Add DOMAIN_NOTES prompt instruction: don't use temporal language like "upcoming/recent/current" for events — just describe them neutrally. (b) Strip event dates from source data entirely so the LLM can't hallucinate temporal context. (c) Post-process: flag or reject pairs containing "upcoming", "recent", "current" in answers for affinity-groups. | `open` | Applies to any domain with dated content (events, awards, allocations). Option (a) is lowest effort. |

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
| `other` | Anything else — describe in Notes |

---

## Patterns / Themes

*Use this section for cross-entity patterns that don't fit cleanly into a single row.*

---

## Resolved Issues

*Move rows here (with resolution notes) once `verified`.*

