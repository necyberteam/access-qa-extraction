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
| 2 | nsf-awards | nsf-awards:2543343 | `decontextualized-question` | Question uses deictic reference with no entity name: "How much funding was awarded for **this NSF project**?" The "this" is meaningless outside the extraction context. At RAG retrieval time: (1) semantic search gets weak signal — no project name/number in the question to match a user query; (2) if retrieved as context, "this" refers to nothing, potentially confusing the consuming model. Seen pervasively across the dataset — likely a systemic prompt issue, not a one-off. | (a) Prompt instruction: questions must name the entity (project title or award number), never use "this project / this resource / this group". (b) Post-process: detect and rewrite questions containing "this [noun]" patterns. (c) Accept as-is if RAG pipeline routes by metadata (source_ref) and semantic search on the question is not the primary retrieval signal — depends on architecture. | `open` | Pervasive — affects all domains. Severity depends on retrieval architecture. Confirm with Andrew whether question text is used for semantic search or if metadata routing dominates. |

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

### P1 — Questions must be self-contained

A Q&A pair should work as a standalone unit with no surrounding context. At retrieval time the question is matched against a user query; at inference time it is served as context to a consuming model. Both scenarios require that the question unambiguously identifies the entity it is about.

**Failing patterns:**
- Deictic references: "this project", "this resource", "this group" (see issue #2)
- Implicit temporal framing: "upcoming events", "recent awards" (see issue #1)
- Bare pronouns: "What does it offer?", "Who leads it?"

**Desired pattern:** "What events does the [Group Name] affinity group host?" — entity named, temporally neutral, self-contained.

This is a prompt-level fix: instruct the LLM that every question must include the entity's name or identifier as if the reader has no prior context.

---

## Resolved Issues

*Move rows here (with resolution notes) once `verified`.*

