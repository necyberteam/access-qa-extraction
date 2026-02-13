# Tier 4 Extraction Run Log — 2026-02-12

First end-to-end test of the "list-all + fixed question categories" pipeline.

## Setup

| Parameter | Value |
|-----------|-------|
| Date | 2026-02-12 (evening) |
| LLM Backend | OpenAI (`gpt-4o`) |
| Branch | `spike/research` @ `f3b1437` |
| MCP Docker | All 5 containers healthy |
| Python | 3.11.6, pytest 9.0.2 |

## Tiers 1-3 (prerequisites)

| Test | Result |
|------|--------|
| Tier 1: `pytest` | 67/67 passed (0.32s) |
| Tier 2: `report allocations` | 5,440 entities (direct API) |
| Tier 2: `report nsf-awards` | 100 sample (direct API) |
| Tier 3: `report compute-resources` | 23 entities (MCP) |
| Tier 3: `report software-discovery` | 1,404 entities (MCP) |
| Tier 3: `report affinity-groups` | 55 entities (MCP) |
| Tier 3: `list-servers` | All 5 registered |

## Tier 4: Full extraction with LLM

### compute-resources (1 entity, `--max-entities 1`)

**Entity selected:** Ranch (TACC archive system) — `ranch.tacc.access-ci.org`

4 Q&A pairs generated (categories: overview, organization, capabilities, access).
The LLM correctly skipped `gpu_hardware` (Ranch is storage, `hasGpu: false`)
and `cpu_hardware` (no compute nodes).

| ID | Category | Question | Answer (truncated) |
|----|----------|----------|--------------------|
| `compute-resources_ranch.tacc.access-ci.org_overview` | overview | What is the Ranch resource and what is it designed for? | Ranch is TACC's archive system designed for high-capacity, long-term storage... |
| `compute-resources_ranch.tacc.access-ci.org_organization` | organization | What organization operates the Ranch resource? | Texas Advanced Computing Center (TACC). |
| `compute-resources_ranch.tacc.access-ci.org_capabilities` | capabilities | What features and capabilities does Ranch support? | High-performance storage, interactive computing, ACCESS allocated resource... |
| `compute-resources_ranch.tacc.access-ci.org_access` | access | How do I get access to the Ranch resource? | Requires allocation on TACC systems (Frontera, Stampede3, Lonestar6). Default 2TB... |

**Observations:**
- Conditional categories working correctly (gpu/cpu skipped for storage resource)
- Citations present on all answers: `<<SRC:compute-resources:ranch.tacc.access-ci.org>>`
- Answers are concise and factual — no hallucination visible
- `source_data` includes both the resource metadata and hardware details

### allocations (2 entities, `--max-entities 2`)

**Entities selected:** Projects 72029 and 72028 (most recent on page 1)

10 Q&A pairs generated (5 categories x 2 entities).

#### Project 72029: "A transient survey in the NIR with NEWFIRM"

| ID | Category | Question | Answer (truncated) |
|----|----------|----------|--------------------|
| `allocations_72029_overview` | overview | What is this project about? | Uses NEWFIRM instrument for NIR transient survey, complementing Vera C. Rubin Observatory... |
| `allocations_72029_people` | people | Who is the PI and what institution? | Shreya Anand from Stanford University. |
| `allocations_72029_resources` | resources | What compute resources are allocated? | 200,000 ACCESS Credits. |
| `allocations_72029_field_of_science` | field_of_science | What field of science? | Astronomy and Planetary Sciences. |
| `allocations_72029_timeline` | timeline | When does this allocation start and end? | Feb 12, 2026 – Feb 11, 2027. |

#### Project 72028: "Resource-Efficient Cyclic Prompt Aggregation for Class Incremental Learning"

| ID | Category | Question | Answer (truncated) |
|----|----------|----------|--------------------|
| `allocations_72028_overview` | overview | What is this project about? | Resource-efficient CIL framework for edge devices with memory/latency constraints... |
| `allocations_72028_people` | people | Who is the PI? | Fariha Ishrat Rahman from UT Dallas. |
| `allocations_72028_resources` | resources | What compute resources? | 750,000 ACCESS Credits. |
| `allocations_72028_field_of_science` | field_of_science | What field of science? | Computer Science. |
| `allocations_72028_timeline` | timeline | When does it start and end? | Feb 12, 2026 – Feb 11, 2027. |

**Observations:**
- Paginator reported "272 pages, 20 on page 1" then stopped at 2 entities (fast!)
- All 5 fixed categories produced for each entity — no skipping
- Abstracts are rich and the LLM summarized them well
- `source_data` includes full project metadata

### nsf-awards (2 entities, `--max-entities 2`)

**Entities selected:** Awards 2529183 and 2449122

10 Q&A pairs generated (5 categories x 2 entities).

#### Award 2529183: "AI-powered Non-lethal Sampling for Forest Management"

| ID | Category | Question | Answer (truncated) |
|----|----------|----------|--------------------|
| `nsf-awards_2529183_overview` | overview | What is this award about? | Non-lethal monitoring of pollinators in forests using acoustics and AI cameras... |
| `nsf-awards_2529183_people` | people | Who are the PI and co-PIs? | PI: Christine Fortuin (Mississippi State). Co-PIs: Khan (KSU), Figueroa (Cornell), Larsen-Gray (NCASI). |
| `nsf-awards_2529183_funding` | funding | How much funding? | $797,762. |
| `nsf-awards_2529183_program` | program | What NSF program? | 01002526DB NSF RESEARCH & RELATED ACTIVIT. |
| `nsf-awards_2529183_timeline` | timeline | When does it start and end? | Sep 1, 2026 – Aug 31, 2030. |

#### Award 2449122: "Toxic Dust from the Great Salt Lake"

| ID | Category | Question | Answer (truncated) |
|----|----------|----------|--------------------|
| `nsf-awards_2449122_overview` | overview | What is this award about? | Impacts of toxic dust (arsenic, lead) from exposed GSL lakebed on agriculture... |
| `nsf-awards_2449122_people` | people | Who is the PI? | Derek Mallia at University of Utah. |
| `nsf-awards_2449122_funding` | funding | How much funding? | $154,191. |
| `nsf-awards_2449122_program` | program | What NSF program? | 01002526DB NSF RESEARCH & RELATED ACTIVIT. |
| `nsf-awards_2449122_timeline` | timeline | When does it start and end? | Sep 1, 2026 – Aug 31, 2029. |

**Observations:**
- `coPDPI` fix confirmed working — co-PIs parsed correctly from list format
- `primaryProgram` came back as a **list** from the NSF API (`["01002526DB ..."]`), not a string. The LLM handled it, but `_transform_nsf_award` should probably join or take the first element. Same pattern as the `coPDPI` bug we already fixed.
- Co-PI entries include email addresses (e.g., `sameekhan@ksu.edu`) — may want to strip those before sending to LLM
- Award 2449122 had no co-PIs, and the LLM correctly didn't mention any

## Issues found

### Bug: `coPDPI` as list (FIXED)

The NSF API returns `coPDPI` as a list, not a semicolon-delimited string.
`_transform_nsf_award` was calling `.split(";")` on it.
Fixed in this session — now handles both list and string formats.

### Bug: `primaryProgram` also a list (FIXED)

Same pattern as `coPDPI`. The `_transform_nsf_award` function was passing
`raw.get("primaryProgram")` directly, which can be a list from the NSF API.
Fixed by joining with `"; "` — e.g., `["OAC", "CISE"]` becomes `"OAC; CISE"`.
Test added: `test_handles_primary_program_as_list`.

### Observation: co-PI email addresses in output

The NSF API returns co-PI entries like `"Samee U Khan sameekhan@ksu.edu"`.
These flow through to the LLM prompt and into the Q&A answer. No precedent
in the codebase for stripping or keeping emails — leaving as-is for now,
flagged as a future decision.

### Fixed: unused MCPClient for allocations/nsf-awards

`BaseExtractor.run()` and `run_report()` create an MCPClient context, but
allocations and nsf-awards use httpx directly. Fixed by overriding `run()`
and `run_report()` in both extractors to skip MCPClient creation, with
comments explaining the pattern.

## Full 5-domain proof-of-run

Single command: `qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups --max-entities 2`

`--max-entities 2` applies per-domain (2 entities from each of the 5 servers).

| Domain | Entities | Pairs | 100% citations |
|--------|----------|-------|----------------|
| compute-resources | 2 | 10 | yes |
| software-discovery | 2 | 10 | yes |
| allocations | 2 | 10 | yes |
| nsf-awards | 2 | 10 | yes |
| affinity-groups | 2 | 5 | yes |
| comparisons | — | 2 | yes |
| **Total** | **10** | **47** | **yes** |

Affinity-groups got 5 pairs from 2 entities (one entity had 3 applicable
categories, the other had 2 — conditional categories like events/knowledge_base
correctly skipped when no data). Comparison generator produced 2 pairs
(needs both compute-resources and software-discovery in the same run).

### Old vs new output format

Old output (Feb 10, spray-search) preserved in `data/output/old-spray-search-feb10/`.
New output (Feb 12, fixed categories) at `data/output/*.jsonl`.

| Aspect | Old (spray-search) | New (fixed categories) |
|--------|-------------------|----------------------|
| ID format | `cr_ranch_tacc_access-ci_org_what_is_ranch_` | `compute-resources_ranch.tacc.access-ci.org_overview` |
| JSONL format | Pretty-printed (multi-line) | Compact (one line per record) |
| Categories | Freeform (LLM decides) | Fixed menu (5-6 per domain) |
| Predictability | 2-8 questions per entity | Exactly 1 per applicable category |

## Summary

All 5 extractors produce correct output with fixed question categories.
The pipeline runs end-to-end in a single command across all domains.
47 Q&A pairs from 10 entities, all with citations, no hallucinations detected.
