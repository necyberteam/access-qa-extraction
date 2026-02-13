# Tier 4 Extraction Run Log — 2026-02-13

First end-to-end test of dual-granularity pipeline (comprehensive + factoid + comparison).

## Setup

| Parameter | Value |
|-----------|-------|
| Date | 2026-02-13 |
| LLM Backend | OpenAI (`gpt-4o`) |
| Branch | `spike/research` @ `fa2ff93` + uncommitted dual-granularity changes |
| MCP Docker | All 5 containers healthy |
| Python | 3.11.6, pytest 9.0.2 |

## What changed since Feb 12 run

| Change | Description |
|--------|-------------|
| Factoid generator | `generators/factoids.py` — 36 templates across 5 domains, zero LLM |
| Granularity field | `models.py` — QAMetadata.granularity: `comprehensive` / `factoid` / `comparison` |
| ComparisonGenerator | Extended from 2 → 5 domains (allocations by FOS/institution/resource, NSF by program/institution, affinity groups by category) |
| Test count | 69 → 114 (+45 new tests) |

## Tiers 1-3 (prerequisites)

| Test | Result |
|------|--------|
| Tier 1: `pytest` | 114/114 passed (0.24s) |
| Tier 2: `report allocations` | 5,440 entities (direct API) |
| Tier 2: `report nsf-awards` | 100 sample (direct API) |
| Tier 3: `report compute-resources` | 23 entities (MCP) |
| Tier 3: `report software-discovery` | 1,404 entities (MCP) |
| Tier 3: `report affinity-groups` | 55 entities (MCP) |
| Tier 3: `list-servers` | All 5 registered |

## Tier 4: Full extraction (`--max-entities 2`)

Command: `qa-extract extract compute-resources software-discovery allocations nsf-awards affinity-groups --max-entities 2`

### Summary table

| Domain | Entities | Comprehensive | Factoid | Comparison | Total | Citations |
|--------|----------|---------------|---------|------------|-------|-----------|
| compute-resources | 2 | 10 | 12 | — | 22 | 22/22 |
| software-discovery | 2 | 10 | 14 | — | 24 | 24/24 |
| allocations | 2 | 10 | 16 | — | 26 | 26/26 |
| nsf-awards | 2 | 10 | 16 | — | 26 | 26/26 |
| affinity-groups | 2 | 5 | 10 | — | 15 | 15/15 |
| comparisons | — | — | — | 2 | 2 | 2/2 |
| **Total** | **10** | **45** | **68** | **2** | **115** | **115/115** |

### Feb 12 vs Feb 13 comparison

| Metric | Feb 12 (categories only) | Feb 13 (dual granularity) |
|--------|--------------------------|---------------------------|
| Q&A pairs | 47 | 115 |
| Comprehensive | 45 | 45 |
| Factoid | 0 | 68 |
| Comparison | 2 | 2 |
| Citations | 47/47 (100%) | 115/115 (100%) |
| Test count | 69 | 114 |

Comprehensive count is the same (5 categories × 2 entities × 5 domains, minus conditional skips on affinity-groups). Factoid pairs are entirely new — 68 template-generated pairs at zero LLM cost. Comparison count is 2 because `--max-entities 2` gives very little data for cross-entity grouping.

### Factoid pair samples

#### compute-resources (ACES)

| ID | Question | Answer |
|----|----------|--------|
| `compute-resources_aces..._fq_resource_type` | What type of resource is ACES? | ACES is a gpu resource. |
| `compute-resources_aces..._fq_operator` | Who operates ACES? | ACES is operated by Texas A&M University. |
| `compute-resources_aces..._fq_has_gpu` | Does ACES have GPUs? | Yes, ACES has GPUs available. |

#### nsf-awards (2449122)

| ID | Question | Answer |
|----|----------|--------|
| `nsf-awards_2449122_fq_pi_name` | Who is the PI for the NSF award "...Toxic Dust..."? | The PI is Derek Mallia. |
| `nsf-awards_2449122_fq_institution` | What institution is the NSF award at? | University of Utah. |
| `nsf-awards_2449122_fq_amount` | How much funding was awarded? | $154,191. |

#### allocations (72029)

| ID | Question | Answer |
|----|----------|--------|
| `allocations_72029_fq_pi_name` | Who is the PI for A transient survey...? | Anand, Shreya. |
| `allocations_72029_fq_institution` | What institution leads...? | Stanford University. |
| `allocations_72029_fq_field` | What field of science...? | Astronomy and Planetary Sciences. |

### Comparison pair samples

| ID | Domain | Question | Answer |
|----|--------|----------|--------|
| `cmp_sw_abaqus` | software-discovery | Which ACCESS resources have abaqus installed? | abaqus is available on ACES and Expanse. |
| `cmp_ag_cat_access_rp` | affinity-groups | What ACCESS affinity groups are in the ACCESS RP category? | The ACCESS RP affinity groups include REPACSS and Voyager. |

The second comparison is a **new type** (affinity groups by category) — did not exist in the Feb 12 run.

## Observations

1. **Factoid templates working correctly** — all 5 domains produce factoid pairs, boolean templates fire with correct yes/no, missing fields gracefully skipped
2. **Comprehensive pairs unchanged** — same count, same quality as Feb 12
3. **100% citation coverage** — all 115 pairs have citations, including all factoid pairs
4. **Factoid `source_data` is `None`** — confirmed no JSONL bloat from template pairs
5. **Granularity metadata correct** — every pair tagged with the right level
6. **Comparison generation extended** — affinity-groups now contribute comparisons (was not possible before)
7. **Performance** — extraction completes quickly; factoid generation adds negligible time (zero LLM calls)

## Projected full-run scale

| Domain | Entities | Comprehensive | Factoid | Comparison | Total |
|--------|----------|---------------|---------|------------|-------|
| compute-resources | 23 | ~130 | ~160 | ~20 | ~310 |
| software-discovery | 1,404 | ~7,000 | ~9,800 | ~50 | ~16,850 |
| affinity-groups | 55 | ~250 | ~330 | ~10 | ~590 |
| allocations | 5,440 | ~27,200 | ~43,500 | ~50 | ~70,750 |
| nsf-awards | 10,000+ | ~50,000 | ~80,000 | ~30 | ~130,030 |
| **TOTAL** | | **~85K** | **~134K** | **~160** | **~219K** |

## Output location

- Current run: `data/output/*.jsonl`
- Previous (Feb 12, categories only): `data/output/categories-only-feb12/`
- Original (Feb 10, spray-search): `data/output/old-spray-search-feb10/`
