# To Factoid or Not: Analysis of Factoid vs Freeform Q&A Pairs

**Date**: 2026-02-19
**Data source**: `data/output/all-domains-freeform/` (full 5-domain run, 2 entities per domain)

## Summary

| Metric | Value |
|---|---|
| Total factoid pairs | 68 |
| Overlap with freeform | 49 (72%) |
| Data quality issues | 10 (15%) |
| Unique + clean factoids | ~9 |

**72% of factoid questions are semantically covered by the freeform pass.** The remaining factoids are either data-quality-impaired or ask questions so narrow ("Does this affinity group have a knowledge base?" → "No") that their RAG retrieval value is questionable.

---

## The Case Against Factoids

### 1. Freeform covers the same ground — better

Every entity below shows the freeform pass naturally producing questions that overlap with factoid templates. The freeform versions are better because:
- They use natural names ("PNRP resource") instead of raw data fields ("PNRP - COMING SOON")
- They produce richer answers with context, not just bare facts
- They don't inherit upstream data quality issues

### 2. Data quality issues are factoid-only

All 10 data quality issues found in this run are in factoid pairs:
- **COMING_SOON_LEAK** (5 instances): PNRP entity name includes "COMING SOON" from upstream data. Factoid templates blindly interpolate it. Freeform LLM uses the proper name.
- **RAW_PROGRAM_CODE** (2 instances): NSF `primaryProgram` field contains raw codes like `"01002526DB NSF RESEARCH & RELATED ACTIVIT"`. Factoid templates pass it through. Freeform LLM describes the program naturally.
- **EMAIL_LEAK** (1 instance): Co-PI factoid includes raw email addresses from source data.
- **SINGULAR_PLURAL** (2 instances): "uses 1 resources" — template grammar bug.

The freeform LLM pass has **zero** of these issues because the LLM naturally cleans up data artifacts.

### 3. Maintenance burden is high

Factoid templates require per-domain:
- Field preparer functions (type coercion, null handling)
- Quality guard logic (`_has_quality_defect()`)
- Template strings with interpolation
- Post-format validation to catch broken interpolations

That's ~200 lines of template code per domain in `generators/factoids.py`, plus tests. Every new domain or data shape change requires template updates.

### 4. Factoid answers are thin

Compare answers for the same question:

**Factoid**: "Does PNRP - COMING SOON have GPUs?" → "Yes, PNRP - COMING SOON has GPUs available."
**Freeform**: "What GPUs are used in the PNRP resource?" → "The PNRP resource uses NVIDIA A100 GPUs along with FPGA accelerators..." (with specific details)

The factoid answer is technically correct but useless for a researcher who wants to know *which* GPUs.

---

## Per-Domain Evidence

### compute-resources (2 entities: PNRP, Ranch)

| Entity | Freeform | Factoid | Factoid overlap | Factoid issues |
|---|---|---|---|---|
| PNRP | 8 pairs | 6 pairs | 3 (50%) | 5 (COMING_SOON_LEAK) |
| Ranch | 8 pairs | 6 pairs | 4 (67%) | 0 |

PNRP factoids are particularly bad — every single one says "PNRP - COMING SOON" because the upstream `resource_descriptive_name` field contains that string. The freeform LLM correctly uses "PNRP" or "Prototype National Research Platform."

### software-discovery (2 entities: Abaqus, ABINIT)

| Entity | Freeform | Factoid | Factoid overlap | Factoid issues |
|---|---|---|---|---|
| Abaqus | 10 pairs | 7 pairs | 5 (71%) | 0 |
| ABINIT | 10 pairs | 7 pairs | 5 (71%) | 0 |

Cleanest domain for factoids (no data quality issues), but still 71% overlap. The unique factoids are version counts ("How many versions of abaqus are available?" → "4") — precise but low retrieval value.

### allocations (2 entities: 72096, 72097)

| Entity | Freeform | Factoid | Factoid overlap | Factoid issues |
|---|---|---|---|---|
| 72096 | 10 pairs | 8 pairs | 2 (25%) | 1 (SINGULAR_PLURAL) |
| 72097 | 13 pairs | 8 pairs | 8 (100%) | 1 (SINGULAR_PLURAL) |

Entity 72097 has 100% factoid overlap — every factoid question is already covered by freeform. The factoid versions repeat the full project title in every question and answer, making them awkwardly long.

### nsf-awards (2 entities: 2449122, 2529183)

| Entity | Freeform | Factoid | Factoid overlap | Factoid issues |
|---|---|---|---|---|
| 2449122 | 11 pairs | 8 pairs | 6 (75%) | 1 (RAW_PROGRAM_CODE) |
| 2529183 | 11 pairs | 8 pairs | 6 (75%) | 2 (RAW_PROGRAM_CODE, EMAIL_LEAK) |

NSF factoids repeat the full award title in every Q&A pair (sometimes 20+ words), making them unwieldy. The program code issue (`"01002526DB NSF RESEARCH & RELATED ACTIVIT"`) is a raw API artifact that the freeform LLM naturally avoids.

### affinity-groups (2 entities: REPACSS, Voyager)

| Entity | Freeform | Factoid | Factoid overlap | Factoid issues |
|---|---|---|---|---|
| REPACSS | 5 pairs | 5 pairs | 5 (100%) | 0 |
| Voyager | 5 pairs | 5 pairs | 5 (100%) | 0 |

**100% overlap for both entities.** Every factoid question is already covered by freeform. This domain has thin source data, so factoids and freeform are asking the same small set of questions.

---

## What Would We Lose?

The strongest argument for factoids was **determinism**: no LLM means zero hallucination risk. But:

1. The judge pass already catches hallucination (0.97 avg confidence on freeform pairs)
2. Factoid "determinism" doesn't prevent *bad* output — it just produces bad output deterministically (COMING_SOON_LEAK, RAW_PROGRAM_CODE, etc.)
3. The freeform LLM pass is grounded in the same source data and required to cite it

The only factoids that add genuinely unique information are:
- "Is X ACCESS-allocated?" (yes/no) — could be covered by freeform prompt guidance
- "What is the award number for X?" — very specific, probably covered if we prompt for it
- Version counts for software — niche but potentially useful

These could be addressed by enriching the freeform prompt rather than maintaining a separate template system.

---

## Recommendation

**Drop factoids.** The pipeline becomes 2-pass (freeform LLM → judge), eliminating:
- ~1000 lines of template code + tests in `generators/factoids.py`
- All 6 per-domain cleanup issues identified in the design doc
- The `granularity: factoid` distinction in output
- Per-domain field preparer maintenance

If specific factoid-style questions prove valuable for RAG retrieval (e.g., "What is the award number for X?"), we can add them as guidance in the freeform prompt rather than maintaining a parallel template system.

**Impact on output volume**: From 162 total pairs to ~94 (freeform + comparisons only). But 68 fewer low-quality duplicates is a feature, not a bug — the reviewer queue shrinks by 42% with no loss of unique information.
