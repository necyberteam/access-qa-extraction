"""LLM judge evaluation for Q&A pair quality.

Scores each pair on faithfulness, relevance, and completeness against the
source data. Confidence = min(three scores). Pairs with confidence >= 0.8
get suggested_decision="approved", otherwise "needs_review".

All pairs for a single entity are batched into one LLM call for cost efficiency.
"""

import json
import re

from ..llm_client import BaseLLMClient
from ..models import QAPair

CONFIDENCE_THRESHOLD = 0.8

JUDGE_SYSTEM_PROMPT = """You are a quality evaluator for Q&A pairs about ACCESS-CI resources.

You will receive a batch of Q&A pairs and the source data they were generated from.
Score each pair on three dimensions (0.0 to 1.0):

- **faithfulness**: Does every claim in the answer match the source data?
  1.0 = perfectly faithful, 0.0 = fabricated.
- **relevance**: Does the answer address what was asked?
  1.0 = directly answers the question, 0.0 = off-topic.
- **completeness**: Does the answer cover the key facts from the source?
  1.0 = comprehensive, 0.0 = missing important information.

## Rules

1. Output a JSON array with one object per pair.
2. Each object has: "pair_id" (string), "faithfulness" (float),
   "relevance" (float), "completeness" (float), "issues" (list of strings).
3. The "issues" list should contain brief descriptions of any problems found.
   Empty list if no issues.
4. Be strict on faithfulness — any claim not directly supported
   by the source data should lower the score.
5. Be lenient on completeness — a focused answer that covers the main points is fine.

## Output format

```json
[
  {"pair_id": "...", "faithfulness": 0.95, "relevance": 0.9,
   "completeness": 0.85, "issues": []},
  {"pair_id": "...", "faithfulness": 1.0, "relevance": 1.0,
   "completeness": 0.8, "issues": ["Minor: could mention end date"]}
]
```"""

JUDGE_USER_PROMPT_TEMPLATE = """## Source data

{source_json}

## Q&A pairs to evaluate

{pairs_block}"""


def _build_pairs_block(pairs: list[QAPair]) -> str:
    """Format pairs for the judge prompt."""
    lines = []
    for pair in pairs:
        question = pair.messages[0].content if pair.messages else ""
        answer = pair.messages[1].content if len(pair.messages) > 1 else ""
        lines.append(f"### {pair.id}\n**Q:** {question}\n**A:** {answer}\n")
    return "\n".join(lines)


# GUIDED-TOUR.md § Step 3A — judge LLM call (scores faithfulness, relevance, completeness per entity)
def evaluate_pairs(
    pairs: list[QAPair],
    source_data: dict,
    llm_client: BaseLLMClient,
    max_tokens: int = 2048,
) -> list[QAPair]:
    """Score a batch of QAPairs from a single entity. Mutates metadata in-place.

    Args:
        pairs: Q&A pairs to evaluate (all from the same entity)
        source_data: Raw entity data the pairs were generated from
        llm_client: LLM client for judge evaluation
        max_tokens: Max tokens for the judge response

    Returns:
        The same pairs list (metadata mutated in-place with scores)
    """
    if not pairs:
        return pairs

    try:
        source_json = json.dumps(source_data, indent=2, default=str)
        pairs_block = _build_pairs_block(pairs)

        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            source_json=source_json,
            pairs_block=pairs_block,
        )

        response = llm_client.generate(
            system=JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max_tokens,
        )

        json_match = re.search(r"\[[\s\S]*\]", response.text)
        if not json_match:
            return pairs

        scores_list = json.loads(json_match.group())

        # Index scores by pair_id for matching
        scores_by_id = {s["pair_id"]: s for s in scores_list if "pair_id" in s}

        for pair in pairs:
            scores = scores_by_id.get(pair.id)
            if not scores:
                continue

            faithfulness = float(scores.get("faithfulness", 0))
            relevance = float(scores.get("relevance", 0))
            completeness = float(scores.get("completeness", 0))
            confidence = min(faithfulness, relevance, completeness)

            pair.metadata.faithfulness_score = faithfulness
            pair.metadata.relevance_score = relevance
            pair.metadata.completeness_score = completeness
            pair.metadata.confidence_score = confidence
            pair.metadata.eval_issues = scores.get("issues", [])
            pair.metadata.suggested_decision = (
                "approved" if confidence >= CONFIDENCE_THRESHOLD else "needs_review"
            )

    except Exception as e:
        print(f"Error in judge evaluation: {e}")

    return pairs
