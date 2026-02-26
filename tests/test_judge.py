"""Tests for LLM judge evaluation (generators/judge.py).

Tests evaluate_pairs(), confidence computation, suggested decisions,
graceful error handling, and the new QAMetadata fields.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

from access_qa_extraction.generators.judge import CONFIDENCE_THRESHOLD, evaluate_pairs
from access_qa_extraction.models import QAMetadata, QAPair


@dataclass
class FakeLLMResponse:
    text: str
    model: str = "fake-judge"
    usage: dict | None = None


def _make_mock_judge(response_json: str) -> MagicMock:
    """Create a mock LLM client that returns the given JSON."""
    mock = MagicMock()
    mock.generate.return_value = FakeLLMResponse(text=response_json)
    return mock


def _make_pair(pair_id: str, question: str = "Q?", answer: str = "A.") -> QAPair:
    """Create a minimal QAPair for testing."""
    return QAPair.create(
        id=pair_id,
        question=question,
        answer=answer,
        source_ref="mcp://test/entity/1",
        domain="test",
    )


SOURCE_DATA = {"name": "Test Entity", "description": "A test."}


class TestEvaluatePairsScoring:
    """Test that evaluate_pairs applies scores correctly."""

    def test_scores_all_pairs(self):
        """Judge returns scores for 3 pairs -> all 3 get metadata populated."""
        pairs = [_make_pair("p1"), _make_pair("p2"), _make_pair("p3")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.95,
                "relevance": 0.9,
                "completeness": 0.85,
                "issues": [],
            },
            {
                "pair_id": "p2",
                "faithfulness": 1.0,
                "relevance": 1.0,
                "completeness": 1.0,
                "issues": [],
            },
            {
                "pair_id": "p3",
                "faithfulness": 0.7,
                "relevance": 0.8,
                "completeness": 0.6,
                "issues": ["Missing details"],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        result = evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert result is pairs  # returns same list
        assert pairs[0].metadata.faithfulness_score == 0.95
        assert pairs[1].metadata.faithfulness_score == 1.0
        assert pairs[2].metadata.faithfulness_score == 0.7
        mock.generate.assert_called_once()

    def test_confidence_is_min(self):
        """confidence = min(faithfulness, relevance, completeness)."""
        pairs = [_make_pair("p1")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.9,
                "relevance": 0.7,
                "completeness": 0.85,
                "issues": [],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert pairs[0].metadata.confidence_score == 0.7  # min of the three

    def test_approved_threshold(self):
        """confidence >= 0.8 -> suggested_decision = 'approved'."""
        pairs = [_make_pair("p1")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.9,
                "relevance": 0.85,
                "completeness": 0.8,
                "issues": [],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert pairs[0].metadata.confidence_score == 0.8
        assert pairs[0].metadata.suggested_decision == "approved"

    def test_needs_review_threshold(self):
        """confidence < 0.8 -> suggested_decision = 'needs_review'."""
        pairs = [_make_pair("p1")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.75,
                "relevance": 0.9,
                "completeness": 0.9,
                "issues": [],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert pairs[0].metadata.confidence_score == 0.75
        assert pairs[0].metadata.suggested_decision == "needs_review"

    def test_eval_issues_populated(self):
        """Judge returns issues -> stored in metadata."""
        pairs = [_make_pair("p1")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.6,
                "relevance": 0.9,
                "completeness": 0.8,
                "issues": ["Answer mentions GPU count not in source", "Missing citation"],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert pairs[0].metadata.eval_issues == [
            "Answer mentions GPU count not in source",
            "Missing citation",
        ]


class TestEvaluatePairsEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pairs_noop(self):
        """Empty list -> no LLM call."""
        mock = _make_mock_judge("[]")

        result = evaluate_pairs([], SOURCE_DATA, mock)

        assert result == []
        mock.generate.assert_not_called()

    def test_llm_error_graceful(self):
        """LLM throws -> pairs returned with None scores."""
        pairs = [_make_pair("p1")]
        mock = MagicMock()
        mock.generate.side_effect = RuntimeError("API error")

        result = evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert len(result) == 1
        assert result[0].metadata.faithfulness_score is None
        assert result[0].metadata.suggested_decision is None

    def test_malformed_json_graceful(self):
        """LLM returns garbage -> pairs returned with None scores."""
        pairs = [_make_pair("p1")]
        mock = _make_mock_judge("not valid json at all")

        result = evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert len(result) == 1
        assert result[0].metadata.faithfulness_score is None

    def test_partial_match(self):
        """Judge returns scores for 2 of 3 pairs -> 2 scored, 1 has None."""
        pairs = [_make_pair("p1"), _make_pair("p2"), _make_pair("p3")]
        scores = [
            {
                "pair_id": "p1",
                "faithfulness": 0.9,
                "relevance": 0.9,
                "completeness": 0.9,
                "issues": [],
            },
            {
                "pair_id": "p3",
                "faithfulness": 0.8,
                "relevance": 0.8,
                "completeness": 0.8,
                "issues": [],
            },
        ]
        mock = _make_mock_judge(json.dumps(scores))

        evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert pairs[0].metadata.faithfulness_score == 0.9
        assert pairs[1].metadata.faithfulness_score is None  # p2 not scored
        assert pairs[2].metadata.faithfulness_score == 0.8

    def test_no_json_array_in_response(self):
        """LLM response without a JSON array -> pairs returned with None scores."""
        pairs = [_make_pair("p1")]
        mock = _make_mock_judge("I cannot evaluate these pairs because...")

        result = evaluate_pairs(pairs, SOURCE_DATA, mock)

        assert len(result) == 1
        assert result[0].metadata.faithfulness_score is None


class TestQAMetadataJudgeFields:
    """Test that the new judge fields work on the model."""

    def test_metadata_fields_on_model(self):
        """QAMetadata accepts all new judge fields."""
        meta = QAMetadata(
            faithfulness_score=0.95,
            relevance_score=0.9,
            completeness_score=0.85,
            confidence_score=0.85,
            eval_issues=["Minor: could mention dates"],
            suggested_decision="approved",
        )
        assert meta.faithfulness_score == 0.95
        assert meta.suggested_decision == "approved"
        assert meta.eval_issues == ["Minor: could mention dates"]

    def test_backward_compat(self):
        """QAPair without judge fields deserializes OK (None defaults)."""
        pair = _make_pair("compat_test")
        assert pair.metadata.faithfulness_score is None
        assert pair.metadata.relevance_score is None
        assert pair.metadata.completeness_score is None
        assert pair.metadata.confidence_score is None
        assert pair.metadata.eval_issues is None
        assert pair.metadata.suggested_decision is None

    def test_serialization_roundtrip(self):
        """Judge fields survive JSON serialization/deserialization."""
        pair = _make_pair("serial_test")
        pair.metadata.faithfulness_score = 0.9
        pair.metadata.relevance_score = 0.85
        pair.metadata.completeness_score = 0.8
        pair.metadata.confidence_score = 0.8
        pair.metadata.suggested_decision = "approved"
        pair.metadata.eval_issues = []

        json_str = pair.model_dump_json()
        restored = QAPair.model_validate_json(json_str)

        assert restored.metadata.faithfulness_score == 0.9
        assert restored.metadata.suggested_decision == "approved"
        assert restored.metadata.confidence_score == 0.8


class TestNoJudgeConfig:
    """Test the no_judge config flag."""

    def test_no_judge_flag(self):
        """no_judge=True on config -> ExtractionConfig.no_judge is True."""
        from access_qa_extraction.config import ExtractionConfig

        config = ExtractionConfig(no_judge=True)
        assert config.no_judge is True

    def test_no_judge_default_false(self):
        """Default ExtractionConfig has no_judge=False."""
        from access_qa_extraction.config import ExtractionConfig

        config = ExtractionConfig()
        assert config.no_judge is False

    def test_confidence_threshold_value(self):
        """Verify the threshold constant matches spec."""
        assert CONFIDENCE_THRESHOLD == 0.8
