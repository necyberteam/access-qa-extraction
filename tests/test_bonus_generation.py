"""Tests for bonus (exploratory) Q&A generation.

Tests has_rich_text(), build_bonus_system_prompt(), generate_bonus_pairs(),
and the exploratory granularity on QAPair.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

from access_qa_extraction.models import QAPair
from access_qa_extraction.question_categories import (
    MIN_RICH_TEXT_LENGTH,
    build_bonus_system_prompt,
    build_freeform_system_prompt,
    generate_bonus_pairs,
    has_rich_text,
)

# --- has_rich_text tests ---


class TestHasRichText:
    """Test the has_rich_text() helper."""

    def test_long_description_returns_true(self):
        data = {"description": "A" * MIN_RICH_TEXT_LENGTH}
        assert has_rich_text("compute-resources", data) is True

    def test_short_description_returns_false(self):
        data = {"description": "Short."}
        assert has_rich_text("compute-resources", data) is False

    def test_empty_description_returns_false(self):
        data = {"description": ""}
        assert has_rich_text("compute-resources", data) is False

    def test_missing_field_returns_false(self):
        data = {"name": "Delta"}
        assert has_rich_text("compute-resources", data) is False

    def test_abstract_field_for_allocations(self):
        data = {"abstract": "X" * 150}
        assert has_rich_text("allocations", data) is True

    def test_abstract_field_for_nsf_awards(self):
        data = {"abstract": "Y" * 200}
        assert has_rich_text("nsf-awards", data) is True

    def test_whitespace_only_returns_false(self):
        data = {"description": "   " * 50}
        assert has_rich_text("compute-resources", data) is False

    def test_unknown_domain_returns_false(self):
        data = {"description": "A" * 200}
        assert has_rich_text("unknown-domain", data) is False

    def test_non_string_field_returns_false(self):
        data = {"description": 12345}
        assert has_rich_text("compute-resources", data) is False


# --- build_bonus_system_prompt tests ---


class TestBuildBonusSystemPrompt:
    """Test the bonus system prompt builder."""

    def test_contains_covered_categories(self):
        prompt = build_bonus_system_prompt("compute-resources")
        assert "overview" in prompt
        assert "gpu_hardware" in prompt
        assert "capabilities" in prompt

    def test_contains_domain_label(self):
        prompt = build_bonus_system_prompt("allocations")
        assert "allocation projects" in prompt

    def test_contains_entity_type(self):
        prompt = build_bonus_system_prompt("nsf-awards")
        assert "NSF award" in prompt

    def test_instructs_1_to_3_questions(self):
        prompt = build_bonus_system_prompt("compute-resources")
        assert "1-3" in prompt

    def test_instructs_empty_array(self):
        prompt = build_bonus_system_prompt("compute-resources")
        assert "[]" in prompt


# --- build_freeform_system_prompt tests ---


class TestBuildFreeformSystemPrompt:
    """Test the freeform system prompt builder."""

    def test_contains_category_descriptions_as_guidance(self):
        prompt = build_freeform_system_prompt("compute-resources")
        # Should have category descriptions but NOT category IDs (no fixed menu)
        assert "What is this resource and what is it designed for?" in prompt
        assert "What GPUs does this resource have?" in prompt
        assert "How do I get access to or start using this resource?" in prompt

    def test_does_not_require_exact_categories(self):
        prompt = build_freeform_system_prompt("compute-resources")
        # Should NOT instruct "exactly one Q&A pair for each category"
        assert "exactly one" not in prompt
        # Should NOT have category IDs as required fields
        assert '"category"' not in prompt

    def test_encourages_variable_count(self):
        prompt = build_freeform_system_prompt("compute-resources")
        assert "10-15" in prompt or "data-rich" in prompt
        assert "4-5" in prompt or "simple entity" in prompt

    def test_contains_domain_label(self):
        prompt = build_freeform_system_prompt("allocations")
        assert "allocation projects" in prompt

    def test_contains_entity_type(self):
        prompt = build_freeform_system_prompt("nsf-awards")
        assert "NSF award" in prompt

    def test_all_domains_build_without_error(self):
        for domain in ["compute-resources", "software-discovery", "allocations",
                        "nsf-awards", "affinity-groups"]:
            prompt = build_freeform_system_prompt(domain)
            assert len(prompt) > 100

    def test_conditional_categories_marked(self):
        prompt = build_freeform_system_prompt("compute-resources")
        assert "only if data is present" in prompt

    def test_output_format_has_question_answer(self):
        prompt = build_freeform_system_prompt("compute-resources")
        assert '"question"' in prompt
        assert '"answer"' in prompt


# --- generate_bonus_pairs tests ---


@dataclass
class FakeLLMResponse:
    text: str
    model: str = "fake"
    usage: dict | None = None


def _make_mock_llm(response_json: str) -> MagicMock:
    """Create a mock LLM client that returns the given JSON."""
    mock = MagicMock()
    mock.generate.return_value = FakeLLMResponse(text=response_json)
    return mock


class TestGenerateBonusPairs:
    """Test the generate_bonus_pairs() function."""

    def test_skips_short_description(self):
        """No bonus pairs when entity has no rich text."""
        mock_llm = _make_mock_llm("[]")
        pairs = generate_bonus_pairs(
            "compute-resources", "delta", {"description": "Short"},
            mock_llm, max_tokens=1024,
        )
        assert pairs == []
        mock_llm.generate.assert_not_called()

    def test_generates_pairs(self):
        """LLM returns 2 bonus Qs → 2 pairs with granularity=exploratory."""
        response = (
            '[{"question": "What cooling system does Delta use?",'
            ' "answer": "Delta uses liquid cooling.\\n\\n<<SRC:compute-resources:delta>>"},'
            ' {"question": "What interconnect does Delta have?",'
            ' "answer": "Delta uses Slingshot 11.\\n\\n<<SRC:compute-resources:delta>>"}]'
        )
        mock_llm = _make_mock_llm(response)
        data = {"description": "A" * 150, "name": "Delta"}
        pairs = generate_bonus_pairs(
            "compute-resources", "delta", data,
            mock_llm, max_tokens=1024,
        )
        assert len(pairs) == 2
        assert all(p.metadata.granularity == "exploratory" for p in pairs)
        mock_llm.generate.assert_called_once()

    def test_ids_sequential(self):
        """IDs follow {domain}_{id}_bonus_1, _bonus_2 pattern."""
        response = """[
            {"question": "Q1?", "answer": "A1\\n\\n<<SRC:allocations:proj1>>"},
            {"question": "Q2?", "answer": "A2\\n\\n<<SRC:allocations:proj1>>"},
            {"question": "Q3?", "answer": "A3\\n\\n<<SRC:allocations:proj1>>"}
        ]"""
        mock_llm = _make_mock_llm(response)
        data = {"abstract": "B" * 200}
        pairs = generate_bonus_pairs(
            "allocations", "proj1", data,
            mock_llm, max_tokens=1024,
        )
        assert [p.id for p in pairs] == [
            "allocations_proj1_bonus_1",
            "allocations_proj1_bonus_2",
            "allocations_proj1_bonus_3",
        ]

    def test_empty_array_ok(self):
        """LLM returns [] → 0 pairs, no error."""
        mock_llm = _make_mock_llm("[]")
        data = {"abstract": "C" * 200}
        pairs = generate_bonus_pairs(
            "nsf-awards", "1234567", data,
            mock_llm, max_tokens=1024,
        )
        assert pairs == []

    def test_caps_at_3(self):
        """Even if LLM returns 5 pairs, we only keep 3."""
        items = [
            {"question": f"Q{i}?", "answer": f"A{i}\\n\\n<<SRC:x:y>>"}
            for i in range(5)
        ]
        import json
        response = json.dumps(items)
        mock_llm = _make_mock_llm(response)
        data = {"description": "D" * 200}
        pairs = generate_bonus_pairs(
            "affinity-groups", "group1", data,
            mock_llm, max_tokens=1024,
        )
        assert len(pairs) == 3

    def test_citations_present(self):
        """All bonus answers have <<SRC:...>> citations."""
        response = """[
            {"question": "Q?", "answer": "Answer text.\\n\\n<<SRC:software-discovery:python>>"}
        ]"""
        mock_llm = _make_mock_llm(response)
        data = {"description": "E" * 200}
        pairs = generate_bonus_pairs(
            "software-discovery", "python", data,
            mock_llm, max_tokens=1024,
        )
        assert len(pairs) == 1
        assert pairs[0].metadata.has_citation is True

    def test_source_ref_correct_per_domain(self):
        """Source ref uses the correct MCP URI pattern for each domain."""
        response = '[{"question": "Q?", "answer": "A\\n\\n<<SRC:nsf-awards:123>>"}]'
        mock_llm = _make_mock_llm(response)
        data = {"abstract": "F" * 200}
        pairs = generate_bonus_pairs(
            "nsf-awards", "123", data,
            mock_llm, max_tokens=1024,
        )
        assert pairs[0].source_ref == "mcp://nsf-awards/awards/123"

    def test_llm_error_returns_empty(self):
        """LLM error → empty list, no exception."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("API error")
        data = {"description": "G" * 200}
        pairs = generate_bonus_pairs(
            "compute-resources", "delta", data,
            mock_llm, max_tokens=1024,
        )
        assert pairs == []

    def test_malformed_json_returns_empty(self):
        """Malformed LLM response → empty list."""
        mock_llm = _make_mock_llm("not valid json at all")
        data = {"description": "H" * 200}
        pairs = generate_bonus_pairs(
            "compute-resources", "delta", data,
            mock_llm, max_tokens=1024,
        )
        assert pairs == []

    def test_skips_items_without_question(self):
        """Items missing question field are skipped."""
        response = """[
            {"question": "", "answer": "A\\n\\n<<SRC:x:y>>"},
            {"question": "Valid Q?", "answer": "Valid A\\n\\n<<SRC:x:y>>"}
        ]"""
        mock_llm = _make_mock_llm(response)
        data = {"description": "I" * 200}
        pairs = generate_bonus_pairs(
            "compute-resources", "delta", data,
            mock_llm, max_tokens=1024,
        )
        assert len(pairs) == 1
        assert pairs[0].id == "compute-resources_delta_bonus_1"


# --- Exploratory granularity on QAPair ---


class TestExploratoryGranularity:
    """Test that exploratory granularity works on QAPair."""

    def test_create_with_exploratory(self):
        pair = QAPair.create(
            id="test_bonus",
            question="What is unique about this?",
            answer="Something unique.\n\n<<SRC:x:y>>",
            source_ref="mcp://x/y/z",
            domain="x",
            granularity="exploratory",
        )
        assert pair.metadata.granularity == "exploratory"

    def test_default_is_comprehensive(self):
        pair = QAPair.create(
            id="test",
            question="Q",
            answer="A",
            source_ref="mcp://x/y/z",
            domain="x",
        )
        assert pair.metadata.granularity == "comprehensive"
