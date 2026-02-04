"""Tests for citation validation."""

import pytest

from access_qa_extraction.citation_validator import (
    CITATION_PATTERN,
    AnswerValidationResult,
    Citation,
    CitationValidator,
    ValidationResult,
)
from access_qa_extraction.config import Config


class TestCitationParsing:
    """Tests for citation parsing."""

    def test_parse_single_citation(self):
        """Test parsing a single citation."""
        citation = Citation.parse("<<SRC:compute-resources:delta.ncsa.access-ci.org>>")
        assert citation is not None
        assert citation.domain == "compute-resources"
        assert citation.entity_id == "delta.ncsa.access-ci.org"

    def test_parse_software_citation(self):
        """Test parsing a software discovery citation."""
        citation = Citation.parse("<<SRC:software-discovery:pytorch>>")
        assert citation is not None
        assert citation.domain == "software-discovery"
        assert citation.entity_id == "pytorch"

    def test_parse_invalid_citation(self):
        """Test that invalid citations return None."""
        assert Citation.parse("not a citation") is None
        assert Citation.parse("<<SRC:incomplete") is None
        assert Citation.parse("<<SRC:>>") is None

    def test_regex_pattern(self):
        """Test the citation regex pattern directly."""
        text = "Some answer text.\n\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"
        matches = CITATION_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0] == ("compute-resources", "delta.ncsa.access-ci.org")


class TestCitationValidator:
    """Tests for CitationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator with pre-loaded entities."""
        config = Config.from_env()
        validator = CitationValidator(config)
        # Manually add known entities instead of calling MCP servers
        validator.add_entities("compute-resources", {
            "delta.ncsa.access-ci.org",
            "bridges2.psc.access-ci.org",
            "expanse.sdsc.access-ci.org",
        })
        validator.add_entities("software-discovery", {
            "pytorch",
            "tensorflow",
            "gromacs",
        })
        return validator

    def test_extract_single_citation(self, validator):
        """Test extracting a single citation from answer."""
        answer = "Delta has GPUs.\n\n<<SRC:compute-resources:delta.ncsa.access-ci.org>>"
        citations = validator.extract_citations(answer)
        assert len(citations) == 1
        assert citations[0].domain == "compute-resources"
        assert citations[0].entity_id == "delta.ncsa.access-ci.org"

    def test_extract_multiple_citations(self, validator):
        """Test extracting multiple citations from answer."""
        answer = """Both resources have GPUs.

<<SRC:compute-resources:delta.ncsa.access-ci.org>>
<<SRC:compute-resources:bridges2.psc.access-ci.org>>"""
        citations = validator.extract_citations(answer)
        assert len(citations) == 2

    def test_extract_cross_domain_citations(self, validator):
        """Test extracting citations from different domains."""
        answer = """PyTorch is available on Delta.

<<SRC:compute-resources:delta.ncsa.access-ci.org>>
<<SRC:software-discovery:pytorch>>"""
        citations = validator.extract_citations(answer)
        assert len(citations) == 2
        domains = {c.domain for c in citations}
        assert domains == {"compute-resources", "software-discovery"}

    def test_validate_valid_citation(self, validator):
        """Test validating a citation that exists."""
        citation = Citation(
            domain="compute-resources",
            entity_id="delta.ncsa.access-ci.org",
            raw="<<SRC:compute-resources:delta.ncsa.access-ci.org>>",
        )
        result = validator.validate_citation(citation)
        assert result.valid is True
        assert result.error is None

    def test_validate_invalid_citation(self, validator):
        """Test validating a citation that doesn't exist (hallucination)."""
        citation = Citation(
            domain="compute-resources",
            entity_id="stampede2.tacc.access-ci.org",  # Not in our test data
            raw="<<SRC:compute-resources:stampede2.tacc.access-ci.org>>",
        )
        result = validator.validate_citation(citation)
        assert result.valid is False
        assert "not found" in result.error

    def test_validate_unknown_domain(self, validator):
        """Test that unknown domains are assumed valid."""
        citation = Citation(
            domain="doc",
            entity_id="some-doc.pdf",
            raw="<<SRC:doc:some-doc.pdf>>",
        )
        result = validator.validate_citation(citation)
        # Unknown domains can't be validated, so assume valid
        assert result.valid is True

    def test_validate_answer_all_valid(self, validator):
        """Test validating an answer with all valid citations."""
        answer = """Delta and Bridges-2 have A100 GPUs.

<<SRC:compute-resources:delta.ncsa.access-ci.org>>
<<SRC:compute-resources:bridges2.psc.access-ci.org>>"""
        result = validator.validate_answer(answer)
        assert result.has_citations is True
        assert result.all_valid is True
        assert len(result.invalid_citations) == 0

    def test_validate_answer_with_hallucination(self, validator):
        """Test detecting hallucinated citations."""
        answer = """Stampede2 and Delta both have A100 GPUs.

<<SRC:compute-resources:stampede2.tacc.access-ci.org>>
<<SRC:compute-resources:delta.ncsa.access-ci.org>>"""
        result = validator.validate_answer(answer)
        assert result.has_citations is True
        assert result.all_valid is False
        assert len(result.invalid_citations) == 1
        assert result.invalid_citations[0].citation.entity_id == "stampede2.tacc.access-ci.org"

    def test_validate_answer_no_citations(self, validator):
        """Test validating an answer without citations."""
        answer = "I cannot answer that question."
        result = validator.validate_answer(answer)
        assert result.has_citations is False
        assert result.all_valid is False  # No citations means not all valid

    def test_validate_software_citation(self, validator):
        """Test validating software citations."""
        answer = """PyTorch is available on ACCESS.

<<SRC:software-discovery:pytorch>>"""
        result = validator.validate_answer(answer)
        assert result.all_valid is True

    def test_validate_invalid_software_citation(self, validator):
        """Test detecting hallucinated software citations."""
        answer = """SuperML is available on ACCESS.

<<SRC:software-discovery:superml>>"""
        result = validator.validate_answer(answer)
        assert result.all_valid is False
        assert len(result.invalid_citations) == 1


class TestAnswerValidationResult:
    """Tests for AnswerValidationResult properties."""

    def test_all_valid_with_valid_citations(self):
        """Test all_valid returns True when all citations valid."""
        result = AnswerValidationResult(
            has_citations=True,
            citations=[
                ValidationResult(
                    citation=Citation("domain", "id1", "raw1"),
                    valid=True,
                ),
                ValidationResult(
                    citation=Citation("domain", "id2", "raw2"),
                    valid=True,
                ),
            ],
        )
        assert result.all_valid is True

    def test_all_valid_with_invalid_citation(self):
        """Test all_valid returns False when any citation invalid."""
        result = AnswerValidationResult(
            has_citations=True,
            citations=[
                ValidationResult(
                    citation=Citation("domain", "id1", "raw1"),
                    valid=True,
                ),
                ValidationResult(
                    citation=Citation("domain", "id2", "raw2"),
                    valid=False,
                    error="Not found",
                ),
            ],
        )
        assert result.all_valid is False

    def test_all_valid_no_citations(self):
        """Test all_valid returns False when no citations."""
        result = AnswerValidationResult(has_citations=False)
        assert result.all_valid is False

    def test_invalid_citations_list(self):
        """Test invalid_citations property returns only invalid ones."""
        result = AnswerValidationResult(
            has_citations=True,
            citations=[
                ValidationResult(
                    citation=Citation("domain", "id1", "raw1"),
                    valid=True,
                ),
                ValidationResult(
                    citation=Citation("domain", "id2", "raw2"),
                    valid=False,
                    error="Not found",
                ),
                ValidationResult(
                    citation=Citation("domain", "id3", "raw3"),
                    valid=False,
                    error="Also not found",
                ),
            ],
        )
        invalid = result.invalid_citations
        assert len(invalid) == 2
        assert all(not v.valid for v in invalid)
