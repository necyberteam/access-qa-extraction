"""Tests for NSF awards extractor.

These tests mock the direct API fetcher and the LLM client, so they
run instantly with no servers needed. The mocks return fake data
that matches the shape of real NSF API responses (MCP-normalized format).
"""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from access_qa_extraction.config import MCPServerConfig
from access_qa_extraction.extractors.nsf_awards import (
    NSFAwardsExtractor,
    _format_currency,
    _transform_nsf_award,
    strip_html,
)

# --- Fake data in MCP-normalized format (what _fetch_all_awards returns) ---

FAKE_AWARDS = [
    {
        "awardNumber": "2345678",
        "title": "Advanced Computing for Climate Research",
        "institution": "MIT",
        "principalInvestigator": "Dr. Jane Smith",
        "coPIs": ["Dr. Bob Jones", "Dr. Alice Chen"],
        "totalIntendedAward": "$1,500,000",
        "totalAwardedToDate": "$750,000",
        "startDate": "2024-01-01",
        "endDate": "2027-12-31",
        "abstract": "<p>This project advances climate modeling.</p>",
        "primaryProgram": "Advanced Cyberinfrastructure",
        "programOfficer": "Dr. Program Officer",
        "ueiNumber": "ABC123",
    },
    {
        "awardNumber": "9876543",
        "title": "HPC for Genomics",
        "institution": "Stanford",
        "principalInvestigator": "Dr. Mike Lee",
        "coPIs": [],
        "totalIntendedAward": "$500,000",
        "totalAwardedToDate": "$500,000",
        "startDate": "2023-06-01",
        "endDate": "2026-05-31",
        "abstract": "Genomics research using HPC.",
        "primaryProgram": "Biological Sciences",
        "programOfficer": "Dr. Another Officer",
        "ueiNumber": "XYZ789",
    },
]


# --- Fake LLM client ---


@dataclass
class FakeLLMResponse:
    """Mimics the response object from BaseLLMClient.generate()."""

    text: str


class FakeLLMClient:
    """Returns canned JSON response with category-based Q&A pairs."""

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> FakeLLMResponse:
        cite = "<<SRC:nsf-awards:2345678>>"
        return FakeLLMResponse(
            text=json.dumps(
                [
                    {
                        "category": "overview",
                        "question": "What is NSF award 2345678?",
                        "answer": (f"An award for climate research at MIT.\n\n{cite}"),
                    },
                    {
                        "category": "people",
                        "question": "Who is the PI on award 2345678?",
                        "answer": (f"Dr. Jane Smith from MIT.\n\n{cite}"),
                    },
                    {
                        "category": "funding",
                        "question": "How much funding does award 2345678 have?",
                        "answer": f"$1,500,000 total intended.\n\n{cite}",
                    },
                    {
                        "category": "program",
                        "question": "What program funds award 2345678?",
                        "answer": f"Advanced Cyberinfrastructure.\n\n{cite}",
                    },
                    {
                        "category": "timeline",
                        "question": "When does award 2345678 run?",
                        "answer": f"From 2024-01-01 to 2027-12-31.\n\n{cite}",
                    },
                ]
            )
        )


class FakeErrorLLMClient:
    """Simulates an LLM that throws an error."""

    def generate(self, **kwargs) -> FakeLLMResponse:
        raise RuntimeError("LLM is down")


# --- The actual tests ---


@pytest.fixture
def server_config():
    """Config for the nsf-awards MCP server."""
    return MCPServerConfig(
        name="nsf-awards",
        url="http://localhost:3007",
        tools=["search_nsf_awards"],
    )


class TestNSFAwardsExtractor:
    """Tests for NSFAwardsExtractor."""

    async def test_basic_extraction(self, server_config):
        """Test that extraction produces Q&A pairs from mock data."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())

        # Mock the direct API fetcher
        extractor._fetch_all_awards = AsyncMock(return_value=FAKE_AWARDS)

        output = await extractor.extract()

        # FakeLLMClient returns 5 pairs per award, 2 awards = 10 pairs
        assert len(output.pairs) == 10
        assert all(p.domain == "nsf-awards" for p in output.pairs)

    async def test_deduplication(self, server_config):
        """Test that duplicate award numbers are skipped."""
        duplicate_awards = [
            {
                "awardNumber": "2345678",
                "title": "Climate Research",
                "institution": "MIT",
                "principalInvestigator": "Dr. Smith",
            },
            {
                "awardNumber": "2345678",
                "title": "Climate Research",
                "institution": "MIT",
                "principalInvestigator": "Dr. Smith",
            },
        ]

        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_awards = AsyncMock(return_value=duplicate_awards)
        output = await extractor.extract()

        # Should only process one award
        assert len(output.pairs) == 5

    async def test_skips_empty_titles(self, server_config):
        """Test that awards without titles are skipped."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_awards = AsyncMock(
            return_value=[
                {
                    "awardNumber": "1111111",
                    "title": "",
                    "institution": "Nowhere",
                }
            ]
        )
        output = await extractor.extract()

        assert len(output.pairs) == 0

    async def test_raw_data_shape(self, server_config):
        """Test that raw_data has expected keys for ComparisonGenerator."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_awards = AsyncMock(return_value=FAKE_AWARDS)
        output = await extractor.extract()

        assert "2345678" in output.raw_data
        entry = output.raw_data["2345678"]
        assert entry["name"] == "Advanced Computing for Climate Research"
        assert entry["award_number"] == "2345678"
        assert entry["pi"] == "Dr. Jane Smith"
        assert entry["institution"] == "MIT"
        assert entry["total_award"] == "$1,500,000"
        assert entry["primary_program"] == "Advanced Cyberinfrastructure"
        assert entry["has_co_pis"] is True

        # Second award has no co-PIs
        entry2 = output.raw_data["9876543"]
        assert entry2["has_co_pis"] is False

    async def test_llm_error_handling(self, server_config):
        """Test that LLM errors don't crash the whole extraction."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeErrorLLMClient())
        extractor._fetch_all_awards = AsyncMock(return_value=FAKE_AWARDS)
        output = await extractor.extract()

        # Should return 0 pairs but not crash
        assert len(output.pairs) == 0
        # raw_data should still be populated
        assert len(output.raw_data) == 2

    async def test_qa_pair_ids_and_citations(self, server_config):
        """Test that Q&A pairs have proper IDs and citations."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_awards = AsyncMock(return_value=FAKE_AWARDS)
        output = await extractor.extract()

        for pair in output.pairs:
            assert pair.id.startswith("nsf-awards_")
            assert pair.source_ref.startswith("mcp://nsf-awards/awards/")
            assert pair.metadata.has_citation is True

    async def test_category_based_ids(self, server_config):
        """Test that IDs use category instead of question slug."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_awards = AsyncMock(return_value=FAKE_AWARDS[:1])
        output = await extractor.extract()

        ids = [p.id for p in output.pairs]
        assert "nsf-awards_2345678_overview" in ids
        assert "nsf-awards_2345678_people" in ids
        assert "nsf-awards_2345678_funding" in ids

    async def test_clean_award_data(self, server_config):
        """Test that award data is properly cleaned."""
        extractor = NSFAwardsExtractor(server_config, llm_client=FakeLLMClient())
        award = FAKE_AWARDS[0]
        cleaned = extractor._clean_award_data(award)

        assert cleaned["title"] == "Advanced Computing for Climate Research"
        assert cleaned["principal_investigator"] == "Dr. Jane Smith"
        assert cleaned["institution"] == "MIT"
        assert cleaned["total_intended_award"] == "$1,500,000"
        assert cleaned["primary_program"] == "Advanced Cyberinfrastructure"
        # HTML should be stripped from abstract
        assert "<p>" not in cleaned["abstract"]
        assert "This project advances climate modeling" in cleaned["abstract"]
        assert cleaned["co_pis"] == ["Dr. Bob Jones", "Dr. Alice Chen"]
        assert cleaned["startDate"] == "2024-01-01"
        assert cleaned["programOfficer"] == "Dr. Program Officer"


class TestTransformNSFAward:
    """Test the _transform_nsf_award helper."""

    def test_transforms_all_fields(self):
        raw = {
            "id": "1234567",
            "title": "Test Award",
            "piFirstName": "John",
            "piLastName": "Doe",
            "coPDPI": "Jane Smith; Bob Jones",
            "awardeeName": "MIT",
            "estimatedTotalAmt": "1500000",
            "fundsObligatedAmt": "750000",
            "startDate": "2024-01-01",
            "expDate": "2027-12-31",
            "abstractText": "Test abstract.",
            "primaryProgram": "OAC",
            "poName": "Dr. Officer",
            "ueiNumber": "ABC123",
        }
        result = _transform_nsf_award(raw)

        assert result["awardNumber"] == "1234567"
        assert result["principalInvestigator"] == "John Doe"
        assert result["coPIs"] == ["Jane Smith", "Bob Jones"]
        assert result["institution"] == "MIT"
        assert result["totalIntendedAward"] == "$1,500,000"
        assert result["totalAwardedToDate"] == "$750,000"
        assert result["endDate"] == "2027-12-31"
        assert result["abstract"] == "Test abstract."
        assert result["primaryProgram"] == "OAC"

    def test_handles_empty_co_pis(self):
        raw = {"id": "1", "title": "X", "coPDPI": ""}
        result = _transform_nsf_award(raw)
        assert result["coPIs"] == []

    def test_falls_back_to_fund_program_name(self):
        raw = {"id": "1", "title": "X", "fundProgramName": "Fallback Program"}
        result = _transform_nsf_award(raw)
        assert result["primaryProgram"] == "Fallback Program"


class TestFormatCurrency:
    """Test the _format_currency helper."""

    def test_formats_number(self):
        assert _format_currency("1500000") == "$1,500,000"

    def test_handles_empty(self):
        assert _format_currency("") == ""

    def test_handles_non_numeric(self):
        assert _format_currency("not-a-number") == "not-a-number"


class TestStripHtml:
    """Test the strip_html helper."""

    def test_strips_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_handles_empty(self):
        assert strip_html("") == ""
        assert strip_html(None) is None

    def test_normalizes_whitespace(self):
        assert strip_html("<p>Hello</p>  <p>World</p>") == "Hello World"
