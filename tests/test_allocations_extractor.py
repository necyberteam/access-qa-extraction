"""Tests for allocations extractor.

These tests mock the direct API fetcher and the LLM client, so they
run instantly with no servers needed. The mocks return fake data
that matches the shape of real API responses.
"""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from access_qa_extraction.config import MCPServerConfig
from access_qa_extraction.extractors.allocations import AllocationsExtractor, strip_html

# --- Fake data that matches what the allocations API returns ---

FAKE_PROJECTS = [
    {
        "projectId": "TG-CIS210014",
        "requestNumber": "TG-CIS210014",
        "requestTitle": "Machine Learning for Climate Prediction",
        "pi": "John Doe",
        "piInstitution": "MIT",
        "fos": "Computer Science",
        "abstract": "<p>This project uses ML to improve climate models.</p>",
        "allocationType": "Research",
        "beginDate": "2024-01-01",
        "endDate": "2025-12-31",
        "resources": [
            {
                "resourceName": "Delta GPU",
                "units": "GPU Hours",
                "allocation": 50000,
                "resourceId": "delta.ncsa.access-ci.org",
            },
            {
                "resourceName": "Expanse",
                "units": "SUs",
                "allocation": 100000,
                "resourceId": "expanse.sdsc.access-ci.org",
            },
        ],
    },
    {
        "projectId": "TG-BIO220001",
        "requestNumber": "TG-BIO220001",
        "requestTitle": "Protein Folding Simulations",
        "pi": "Jane Smith",
        "piInstitution": "Stanford",
        "fos": "Biophysics",
        "abstract": "Molecular dynamics simulations of protein folding.",
        "allocationType": "Research",
        "beginDate": "2024-06-01",
        "endDate": "2026-05-31",
        "resources": [
            {
                "resourceName": "Bridges-2",
                "units": "SUs",
                "allocation": 200000,
                "resourceId": "bridges2.psc.access-ci.org",
            },
        ],
    },
]


# --- Fake LLM client ---


@dataclass
class FakeLLMResponse:
    """Mimics the response object from BaseLLMClient.generate()."""

    text: str


class FakeLLMClient:
    """Returns a canned JSON response with category-based Q&A pairs."""

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> FakeLLMResponse:
        cite = "<<SRC:allocations:TG-CIS210014>>"
        return FakeLLMResponse(
            text=json.dumps(
                [
                    {
                        "category": "overview",
                        "question": "What is allocation project TG-CIS210014?",
                        "answer": f"A research project for ML climate prediction.\n\n{cite}",
                    },
                    {
                        "category": "people",
                        "question": "Who is the PI on TG-CIS210014?",
                        "answer": f"John Doe from MIT.\n\n{cite}",
                    },
                    {
                        "category": "resources",
                        "question": "What resources are allocated to TG-CIS210014?",
                        "answer": (
                            f"Delta GPU (50,000 GPU Hours) and Expanse (100,000 SUs).\n\n{cite}"
                        ),
                    },
                    {
                        "category": "field_of_science",
                        "question": "What field of science is TG-CIS210014 in?",
                        "answer": f"Computer Science.\n\n{cite}",
                    },
                    {
                        "category": "timeline",
                        "question": "When does TG-CIS210014 run?",
                        "answer": f"From 2024-01-01 to 2025-12-31.\n\n{cite}",
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
    """Config for the allocations MCP server."""
    return MCPServerConfig(
        name="allocations",
        url="http://localhost:3006",
        tools=["search_projects", "get_allocation_statistics"],
    )


class TestAllocationsExtractor:
    """Tests for AllocationsExtractor."""

    async def test_basic_extraction(self, server_config):
        """Test that extraction produces Q&A pairs from mock data."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())

        # Mock the direct API fetcher
        extractor._fetch_all_projects = AsyncMock(return_value=FAKE_PROJECTS)

        output = await extractor.extract()

        # 5 comprehensive (LLM) + 8 factoid (template) per project × 2 projects
        comprehensive = [p for p in output.pairs if p.metadata.granularity == "comprehensive"]
        factoid = [p for p in output.pairs if p.metadata.granularity == "factoid"]
        assert len(comprehensive) == 10
        assert len(factoid) == 16
        assert all(p.domain == "allocations" for p in output.pairs)

    async def test_skips_empty_titles(self, server_config):
        """Test that projects without titles are skipped."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_projects = AsyncMock(
            return_value=[{"projectId": "TG-XXX", "requestTitle": "", "pi": "Nobody"}]
        )
        output = await extractor.extract()

        assert len(output.pairs) == 0

    async def test_raw_data_shape(self, server_config):
        """Test that raw_data has the expected keys for ComparisonGenerator."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_projects = AsyncMock(return_value=FAKE_PROJECTS)
        output = await extractor.extract()

        assert "TG-CIS210014" in output.raw_data
        entry = output.raw_data["TG-CIS210014"]
        assert entry["name"] == "Machine Learning for Climate Prediction"
        assert entry["project_id"] == "TG-CIS210014"
        assert entry["pi"] == "John Doe"
        assert entry["institution"] == "MIT"
        assert entry["fos"] == "Computer Science"
        assert entry["allocation_type"] == "Research"
        assert entry["resource_count"] == 2

    async def test_llm_error_handling(self, server_config):
        """Test that LLM errors don't crash the whole extraction."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeErrorLLMClient())
        extractor._fetch_all_projects = AsyncMock(return_value=FAKE_PROJECTS)
        output = await extractor.extract()

        # LLM fails → 0 comprehensive pairs, but factoid pairs still generated
        comprehensive = [p for p in output.pairs if p.metadata.granularity == "comprehensive"]
        factoid = [p for p in output.pairs if p.metadata.granularity == "factoid"]
        assert len(comprehensive) == 0
        assert len(factoid) == 16  # 8 per project × 2 projects
        # raw_data should still be populated
        assert len(output.raw_data) == 2

    async def test_qa_pair_ids_and_citations(self, server_config):
        """Test that Q&A pairs have proper IDs and citations."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_projects = AsyncMock(return_value=FAKE_PROJECTS)
        output = await extractor.extract()

        for pair in output.pairs:
            assert pair.id.startswith("allocations_")
            assert pair.source_ref.startswith("mcp://allocations/projects/")
            assert pair.metadata.has_citation is True

    async def test_category_based_ids(self, server_config):
        """Test that IDs use category instead of question slug."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())
        extractor._fetch_all_projects = AsyncMock(return_value=FAKE_PROJECTS[:1])
        output = await extractor.extract()

        ids = [p.id for p in output.pairs]
        assert "allocations_TG-CIS210014_overview" in ids
        assert "allocations_TG-CIS210014_people" in ids
        assert "allocations_TG-CIS210014_resources" in ids

    async def test_clean_project_data(self, server_config):
        """Test that project data is properly cleaned."""
        extractor = AllocationsExtractor(server_config, llm_client=FakeLLMClient())
        project = FAKE_PROJECTS[0]
        cleaned = extractor._clean_project_data(project)

        assert cleaned["title"] == "Machine Learning for Climate Prediction"
        assert cleaned["pi"] == "John Doe"
        assert cleaned["institution"] == "MIT"
        # HTML should be stripped from abstract
        assert "<p>" not in cleaned["abstract"]
        assert "This project uses ML" in cleaned["abstract"]
        assert len(cleaned["resources"]) == 2
        assert cleaned["resources"][0]["name"] == "Delta GPU"


class TestStripHtml:
    """Test the strip_html helper."""

    def test_strips_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_handles_empty(self):
        assert strip_html("") == ""
        assert strip_html(None) is None

    def test_normalizes_whitespace(self):
        assert strip_html("<p>Hello</p>  <p>World</p>") == "Hello World"
