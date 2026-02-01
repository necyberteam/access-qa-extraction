"""Tests for affinity groups extractor.

These tests mock both the MCP client and the LLM client, so they
run instantly with no servers needed. The mocks return fake data
that matches the shape of real MCP responses.
"""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from access_qa_extraction.config import MCPServerConfig
from access_qa_extraction.extractors.affinity_groups import AffinityGroupsExtractor, strip_html

# --- Fake data that matches what the MCP server actually returns ---

FAKE_GROUPS = {
    "total": 2,
    "items": [
        {
            "id": 42,
            "name": "GPU Computing",
            "description": "<p>A community for GPU computing enthusiasts.</p>",
            "coordinator": "Jane Smith",
            "category": "Technology",
            "slack_link": "https://slack.example.com/gpu",
            "support_url": "https://support.example.com/gpu",
            "ask_ci_forum": "",
        },
        {
            "id": 99,
            "name": "Climate Modeling",
            "description": "Climate and weather modeling community.",
            "coordinator": "Bob Jones",
            "category": "Science",
            "slack_link": "",
            "support_url": "",
            "ask_ci_forum": "https://ask.ci/climate",
        },
    ],
}

FAKE_GROUP_DETAIL = {
    "group": {"id": 42, "name": "GPU Computing"},
    "events": {
        "total": 1,
        "items": [{"title": "GPU Workshop 2025", "date": "2025-03-15"}],
    },
    "knowledge_base": {
        "total": 1,
        "items": [{"title": "Getting Started with GPUs on ACCESS"}],
    },
}


# --- Fake LLM client ---


@dataclass
class FakeLLMResponse:
    """Mimics the response object from BaseLLMClient.generate()."""

    text: str


class FakeLLMClient:
    """Returns a canned JSON response that looks like what a real LLM would produce."""

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> FakeLLMResponse:
        cite = "<<SRC:affinity-groups:42>>"
        return FakeLLMResponse(
            text=json.dumps(
                [
                    {
                        "question": "What is the GPU Computing affinity group?",
                        "answer": f"A community for GPU computing.\n\n{cite}",
                    },
                    {
                        "question": "Who coordinates the GPU Computing group?",
                        "answer": f"Coordinated by Jane Smith.\n\n{cite}",
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
    """Config for the affinity-groups MCP server."""
    return MCPServerConfig(
        name="affinity-groups",
        url="http://localhost:3011",
        tools=["search_affinity_groups"],
    )


class TestAffinityGroupsExtractor:
    """Tests for AffinityGroupsExtractor."""

    async def test_basic_extraction(self, server_config):
        """Test that extraction produces Q&A pairs from mock data."""
        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeLLMClient())

        # Mock the MCP client so we don't need a real server
        mock_client = AsyncMock()

        # First call: list all groups. Subsequent calls: detail per group.
        mock_client.call_tool = AsyncMock(
            side_effect=[
                FAKE_GROUPS,  # search_affinity_groups({})
                FAKE_GROUP_DETAIL,  # search_affinity_groups({id: "42", include: "all"})
                {},  # search_affinity_groups({id: "99", include: "all"})
            ]
        )

        extractor.client = mock_client
        output = await extractor.extract()

        # FakeLLMClient returns 2 pairs per group, 2 groups = 4 pairs
        assert len(output.pairs) == 4
        assert all(p.domain == "affinity-groups" for p in output.pairs)

    async def test_deduplication(self, server_config):
        """Test that duplicate group IDs are skipped."""
        # Same group appears twice in the response
        duplicate_groups = {
            "total": 2,
            "items": [
                {"id": 42, "name": "GPU Computing", "description": "Desc", "coordinator": "J"},
                {"id": 42, "name": "GPU Computing", "description": "Desc", "coordinator": "J"},
            ],
        }

        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeLLMClient())
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            side_effect=[
                duplicate_groups,
                FAKE_GROUP_DETAIL,
            ]
        )
        extractor.client = mock_client
        output = await extractor.extract()

        # Should only process one group, not two
        assert len(output.pairs) == 2  # 2 pairs from the single group

    async def test_skips_empty_names(self, server_config):
        """Test that groups without names are skipped."""
        no_name_groups = {
            "total": 1,
            "items": [{"id": 1, "name": "", "description": "No name"}],
        }

        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeLLMClient())
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=no_name_groups)
        extractor.client = mock_client
        output = await extractor.extract()

        assert len(output.pairs) == 0

    async def test_raw_data_shape(self, server_config):
        """Test that raw_data has the expected keys for ComparisonGenerator."""
        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeLLMClient())
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            side_effect=[
                FAKE_GROUPS,
                FAKE_GROUP_DETAIL,
                {},
            ]
        )
        extractor.client = mock_client
        output = await extractor.extract()

        assert "42" in output.raw_data
        entry = output.raw_data["42"]
        assert entry["name"] == "GPU Computing"
        assert entry["group_id"] == "42"
        assert entry["category"] == "Technology"
        assert entry["coordinator"] == "Jane Smith"
        assert entry["has_events"] is True
        assert entry["has_knowledge_base"] is True

    async def test_llm_error_handling(self, server_config):
        """Test that LLM errors don't crash the whole extraction."""
        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeErrorLLMClient())
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            side_effect=[
                FAKE_GROUPS,
                FAKE_GROUP_DETAIL,
                {},
            ]
        )
        extractor.client = mock_client
        output = await extractor.extract()

        # Should return 0 pairs but not crash
        assert len(output.pairs) == 0
        # raw_data should still be populated
        assert len(output.raw_data) == 2

    async def test_qa_pair_ids_and_citations(self, server_config):
        """Test that Q&A pairs have proper IDs and citations."""
        extractor = AffinityGroupsExtractor(server_config, llm_client=FakeLLMClient())
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            side_effect=[
                FAKE_GROUPS,
                FAKE_GROUP_DETAIL,
                {},
            ]
        )
        extractor.client = mock_client
        output = await extractor.extract()

        for pair in output.pairs:
            assert pair.id.startswith("ag_")
            assert pair.source_ref.startswith("mcp://affinity-groups/groups/")
            assert pair.metadata.has_citation is True


class TestStripHtml:
    """Test the strip_html helper."""

    def test_strips_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_handles_empty(self):
        assert strip_html("") == ""
        assert strip_html(None) is None

    def test_normalizes_whitespace(self):
        assert strip_html("<p>Hello</p>  <p>World</p>") == "Hello World"
