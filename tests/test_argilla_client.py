"""Tests for Argilla client."""

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from access_qa_extraction.models import QAPair


def _make_pair(id: str = "test_001", question: str = "What is Delta?") -> QAPair:
    return QAPair.create(
        id=id,
        question=question,
        answer="Delta is a compute resource.\n\n<<SRC:compute-resources:delta>>",
        source_ref="mcp://compute-resources/resources/delta",
        domain="compute:resource_specs",
        source_data={"name": "Delta", "type": "GPU"},
    )


def _create_mock_rg():
    """Create a mock argilla module with all needed classes."""
    mock_rg = MagicMock(spec=ModuleType)
    mock_rg.Argilla = MagicMock()
    mock_rg.Dataset = MagicMock()
    mock_rg.Settings = MagicMock()
    mock_rg.TextField = MagicMock()
    mock_rg.LabelQuestion = MagicMock()
    mock_rg.TextQuestion = MagicMock()
    mock_rg.RatingQuestion = MagicMock()
    mock_rg.TermsMetadataProperty = MagicMock()
    mock_rg.VectorField = MagicMock()
    mock_rg.Record = MagicMock()
    mock_rg.Query = MagicMock()
    mock_rg.Similar = MagicMock()
    return mock_rg


@pytest.fixture
def mock_argilla():
    """Mock the argilla module so tests don't need a running server."""
    mock_rg = _create_mock_rg()

    # Mock the Argilla client instance
    mock_client = MagicMock()
    mock_rg.Argilla.return_value = mock_client

    # Mock dataset lookup returning None (not found)
    mock_client.datasets.return_value = None

    # Mock dataset creation
    mock_dataset = MagicMock()
    mock_rg.Dataset.return_value = mock_dataset

    with patch(
        "access_qa_extraction.argilla_client._import_argilla",
        return_value=mock_rg,
    ):
        yield mock_rg, mock_client, mock_dataset


@pytest.fixture
def mock_embedding():
    """Mock the sentence-transformers embedding model."""
    with patch("access_qa_extraction.argilla_client._get_embedding_model") as mock_get:
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
        mock_get.return_value = mock_model
        yield mock_model


class TestArgillaClient:
    def test_connect(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, mock_client, _ = mock_argilla

        client = ArgillaClient(api_url="http://localhost:6900", api_key="test-key")
        result = client.connect()

        mock_rg.Argilla.assert_called_once_with(api_url="http://localhost:6900", api_key="test-key")
        assert result == mock_client

    def test_get_or_create_dataset_creates_new(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()
        client.get_or_create_dataset()

        mock_rg.Dataset.assert_called_once()
        mock_dataset.create.assert_called_once()

    def test_get_or_create_dataset_finds_existing(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, mock_client, _ = mock_argilla

        existing_dataset = MagicMock()
        mock_client.datasets.return_value = existing_dataset

        client = ArgillaClient()
        client.connect()
        dataset = client.get_or_create_dataset()

        assert dataset == existing_dataset
        mock_rg.Dataset.assert_not_called()

    def test_qa_pair_to_record(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, _ = mock_argilla

        client = ArgillaClient()
        client.connect()
        pair = _make_pair()
        client.qa_pair_to_record(pair)

        mock_rg.Record.assert_called_once()
        call_kwargs = mock_rg.Record.call_args[1]

        assert call_kwargs["fields"]["question"] == "What is Delta?"
        assert "Delta is a compute resource" in call_kwargs["fields"]["answer"]
        assert call_kwargs["metadata"]["domain"] == "compute:resource_specs"
        assert call_kwargs["metadata"]["source_type"] == "mcp_extraction"
        assert call_kwargs["id"] == "test_001"

        mock_embedding.encode.assert_called_once_with("What is Delta?")

    def test_push_pairs(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()

        pairs = [_make_pair("p1", "Q1"), _make_pair("p2", "Q2")]
        pushed, skipped = client.push_pairs(pairs, check_duplicates=False)

        assert pushed == 2
        assert skipped == 0
        mock_dataset.records.log.assert_called_once()

    def test_push_pairs_empty(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()

        pushed, skipped = client.push_pairs([], check_duplicates=False)

        assert pushed == 0
        assert skipped == 0
        mock_dataset.records.log.assert_not_called()

    def test_push_pairs_with_dedup_skips_duplicates(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        # Make vector search return a result (duplicate found)
        mock_dataset.records.return_value.to_list.return_value = [{"id": "existing"}]

        client = ArgillaClient()
        client.connect()

        pairs = [_make_pair("p1", "Q1")]
        pushed, skipped = client.push_pairs(pairs, check_duplicates=True)

        assert pushed == 0
        assert skipped == 1

    def test_generate_embedding(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        client = ArgillaClient()
        embedding = client.generate_embedding("test question")

        assert len(embedding) == 384
        mock_embedding.encode.assert_called_once_with("test question")
