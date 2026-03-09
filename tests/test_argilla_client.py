"""Tests for Argilla client."""

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from access_qa_extraction.models import QAPair


def _make_pair(
    id: str = "test_001",
    question: str = "What is Delta?",
    source_ref: str = "mcp://compute-resources/resources/delta",
) -> QAPair:
    return QAPair.create(
        id=id,
        question=question,
        answer="Delta is a compute resource.\n\n<<SRC:compute-resources:delta>>",
        source_ref=source_ref,
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
    mock_rg.FloatMetadataProperty = MagicMock()
    mock_rg.VectorField = MagicMock()
    mock_rg.Record = MagicMock()
    mock_rg.Query = MagicMock()
    return mock_rg


def _make_mock_record(
    id: str = "rec_1",
    source_ref: str = "mcp://compute-resources/resources/delta",
    submitted_responses: list[dict] | None = None,
):
    """Build a mock Argilla record with optional submitted responses.

    submitted_responses: list of dicts like {"question_name": "edited_question", "value": "new q"}
    """
    record = MagicMock()
    record.id = id
    record.metadata = {"domain": "compute:resource_specs", "source_ref": source_ref}
    record.fields = {"question": "What is Delta?", "answer": "Delta is a resource."}
    record.vectors = {"question_embedding": [0.1] * 384}

    responses = []
    if submitted_responses:
        for resp_data in submitted_responses:
            resp = MagicMock()
            resp.status = "submitted"
            resp.question_name = resp_data["question_name"]
            resp.value = resp_data.get("value", "")
            responses.append(resp)
    record.responses = responses
    return record


@pytest.fixture
def mock_argilla():
    """Mock the argilla module so tests don't need a running server."""
    mock_rg = _create_mock_rg()

    mock_client = MagicMock()
    mock_rg.Argilla.return_value = mock_client

    # Dataset lookup returning None (not found) triggers creation
    mock_client.datasets.return_value = None

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
        assert call_kwargs["metadata"]["granularity"] == "comprehensive"
        assert call_kwargs["metadata"]["source_ref"] == "mcp://compute-resources/resources/delta"
        assert call_kwargs["fields"]["eval_issues"] == ""
        assert call_kwargs["id"] == "test_001"

        mock_embedding.encode.assert_called_once_with("What is Delta?")

    def test_qa_pair_to_record_with_judge_scores(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, _ = mock_argilla

        client = ArgillaClient()
        client.connect()
        pair = _make_pair()
        pair.metadata.faithfulness_score = 0.95
        pair.metadata.relevance_score = 0.90
        pair.metadata.completeness_score = 0.85
        pair.metadata.confidence_score = 0.85
        pair.metadata.suggested_decision = "approved"
        pair.metadata.eval_issues = ["minor factual gap", "could cite more specifically"]
        client.qa_pair_to_record(pair)

        call_kwargs = mock_rg.Record.call_args[1]
        assert call_kwargs["metadata"]["faithfulness_score"] == 0.95
        assert call_kwargs["metadata"]["relevance_score"] == 0.90
        assert call_kwargs["metadata"]["completeness_score"] == 0.85
        assert call_kwargs["metadata"]["confidence_score"] == 0.85
        assert call_kwargs["metadata"]["suggested_decision"] == "approved"
        assert (
            call_kwargs["fields"]["eval_issues"]
            == "minor factual gap; could cite more specifically"
        )

    def test_generate_embedding(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        client = ArgillaClient()
        embedding = client.generate_embedding("test question")

        assert len(embedding) == 384
        mock_embedding.encode.assert_called_once_with("test question")


class TestGetOrCreateArchiveDataset:
    def test_creates_new_archive_dataset(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()
        result = client.get_or_create_archive_dataset()

        assert result == mock_dataset
        mock_dataset.create.assert_called_once()

    def test_finds_existing_archive_dataset(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, mock_client, _ = mock_argilla

        existing = MagicMock()
        mock_client.datasets.return_value = existing

        client = ArgillaClient()
        client.connect()
        result = client.get_or_create_archive_dataset()

        assert result == existing

    def test_caches_archive_dataset(self, mock_argilla):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()
        first = client.get_or_create_archive_dataset()
        second = client.get_or_create_archive_dataset()

        assert first is second
        mock_dataset.create.assert_called_once()


class TestComputeAnnotationDepth:
    def test_no_responses_returns_approved_only(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(submitted_responses=None)
        assert ArgillaClient._compute_annotation_depth(record) == "approved_only"

    def test_approved_only_no_edits(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(
            submitted_responses=[{"question_name": "review_decision", "value": "approved"}]
        )
        assert ArgillaClient._compute_annotation_depth(record) == "approved_only"

    def test_has_edits_edited_question(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(
            submitted_responses=[{"question_name": "edited_question", "value": "Better question?"}]
        )
        assert ArgillaClient._compute_annotation_depth(record) == "has_edits"

    def test_has_edits_edited_answer(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(
            submitted_responses=[{"question_name": "edited_answer", "value": "Better answer."}]
        )
        assert ArgillaClient._compute_annotation_depth(record) == "has_edits"

    def test_has_edits_rejection_notes(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(
            submitted_responses=[{"question_name": "rejection_notes", "value": "Too vague."}]
        )
        assert ArgillaClient._compute_annotation_depth(record) == "has_edits"

    def test_empty_edit_value_returns_approved_only(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = _make_mock_record(
            submitted_responses=[{"question_name": "edited_question", "value": ""}]
        )
        assert ArgillaClient._compute_annotation_depth(record) == "approved_only"

    def test_non_submitted_response_ignored(self):
        from access_qa_extraction.argilla_client import ArgillaClient

        record = MagicMock()
        resp = MagicMock()
        resp.status = "draft"
        resp.question_name = "edited_question"
        resp.value = "Some edit"
        record.responses = [resp]
        assert ArgillaClient._compute_annotation_depth(record) == "approved_only"


class TestArchiveAnnotatedRecords:
    def test_no_annotated_records_returns_zero(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()

        records = [_make_mock_record(submitted_responses=None)]
        count = client._archive_annotated_records(records, "mcp://test/ref")

        assert count == 0

    def test_archives_submitted_records(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()

        records = [
            _make_mock_record(
                id="annotated_1",
                submitted_responses=[{"question_name": "review_decision", "value": "approved"}],
            ),
            _make_mock_record(id="unannotated", submitted_responses=None),
        ]
        count = client._archive_annotated_records(records, "mcp://test/ref")

        assert count == 1
        # Archive dataset's records.log should have been called with 1 record
        archive_ds = client.get_or_create_archive_dataset()
        archive_ds.records.log.assert_called_once()
        logged_records = archive_ds.records.log.call_args[0][0]
        assert len(logged_records) == 1

    def test_archive_metadata_tags(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, _ = mock_argilla

        client = ArgillaClient()
        client.connect()

        records = [
            _make_mock_record(
                submitted_responses=[
                    {"question_name": "edited_answer", "value": "Corrected answer."}
                ],
            ),
        ]
        client._archive_annotated_records(records, "mcp://test/ref")

        # Check the Record constructor was called with correct metadata
        record_call = mock_rg.Record.call_args[1]
        assert record_call["metadata"]["replaced_reason"] == "source_data_changed"
        assert record_call["metadata"]["annotation_depth"] == "has_edits"
        assert "archived_at" in record_call["metadata"]


class TestDeleteRecordsBySourceRef:
    def test_no_records_found(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla
        # dataset.records() returns empty list
        mock_dataset.records.return_value = iter([])

        client = ArgillaClient()
        client.connect()
        deleted, archived = client.delete_records_by_source_ref("mcp://test/nothing")

        assert deleted == 0
        assert archived == 0

    def test_deletes_records_without_annotations(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        records = [
            _make_mock_record(id="r1", submitted_responses=None),
            _make_mock_record(id="r2", submitted_responses=None),
        ]
        mock_dataset.records.return_value = iter(records)

        client = ArgillaClient()
        client.connect()
        deleted, archived = client.delete_records_by_source_ref("mcp://test/ref")

        assert deleted == 2
        assert archived == 0
        mock_dataset.records.delete.assert_called_once_with(records=records)

    def test_archives_then_deletes_annotated_records(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        mock_rg, _, mock_dataset = mock_argilla

        annotated = _make_mock_record(
            id="annotated",
            submitted_responses=[{"question_name": "review_decision", "value": "approved"}],
        )
        unannotated = _make_mock_record(id="plain", submitted_responses=None)
        records = [annotated, unannotated]
        mock_dataset.records.return_value = iter(records)

        client = ArgillaClient()
        client.connect()
        deleted, archived = client.delete_records_by_source_ref("mcp://test/ref")

        assert deleted == 2
        assert archived == 1
        mock_dataset.records.delete.assert_called_once_with(records=records)


class TestPushPairs:
    def test_empty_list(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        client = ArgillaClient()
        client.connect()
        pushed, archived = client.push_pairs([])

        assert pushed == 0
        assert archived == 0
        mock_dataset.records.log.assert_not_called()

    def test_single_source_ref(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla
        # delete_records_by_source_ref will query and find nothing
        mock_dataset.records.return_value = iter([])

        client = ArgillaClient()
        client.connect()

        pairs = [_make_pair("p1", "Q1"), _make_pair("p2", "Q2")]
        pushed, archived = client.push_pairs(pairs)

        assert pushed == 2
        assert archived == 0
        mock_dataset.records.log.assert_called_once()

    def test_multiple_source_refs(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla
        mock_dataset.records.return_value = iter([])

        client = ArgillaClient()
        client.connect()

        pairs = [
            _make_pair("p1", "Q1", source_ref="mcp://compute/delta"),
            _make_pair("p2", "Q2", source_ref="mcp://compute/bridges"),
        ]
        pushed, archived = client.push_pairs(pairs)

        assert pushed == 2
        assert archived == 0
        # One log call per source_ref
        assert mock_dataset.records.log.call_count == 2

    def test_replaces_existing_records(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla

        old_records = [_make_mock_record(id="old_1", submitted_responses=None)]

        call_count = [0]

        def records_side_effect(*args, **kwargs):
            # First call is from delete_records_by_source_ref (query), returns old records
            # Subsequent calls are mock default
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(old_records)
            return iter([])

        mock_dataset.records.side_effect = records_side_effect
        mock_dataset.records.log = MagicMock()
        mock_dataset.records.delete = MagicMock()

        client = ArgillaClient()
        client.connect()

        pairs = [_make_pair("p1", "Q1")]
        pushed, archived = client.push_pairs(pairs)

        assert pushed == 1
        assert archived == 0
        mock_dataset.records.delete.assert_called_once()
        mock_dataset.records.log.assert_called_once()


class TestPushFromJsonl:
    def test_delegates_to_push_pairs(self, mock_argilla, mock_embedding):
        from access_qa_extraction.argilla_client import ArgillaClient

        _, _, mock_dataset = mock_argilla
        mock_dataset.records.return_value = iter([])

        client = ArgillaClient()
        client.connect()

        pairs = [_make_pair("p1", "Q1")]
        with patch("access_qa_extraction.argilla_client.ArgillaClient.push_pairs") as mock_push:
            mock_push.return_value = (1, 0)
            with patch("access_qa_extraction.output.jsonl_writer.load_jsonl", return_value=pairs):
                pushed, archived = client.push_from_jsonl("test.jsonl")

        assert pushed == 1
        assert archived == 0
        mock_push.assert_called_once_with(pairs)
