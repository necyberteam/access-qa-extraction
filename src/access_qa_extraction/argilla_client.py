"""Argilla client for pushing Q&A pairs to review.

Uses entity-replace semantics: when pushing pairs for an entity, all existing
Argilla records for that entity's source_ref are deleted and replaced with fresh
records. Human-annotated records are archived before deletion.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .models import QAPair

if TYPE_CHECKING:
    import argilla as rg

logger = logging.getLogger(__name__)

# Dataset schema constants
DATASET_NAME = "qa-review"
ARCHIVE_DATASET_NAME = "qa-review-archive-superseded"
WORKSPACE_NAME = "default"
VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2 output dimensions

# Response fields that indicate substantive human edits (not just rubber-stamp approval)
_EDIT_FIELDS = {"edited_question", "edited_answer", "rejection_notes"}


def _import_argilla():
    """Import argilla, raising a clear error if not installed."""
    try:
        import argilla as rg
    except ImportError:
        raise ImportError(
            "argilla is required for Argilla integration. Install with: pip install -e '.[dev]'"
        ) from None
    return rg


def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


class ArgillaClient:
    """Client for pushing Q&A pairs to Argilla for human review.

    Uses entity-replace: for each source_ref, deletes existing records
    (archiving any with human annotations) before pushing fresh ones.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        workspace: str = WORKSPACE_NAME,
    ):
        self.api_url = api_url or os.getenv("ARGILLA_URL", "http://localhost:6900")
        self.api_key = api_key or os.getenv("ARGILLA_API_KEY", "argilla.apikey")
        self.workspace = workspace
        self._rg: Any = None
        self._client: rg.Argilla | None = None
        self._dataset: rg.Dataset | None = None
        self._archive_dataset: rg.Dataset | None = None
        self._embedding_model = None

    @property
    def rg(self):
        """Lazy-import the argilla module."""
        if self._rg is None:
            self._rg = _import_argilla()
        return self._rg

    def connect(self) -> rg.Argilla:
        """Connect to Argilla server."""
        rg = self.rg
        self._client = rg.Argilla(api_url=self.api_url, api_key=self.api_key)
        logger.info("Connected to Argilla at %s", self.api_url)
        return self._client

    @property
    def client(self) -> rg.Argilla:
        if self._client is None:
            self.connect()
        return self._client

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = _get_embedding_model()
        return self._embedding_model

    # ── Schema helpers (shared between live + archive datasets) ──────────

    def _build_fields(self, rg) -> list:
        return [
            rg.TextField(name="question", title="Question", required=True),
            rg.TextField(
                name="answer",
                title="Answer",
                use_markdown=True,
                required=True,
            ),
            rg.TextField(
                name="source_data",
                title="Source Data",
                use_markdown=True,
                required=False,
            ),
            rg.TextField(
                name="eval_issues",
                title="Judge Issues",
                required=False,
            ),
        ]

    def _build_questions(self, rg) -> list:
        return [
            rg.LabelQuestion(
                name="review_decision",
                title="Review Decision",
                labels=["approved", "rejected", "needs_edit", "flagged"],
                required=True,
            ),
            rg.LabelQuestion(
                name="rejection_reason",
                title="Rejection Reason (if rejected)",
                labels=[
                    "duplicate",
                    "incorrect",
                    "vague",
                    "incomplete",
                    "citation_issue",
                    "other",
                ],
                required=False,
            ),
            rg.TextQuestion(
                name="rejection_notes",
                title="Rejection Notes",
                required=False,
            ),
            rg.TextQuestion(
                name="edited_question",
                title="Edited Question (if needs changes)",
                required=False,
            ),
            rg.TextQuestion(
                name="edited_answer",
                title="Edited Answer (if needs changes)",
                required=False,
            ),
            rg.RatingQuestion(
                name="quality_rating",
                title="Quality Rating",
                values=[1, 2, 3, 4, 5],
                required=False,
            ),
        ]

    def _build_base_metadata(self, rg) -> list:
        return [
            rg.TermsMetadataProperty(name="domain", title="Domain"),
            rg.TermsMetadataProperty(name="source_type", title="Source Type"),
            rg.TermsMetadataProperty(name="complexity", title="Complexity"),
            rg.TermsMetadataProperty(name="granularity", title="Granularity"),
            rg.TermsMetadataProperty(name="has_citation", title="Has Citation"),
            rg.FloatMetadataProperty(name="faithfulness_score", title="Faithfulness Score"),
            rg.FloatMetadataProperty(name="relevance_score", title="Relevance Score"),
            rg.FloatMetadataProperty(name="completeness_score", title="Completeness Score"),
            rg.FloatMetadataProperty(name="confidence_score", title="Confidence Score"),
            rg.TermsMetadataProperty(name="suggested_decision", title="Suggested Decision"),
            rg.TermsMetadataProperty(name="source_ref", title="Source Reference"),
        ]

    # ── Dataset management ───────────────────────────────────────────────

    def get_or_create_dataset(self) -> rg.Dataset:
        """Get existing qa-review dataset or create it."""
        if self._dataset is not None:
            return self._dataset

        rg = self.rg

        try:
            self._dataset = self.client.datasets(name=DATASET_NAME, workspace=self.workspace)
            if self._dataset is not None:
                logger.info("Found existing dataset '%s'", DATASET_NAME)
                return self._dataset
        except Exception:
            pass

        settings = rg.Settings(
            guidelines=(
                "Review Q&A pairs extracted from ACCESS-CI MCP servers. "
                "Check that questions are clear, answers are accurate "
                "and well-cited, and source data supports the answer."
            ),
            fields=self._build_fields(rg),
            questions=self._build_questions(rg),
            metadata=self._build_base_metadata(rg),
            vectors=[
                rg.VectorField(name="question_embedding", dimensions=VECTOR_DIMENSIONS),
            ],
        )

        self._dataset = rg.Dataset(
            name=DATASET_NAME,
            workspace=self.workspace,
            settings=settings,
        )
        self._dataset.create()
        logger.info("Created dataset '%s'", DATASET_NAME)
        return self._dataset

    def get_or_create_archive_dataset(self) -> rg.Dataset:
        """Get or create the archive dataset for superseded annotated records."""
        if self._archive_dataset is not None:
            return self._archive_dataset

        rg = self.rg

        try:
            self._archive_dataset = self.client.datasets(
                name=ARCHIVE_DATASET_NAME, workspace=self.workspace
            )
            if self._archive_dataset is not None:
                logger.info("Found existing archive dataset '%s'", ARCHIVE_DATASET_NAME)
                return self._archive_dataset
        except Exception:
            pass

        archive_metadata = self._build_base_metadata(rg) + [
            rg.TermsMetadataProperty(name="archived_at", title="Archived At"),
            rg.TermsMetadataProperty(name="replaced_reason", title="Replaced Reason"),
            rg.TermsMetadataProperty(name="annotation_depth", title="Annotation Depth"),
        ]

        settings = rg.Settings(
            guidelines=(
                "Archived Q&A pairs replaced by re-extraction. "
                "Not used for RAG. Preserved for reference during re-review."
            ),
            fields=self._build_fields(rg),
            questions=self._build_questions(rg),
            metadata=archive_metadata,
            vectors=[
                rg.VectorField(name="question_embedding", dimensions=VECTOR_DIMENSIONS),
            ],
        )

        self._archive_dataset = rg.Dataset(
            name=ARCHIVE_DATASET_NAME,
            workspace=self.workspace,
            settings=settings,
        )
        self._archive_dataset.create()
        logger.info("Created archive dataset '%s'", ARCHIVE_DATASET_NAME)
        return self._archive_dataset

    # ── Embedding ────────────────────────────────────────────────────────

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text string using all-MiniLM-L6-v2."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    # ── Annotation archiving ─────────────────────────────────────────────

    @staticmethod
    def _compute_annotation_depth(record) -> str:
        """Determine annotation depth from a record's submitted responses.

        Returns "has_edits" if edited_question, edited_answer, or rejection_notes
        has a non-empty submitted value. "approved_only" otherwise.
        """
        for response in record.responses:
            if getattr(response, "status", None) == "submitted":
                if response.question_name in _EDIT_FIELDS and response.value:
                    return "has_edits"
        return "approved_only"

    def _archive_annotated_records(self, records: list, source_ref: str) -> int:
        """Archive records that have human annotations before deletion.

        Returns count of records archived.
        """
        annotated = [
            r
            for r in records
            if any(getattr(resp, "status", None) == "submitted" for resp in r.responses)
        ]

        if not annotated:
            return 0

        archive_dataset = self.get_or_create_archive_dataset()
        archived_at = datetime.now(UTC).isoformat()
        rg = self.rg

        archive_records = []
        for record in annotated:
            annotation_depth = self._compute_annotation_depth(record)

            updated_metadata = dict(record.metadata) if record.metadata else {}
            updated_metadata["archived_at"] = archived_at
            updated_metadata["replaced_reason"] = "source_data_changed"
            updated_metadata["annotation_depth"] = annotation_depth

            archive_record = rg.Record(
                fields=dict(record.fields) if record.fields else {},
                metadata=updated_metadata,
                vectors=dict(record.vectors) if record.vectors else {},
            )
            archive_records.append(archive_record)

        archive_dataset.records.log(archive_records)
        logger.warning(
            "Archived %d annotated records for %s to %s",
            len(annotated),
            source_ref,
            ARCHIVE_DATASET_NAME,
        )
        return len(annotated)

    # ── Entity-replace ───────────────────────────────────────────────────

    def delete_records_by_source_ref(self, source_ref: str) -> tuple[int, int]:
        """Delete all live dataset records for a given source_ref.

        Before deleting, archives any records that have human annotations.

        Returns:
            Tuple of (deleted_count, archived_count).
        """
        rg = self.rg
        dataset = self.get_or_create_dataset()

        query = rg.Query(filter=[("metadata.source_ref", "==", source_ref)])
        records = list(dataset.records(query=query, with_responses=True))

        if not records:
            return 0, 0

        archived_count = self._archive_annotated_records(records, source_ref)
        dataset.records.delete(records=records)
        logger.info(
            "Deleted %d records for source_ref=%s (%d archived)",
            len(records),
            source_ref,
            archived_count,
        )
        return len(records), archived_count

    # ── Record conversion ────────────────────────────────────────────────

    def qa_pair_to_record(self, pair: QAPair) -> rg.Record:
        """Convert a QAPair to an Argilla Record."""
        rg = self.rg
        question = pair.messages[0].content if pair.messages else ""
        answer = pair.messages[1].content if len(pair.messages) > 1 else ""

        source_data_str = ""
        if pair.metadata.source_data:
            raw = json.dumps(pair.metadata.source_data, indent=2, default=str)
            source_data_str = f"```json\n{raw[:5000]}\n```"

        eval_issues_str = ""
        if pair.metadata.eval_issues:
            eval_issues_str = "; ".join(pair.metadata.eval_issues)

        question_embedding = self.generate_embedding(question)

        metadata = {
            "domain": pair.domain,
            "source_type": pair.source,
            "complexity": pair.metadata.complexity,
            "granularity": pair.metadata.granularity,
            "has_citation": str(pair.metadata.has_citation),
            "source_ref": pair.source_ref,
        }

        for score_field in (
            "faithfulness_score",
            "relevance_score",
            "completeness_score",
            "confidence_score",
        ):
            value = getattr(pair.metadata, score_field, None)
            if value is not None:
                metadata[score_field] = value

        if pair.metadata.suggested_decision is not None:
            metadata["suggested_decision"] = pair.metadata.suggested_decision

        return rg.Record(
            fields={
                "question": question,
                "answer": answer,
                "source_data": source_data_str,
                "eval_issues": eval_issues_str,
            },
            metadata=metadata,
            vectors={"question_embedding": question_embedding},
            id=pair.id,
        )

    # ── Push ─────────────────────────────────────────────────────────────

    # TRACE-TOUR.extract[22] — push_pairs() entity-replace
    def push_pairs(self, pairs: list[QAPair]) -> tuple[int, int]:
        """Push Q&A pairs to Argilla with entity-replace semantics.

        Groups pairs by source_ref. For each source_ref, deletes existing
        records (archiving annotated ones) before pushing fresh.

        Returns:
            Tuple of (pushed_count, archived_count).
        """
        if not pairs:
            return 0, 0

        dataset = self.get_or_create_dataset()

        by_source_ref: dict[str, list[QAPair]] = defaultdict(list)
        for pair in pairs:
            by_source_ref[pair.source_ref].append(pair)

        total_pushed = 0
        total_archived = 0

        for source_ref, entity_pairs in by_source_ref.items():
            _, archived = self.delete_records_by_source_ref(source_ref)
            total_archived += archived

            records = [self.qa_pair_to_record(p) for p in entity_pairs]
            dataset.records.log(records)
            total_pushed += len(records)
            logger.info("Pushed %d records for source_ref=%s", len(records), source_ref)

        return total_pushed, total_archived

    def push_from_jsonl(self, filepath: str) -> tuple[int, int]:
        """Push Q&A pairs from a JSONL file to Argilla.

        Returns:
            Tuple of (pushed_count, archived_count).
        """
        from .output.jsonl_writer import load_jsonl

        pairs = load_jsonl(filepath)
        return self.push_pairs(pairs)
