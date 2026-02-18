"""Argilla client for pushing Q&A pairs to review."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from .models import QAPair

if TYPE_CHECKING:
    import argilla as rg

logger = logging.getLogger(__name__)

# Dataset schema constants
DATASET_NAME = "qa-review"
WORKSPACE_NAME = "default"
SIMILARITY_THRESHOLD = 0.85
VECTOR_DIMENSIONS = 384  # all-MiniLM-L6-v2 output dimensions


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
    """Client for pushing Q&A pairs to Argilla for human review."""

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

    def get_or_create_dataset(self) -> rg.Dataset:
        """Get existing qa-review dataset or create it."""
        if self._dataset is not None:
            return self._dataset

        rg = self.rg

        # Try to get existing dataset
        try:
            self._dataset = self.client.datasets(name=DATASET_NAME, workspace=self.workspace)
            if self._dataset is not None:
                logger.info("Found existing dataset '%s'", DATASET_NAME)
                return self._dataset
        except Exception:
            pass

        # Create new dataset
        settings = rg.Settings(
            guidelines=(
                "Review Q&A pairs extracted from ACCESS-CI MCP servers. "
                "Check that questions are clear, answers are accurate "
                "and well-cited, and source data supports the answer."
            ),
            fields=[
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
            ],
            questions=[
                rg.LabelQuestion(
                    name="review_decision",
                    title="Review Decision",
                    labels=[
                        "approved",
                        "rejected",
                        "needs_edit",
                        "flagged",
                    ],
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
            ],
            metadata=[
                rg.TermsMetadataProperty(name="domain", title="Domain"),
                rg.TermsMetadataProperty(name="source_type", title="Source Type"),
                rg.TermsMetadataProperty(name="complexity", title="Complexity"),
                rg.TermsMetadataProperty(name="granularity", title="Granularity"),
                rg.TermsMetadataProperty(name="has_citation", title="Has Citation"),
                # LLM judge evaluation scores
                rg.FloatMetadataProperty(name="faithfulness_score", title="Faithfulness Score"),
                rg.FloatMetadataProperty(name="relevance_score", title="Relevance Score"),
                rg.FloatMetadataProperty(name="completeness_score", title="Completeness Score"),
                rg.FloatMetadataProperty(name="confidence_score", title="Confidence Score"),
                rg.TermsMetadataProperty(name="suggested_decision", title="Suggested Decision"),
                rg.TermsMetadataProperty(name="source_ref", title="Source Reference"),
            ],
            vectors=[
                rg.VectorField(
                    name="question_embedding",
                    dimensions=VECTOR_DIMENSIONS,
                ),
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

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text string using all-MiniLM-L6-v2."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def find_duplicate(self, question_embedding: list[float]) -> bool:
        """Check if a similar record already exists in the dataset.

        Returns True if a record with similarity >= SIMILARITY_THRESHOLD exists.
        """
        rg = self.rg
        dataset = self.get_or_create_dataset()
        try:
            similar_query = rg.Query(
                similar=rg.Similar(
                    name="question_embedding",
                    value=question_embedding,
                )
            )
            similar_records = dataset.records(similar_query).to_list(flatten=True)
            return len(similar_records) > 0
        except Exception as e:
            logger.warning("Duplicate check failed: %s", e)
            return False

    def qa_pair_to_record(self, pair: QAPair) -> rg.Record:
        """Convert a QAPair to an Argilla Record."""
        rg = self.rg
        question = pair.messages[0].content if pair.messages else ""
        answer = pair.messages[1].content if len(pair.messages) > 1 else ""

        # Format source_data as JSON for display
        source_data_str = ""
        if pair.metadata.source_data:
            raw = json.dumps(pair.metadata.source_data, indent=2, default=str)
            source_data_str = f"```json\n{raw[:5000]}\n```"

        # Format eval_issues as readable text for reviewer
        eval_issues_str = ""
        if pair.metadata.eval_issues:
            eval_issues_str = "; ".join(pair.metadata.eval_issues)

        # Generate embedding
        question_embedding = self.generate_embedding(question)

        metadata = {
            "domain": pair.domain,
            "source_type": pair.source,
            "complexity": pair.metadata.complexity,
            "granularity": pair.metadata.granularity,
            "has_citation": str(pair.metadata.has_citation),
            "source_ref": pair.source_ref,
        }

        # Add judge scores if present
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

    def push_pairs(
        self,
        pairs: list[QAPair],
        check_duplicates: bool = True,
    ) -> tuple[int, int]:
        """Push Q&A pairs to Argilla.

        Args:
            pairs: Q&A pairs to push.
            check_duplicates: Whether to check for duplicates.

        Returns:
            Tuple of (pushed_count, skipped_count).
        """
        dataset = self.get_or_create_dataset()

        records = []
        skipped = 0

        for pair in pairs:
            record = self.qa_pair_to_record(pair)

            if check_duplicates:
                embedding = record.vectors["question_embedding"]
                if self.find_duplicate(embedding):
                    logger.info("Skipping duplicate: %s", pair.id)
                    skipped += 1
                    continue

            records.append(record)

        if records:
            dataset.records.log(records)
            logger.info("Pushed %d records to Argilla", len(records))

        return len(records), skipped

    def push_from_jsonl(
        self,
        filepath: str,
        check_duplicates: bool = True,
    ) -> tuple[int, int]:
        """Push Q&A pairs from a JSONL file to Argilla.

        Args:
            filepath: Path to JSONL file.
            check_duplicates: Whether to check for duplicates.

        Returns:
            Tuple of (pushed_count, skipped_count).
        """
        from .output.jsonl_writer import load_jsonl

        pairs = load_jsonl(filepath)
        return self.push_pairs(pairs, check_duplicates=check_duplicates)
