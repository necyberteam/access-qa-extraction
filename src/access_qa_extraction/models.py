"""Data models for Q&A pairs."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a Q&A conversation."""

    role: Literal["user", "assistant"]
    content: str


class QAMetadata(BaseModel):
    """Metadata for a Q&A pair."""

    complexity: Literal["simple", "moderate", "complex"] = "simple"
    granularity: Literal[
        "comprehensive", "factoid", "comparison", "exploratory"
    ] = "comprehensive"
    has_citation: bool = True
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    source_hash: str | None = None
    source_modified: str | None = None
    source_data: dict | None = None  # Raw data preview for reviewer verification
    # LLM judge evaluation scores (populated by generators/judge.py)
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    completeness_score: float | None = None
    confidence_score: float | None = None
    eval_issues: list[str] | None = None
    suggested_decision: Literal["approved", "needs_review"] | None = None


class QAPair(BaseModel):
    """A single Q&A pair for training.

    Schema matches access-qa-planning/02-training-data.md
    """

    id: str
    source: Literal["mcp_extraction", "user_qa", "doc_generated"]
    source_ref: str
    domain: str
    messages: list[Message]
    metadata: QAMetadata = Field(default_factory=QAMetadata)

    @classmethod
    def create(
        cls,
        id: str,
        question: str,
        answer: str,
        source_ref: str,
        domain: str,
        complexity: Literal["simple", "moderate", "complex"] = "simple",
        granularity: Literal[
            "comprehensive", "factoid", "comparison", "exploratory"
        ] = "comprehensive",
        source_data: dict | None = None,
        source_hash: str | None = None,
    ) -> "QAPair":
        """Create a Q&A pair from question and answer strings."""
        has_citation = "<<SRC:" in answer
        return cls(
            id=id,
            source="mcp_extraction",
            source_ref=source_ref,
            domain=domain,
            messages=[
                Message(role="user", content=question),
                Message(role="assistant", content=answer),
            ],
            metadata=QAMetadata(
                complexity=complexity,
                granularity=granularity,
                has_citation=has_citation,
                source_data=source_data,
                source_hash=source_hash,
            ),
        )


# Type alias for extraction results
ExtractionResult = list[QAPair]
