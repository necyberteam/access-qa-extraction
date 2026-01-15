"""Tests for Q&A pair models."""

from access_qa_extraction.models import Message, QAMetadata, QAPair


class TestQAPair:
    """Tests for QAPair model."""

    def test_create_simple_pair(self):
        """Test creating a Q&A pair with the factory method."""
        pair = QAPair.create(
            id="test_001",
            question="What is Delta?",
            answer="Delta is a compute resource.\n\n<<SRC:compute-resources:delta>>",
            source_ref="mcp://compute-resources/resources/delta",
            domain="compute:resource_specs",
        )

        assert pair.id == "test_001"
        assert pair.source == "mcp_extraction"
        assert pair.source_ref == "mcp://compute-resources/resources/delta"
        assert pair.domain == "compute:resource_specs"
        assert len(pair.messages) == 2
        assert pair.messages[0].role == "user"
        assert pair.messages[0].content == "What is Delta?"
        assert pair.messages[1].role == "assistant"
        assert pair.metadata.has_citation is True
        assert pair.metadata.complexity == "simple"

    def test_create_pair_without_citation(self):
        """Test that has_citation is False when no citation marker."""
        pair = QAPair.create(
            id="test_002",
            question="What is the weather?",
            answer="I cannot answer questions about weather.",
            source_ref="mcp://test/deferral",
            domain="status:defer_to_live",
        )

        assert pair.metadata.has_citation is False

    def test_serialization(self):
        """Test JSONL serialization."""
        pair = QAPair.create(
            id="test_003",
            question="Test question",
            answer="Test answer <<SRC:test:123>>",
            source_ref="mcp://test/123",
            domain="test:domain",
        )

        json_str = pair.model_dump_json()
        assert "test_003" in json_str
        assert "Test question" in json_str
        assert "mcp_extraction" in json_str

        # Verify it can be deserialized
        restored = QAPair.model_validate_json(json_str)
        assert restored.id == pair.id
        assert restored.messages[0].content == pair.messages[0].content
