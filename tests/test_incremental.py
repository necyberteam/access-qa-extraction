"""Tests for incremental extraction — hash-based change detection.

Tests compute_entity_hash() determinism, IncrementalCache round-trip,
change detection, and source_hash population on QAPair.
"""

from pathlib import Path

from access_qa_extraction.generators.incremental import (
    IncrementalCache,
    compute_entity_hash,
)
from access_qa_extraction.models import QAPair


class TestComputeEntityHash:
    """Test the entity hashing function."""

    def test_deterministic(self):
        data = {"name": "Delta", "type": "Compute", "gpus": True}
        h1 = compute_entity_hash(data)
        h2 = compute_entity_hash(data)
        assert h1 == h2

    def test_key_order_irrelevant(self):
        data_a = {"name": "Delta", "type": "Compute"}
        data_b = {"type": "Compute", "name": "Delta"}
        assert compute_entity_hash(data_a) == compute_entity_hash(data_b)

    def test_different_data_different_hash(self):
        data_a = {"name": "Delta", "type": "Compute"}
        data_b = {"name": "Bridges-2", "type": "Compute"}
        assert compute_entity_hash(data_a) != compute_entity_hash(data_b)

    def test_length_is_16_hex(self):
        h = compute_entity_hash({"x": 1})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_nested_dicts(self):
        data = {"hw": {"gpus": [{"name": "A100"}]}, "name": "Delta"}
        h = compute_entity_hash(data)
        assert len(h) == 16


class TestIncrementalCache:
    """Test the IncrementalCache round-trip and change detection."""

    def _make_pair(self, pair_id: str = "test_pair") -> QAPair:
        return QAPair.create(
            id=pair_id,
            question="What is Delta?",
            answer="Delta is an HPC system.\n\n<<SRC:compute-resources:delta>>",
            source_ref="mcp://compute-resources/resources/delta",
            domain="compute-resources",
            source_hash="abc123",
        )

    def test_round_trip(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        pair = self._make_pair()
        cache.store("compute-resources", "delta", "hash123", [pair])
        cache.save()

        # Load fresh
        cache2 = IncrementalCache(tmp_path)
        assert cache2.is_unchanged("compute-resources", "delta", "hash123")
        cached_pairs = cache2.get_cached_pairs("compute-resources", "delta")
        assert cached_pairs is not None
        assert len(cached_pairs) == 1
        assert cached_pairs[0].id == "test_pair"
        assert cached_pairs[0].domain == "compute-resources"

    def test_detects_change(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        pair = self._make_pair()
        cache.store("compute-resources", "delta", "hash_v1", [pair])
        cache.save()

        cache2 = IncrementalCache(tmp_path)
        assert not cache2.is_unchanged("compute-resources", "delta", "hash_v2")

    def test_missing_entity(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        assert not cache.is_unchanged("compute-resources", "nonexistent", "anyhash")

    def test_empty_cache_file(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        assert not cache.is_unchanged("x", "y", "z")
        assert cache.get_cached_pairs("x", "y") is None

    def test_stats_tracking(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        pair = self._make_pair()
        cache.store("compute-resources", "delta", "hash1", [pair])

        cache.is_unchanged("compute-resources", "delta", "hash1")  # hit
        cache.is_unchanged("compute-resources", "delta", "hash2")  # miss
        cache.is_unchanged("compute-resources", "other", "hash1")  # miss

        hits, misses = cache.stats
        assert hits == 1
        assert misses == 2

    def test_multiple_pairs_per_entity(self, tmp_path: Path):
        cache = IncrementalCache(tmp_path)
        pairs = [
            self._make_pair("pair_1"),
            self._make_pair("pair_2"),
            self._make_pair("pair_3"),
        ]
        cache.store("compute-resources", "delta", "hash1", pairs)
        cache.save()

        cache2 = IncrementalCache(tmp_path)
        cached = cache2.get_cached_pairs("compute-resources", "delta")
        assert cached is not None
        assert len(cached) == 3
        assert [p.id for p in cached] == ["pair_1", "pair_2", "pair_3"]

    def test_corrupt_cache_file_handled(self, tmp_path: Path):
        cache_file = tmp_path / ".extraction_cache.json"
        cache_file.write_text("not valid json{{{")
        cache = IncrementalCache(tmp_path)
        assert not cache.is_unchanged("x", "y", "z")


class TestSourceHashOnQAPair:
    """Test that source_hash flows through QAPair.create()."""

    def test_source_hash_populated(self):
        pair = QAPair.create(
            id="test",
            question="Q",
            answer="A <<SRC:x:y>>",
            source_ref="mcp://x/y/z",
            domain="x",
            source_hash="abcdef1234567890",
        )
        assert pair.metadata.source_hash == "abcdef1234567890"

    def test_source_hash_default_none(self):
        pair = QAPair.create(
            id="test",
            question="Q",
            answer="A",
            source_ref="mcp://x/y/z",
            domain="x",
        )
        assert pair.metadata.source_hash is None
