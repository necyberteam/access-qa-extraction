"""Incremental extraction support — hash-based change detection.

Computes a content hash per entity so that unchanged entities can be skipped
on subsequent runs. The IncrementalCache stores entity hashes and serialized
QAPair dicts alongside the JSONL output directory.
"""

import hashlib
import json
from pathlib import Path

from ..models import QAPair


# GUIDED-TOUR.md § Step 5 — hash-based change detection; unchanged entities skip all LLM calls
def compute_entity_hash(entity_data: dict) -> str:
    """Deterministic SHA-256 of entity data for change detection.

    Returns first 16 hex chars (64 bits) — collision probability
    negligible for <100K entities.
    """
    canonical = json.dumps(entity_data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class IncrementalCache:
    """Cache of entity hashes and serialized Q&A pairs for incremental extraction.

    Stores a JSON file mapping "{domain}_{entity_id}" → {hash, pairs}.
    On subsequent runs, entities with unchanged hashes skip LLM calls
    and reuse cached pairs.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_file = Path(cache_dir) / ".extraction_cache.json"
        self._data: dict = self._load()
        self._hits = 0
        self._misses = 0

    def _load(self) -> dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def is_unchanged(self, domain: str, entity_id: str, current_hash: str) -> bool:
        """Check if entity data matches the cached hash."""
        key = f"{domain}_{entity_id}"
        cached = self._data.get(key, {})
        match = cached.get("hash") == current_hash
        if match:
            self._hits += 1
        else:
            self._misses += 1
        return match

    def get_cached_pairs(self, domain: str, entity_id: str) -> list[QAPair] | None:
        """Return cached QAPair objects for an entity, or None if not cached."""
        key = f"{domain}_{entity_id}"
        cached = self._data.get(key, {})
        pair_dicts = cached.get("pairs")
        if pair_dicts is None:
            return None
        return [QAPair.model_validate(p) for p in pair_dicts]

    def store(
        self, domain: str, entity_id: str, hash_val: str, pairs: list[QAPair]
    ):
        """Store entity hash and serialized pairs in the cache."""
        key = f"{domain}_{entity_id}"
        self._data[key] = {
            "hash": hash_val,
            "pairs": [p.model_dump(mode="json") for p in pairs],
        }

    def save(self):
        """Persist cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self._data, indent=2))

    @property
    def stats(self) -> tuple[int, int]:
        """Return (hits, misses) counts."""
        return self._hits, self._misses
