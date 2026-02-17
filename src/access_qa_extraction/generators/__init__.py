"""Deterministic Q&A generators (zero LLM, zero hallucination)."""

from .comparisons import ComparisonGenerator
from .factoids import generate_factoid_pairs
from .incremental import IncrementalCache, compute_entity_hash

__all__ = [
    "ComparisonGenerator",
    "IncrementalCache",
    "compute_entity_hash",
    "generate_factoid_pairs",
]
