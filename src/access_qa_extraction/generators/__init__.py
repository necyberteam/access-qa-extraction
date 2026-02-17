"""Q&A generators and evaluation."""

from .comparisons import ComparisonGenerator
from .factoids import generate_factoid_pairs
from .incremental import IncrementalCache, compute_entity_hash
from .judge import evaluate_pairs

__all__ = [
    "ComparisonGenerator",
    "IncrementalCache",
    "compute_entity_hash",
    "evaluate_pairs",
    "generate_factoid_pairs",
]
