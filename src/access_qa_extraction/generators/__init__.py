"""Deterministic Q&A generators (zero LLM, zero hallucination)."""

from .comparisons import ComparisonGenerator
from .factoids import generate_factoid_pairs

__all__ = ["ComparisonGenerator", "generate_factoid_pairs"]
