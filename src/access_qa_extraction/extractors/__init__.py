"""Extractors for each MCP server."""

from .base import BaseExtractor, ExtractionOutput
from .affinity_groups import AffinityGroupsExtractor
from .allocations import AllocationsExtractor
from .compute_resources import ComputeResourcesExtractor
from .software_discovery import SoftwareDiscoveryExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionOutput",
    "AffinityGroupsExtractor",
    "AllocationsExtractor",
    "ComputeResourcesExtractor",
    "SoftwareDiscoveryExtractor",
]
