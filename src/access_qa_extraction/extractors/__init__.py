"""Extractors for each MCP server."""

from .base import BaseExtractor, ExtractionOutput
from .affinity_groups import AffinityGroupsExtractor
from .compute_resources import ComputeResourcesExtractor
from .software_discovery import SoftwareDiscoveryExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionOutput",
    "AffinityGroupsExtractor",
    "ComputeResourcesExtractor",
    "SoftwareDiscoveryExtractor",
]
