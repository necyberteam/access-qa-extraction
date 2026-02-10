"""Base extractor interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import ExtractionConfig, MCPServerConfig
from ..mcp_client import MCPClient
from ..models import ExtractionResult


@dataclass
class ExtractionOutput:
    """Output from an extractor including Q&A pairs and raw data for comparisons."""

    pairs: ExtractionResult
    raw_data: dict = field(default_factory=dict)


class BaseExtractor(ABC):
    """Base class for MCP server extractors."""

    server_name: str

    def __init__(self, config: MCPServerConfig, extraction_config: ExtractionConfig | None = None):
        self.config = config
        self.extraction_config = extraction_config or ExtractionConfig()

    @abstractmethod
    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs from this MCP server.

        Returns:
            ExtractionOutput containing Q&A pairs and raw data for comparisons.
        """
        pass

    async def run(self) -> ExtractionOutput:
        """Run extraction with managed HTTP client."""
        async with MCPClient(self.config) as client:
            self.client = client
            return await self.extract()
