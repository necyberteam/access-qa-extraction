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


@dataclass
class ExtractionReport:
    """Stats from an MCP-only fetch (no LLM calls). Used by `qa-extract report`."""

    server_name: str
    strategy: str  # "list-all", "search-terms", or "broad-queries"
    queries_used: list[str]  # search terms/queries used (empty for list-all)
    total_fetched: int  # total results before dedup
    unique_entities: int  # after dedup
    sample_ids: list[str] = field(default_factory=list)  # first 5 entity IDs


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

    async def report(self) -> ExtractionReport:
        """Fetch data from MCP and return coverage stats (no LLM calls).

        Subclasses should override this. The default raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.server_name} does not implement report()")

    async def run(self) -> ExtractionOutput:
        """Run extraction with managed HTTP client."""
        async with MCPClient(self.config) as client:
            self.client = client
            return await self.extract()

    async def run_report(self) -> ExtractionReport:
        """Run report with managed HTTP client."""
        async with MCPClient(self.config) as client:
            self.client = client
            return await self.report()
