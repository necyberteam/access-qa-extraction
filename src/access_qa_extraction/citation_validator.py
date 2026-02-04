"""Citation validation against MCP entities.

Validates that <<SRC:domain:entity_id>> citations reference real entities
that exist in the corresponding MCP servers.
"""

import re
from dataclasses import dataclass, field

from .config import Config
from .mcp_client import MCPClient

# Regex to extract citations: <<SRC:domain:entity_id>>
CITATION_PATTERN = re.compile(r"<<SRC:([^:>]+):([^>]+)>>")


@dataclass
class Citation:
    """A parsed citation from an answer."""

    domain: str
    entity_id: str
    raw: str  # Original citation string

    @classmethod
    def parse(cls, raw: str) -> "Citation | None":
        """Parse a citation string into a Citation object."""
        match = CITATION_PATTERN.match(raw)
        if not match:
            return None
        return cls(domain=match.group(1), entity_id=match.group(2), raw=raw)


@dataclass
class ValidationResult:
    """Result of validating a single citation."""

    citation: Citation
    valid: bool
    error: str | None = None


@dataclass
class AnswerValidationResult:
    """Result of validating all citations in an answer."""

    citations: list[ValidationResult] = field(default_factory=list)
    has_citations: bool = False

    @property
    def all_valid(self) -> bool:
        """True if all citations are valid."""
        return self.has_citations and all(c.valid for c in self.citations)

    @property
    def invalid_citations(self) -> list[ValidationResult]:
        """List of invalid citations."""
        return [c for c in self.citations if not c.valid]


class CitationValidator:
    """Validates citations against known MCP entities.

    Caches entity lists from MCP servers to avoid repeated API calls.
    """

    def __init__(self, config: Config):
        self.config = config
        self._entity_cache: dict[str, set[str]] = {}

    async def load_entities(self) -> None:
        """Load all known entities from MCP servers into cache."""
        await self._load_compute_resources()
        await self._load_software()
        await self._load_affinity_groups()
        await self._load_allocations()
        await self._load_nsf_awards()

    async def _load_compute_resources(self) -> None:
        """Load compute resource IDs from MCP server."""
        server_config = self.config.servers.get("compute-resources")
        if not server_config:
            return

        try:
            async with MCPClient(server_config) as client:
                data = await client.call_tool("search_resources", {})
                # Response is {"total": N, "items": [...]} or a list
                resources = data if isinstance(data, list) else data.get("items", data.get("resources", []))
                self._entity_cache["compute-resources"] = {
                    r.get("id") or r.get("resource_id") or r.get("ResourceID")
                    for r in resources
                    if r.get("id") or r.get("resource_id") or r.get("ResourceID")
                }
        except Exception as e:
            print(f"Warning: Could not load compute resources: {e}")
            self._entity_cache["compute-resources"] = set()

    async def _load_software(self) -> None:
        """Load software names from MCP server.

        Tries list_all_software first, falls back to querying by resource.
        """
        server_config = self.config.servers.get("software-discovery")
        if not server_config:
            return

        all_software: set[str] = set()
        try:
            async with MCPClient(server_config) as client:
                # Try list_all_software first (if server has been updated)
                try:
                    data = await client.call_tool("list_all_software", {"limit": 10000})
                    software_list = data.get("items", data.get("software", []))
                    for s in software_list:
                        name = s.get("name") or s.get("Name")
                        if name:
                            all_software.add(name)
                    if all_software:
                        self._entity_cache["software-discovery"] = all_software
                        return
                except Exception:
                    pass  # Fall back to resource-based query

                # Fall back: query software by resource ID
                # First get all resource IDs from compute-resources
                resource_ids = list(self._entity_cache.get("compute-resources", set()))
                for resource_id in resource_ids:
                    try:
                        data = await client.call_tool(
                            "search_software",
                            {"resource_id": resource_id, "limit": 500}
                        )
                        software_list = data.get("software", [])
                        for s in software_list:
                            name = s.get("name") or s.get("Name")
                            if name:
                                all_software.add(name)
                    except Exception:
                        pass

            self._entity_cache["software-discovery"] = all_software
        except Exception as e:
            print(f"Warning: Could not load software: {e}")
            self._entity_cache["software-discovery"] = set()

    async def _load_affinity_groups(self) -> None:
        """Load affinity group IDs from MCP server."""
        server_config = self.config.servers.get("affinity-groups")
        if not server_config:
            return

        try:
            async with MCPClient(server_config) as client:
                data = await client.call_tool("search_affinity_groups", {})
                groups = data.get("items", data.get("groups", []))
                self._entity_cache["affinity-groups"] = {
                    str(g.get("id"))
                    for g in groups
                    if g.get("id")
                }
        except Exception as e:
            print(f"Warning: Could not load affinity groups: {e}")
            self._entity_cache["affinity-groups"] = set()

    async def _load_allocations(self) -> None:
        """Load allocation project IDs from MCP server."""
        server_config = self.config.servers.get("allocations")
        if not server_config:
            return

        try:
            async with MCPClient(server_config) as client:
                data = await client.call_tool("search_projects", {})
                projects = data.get("items", data.get("projects", []))
                self._entity_cache["allocations"] = {
                    p.get("projectId") or p.get("requestNumber")
                    for p in projects
                    if p.get("projectId") or p.get("requestNumber")
                }
        except Exception as e:
            print(f"Warning: Could not load allocations: {e}")
            self._entity_cache["allocations"] = set()

    async def _load_nsf_awards(self) -> None:
        """Load NSF award numbers from MCP server."""
        server_config = self.config.servers.get("nsf-awards")
        if not server_config:
            return

        try:
            async with MCPClient(server_config) as client:
                data = await client.call_tool("search_nsf_awards", {})
                awards = data.get("items", data.get("awards", []))
                self._entity_cache["nsf-awards"] = {
                    str(a.get("awardNumber"))
                    for a in awards
                    if a.get("awardNumber")
                }
        except Exception as e:
            print(f"Warning: Could not load NSF awards: {e}")
            self._entity_cache["nsf-awards"] = set()

    def extract_citations(self, answer: str) -> list[Citation]:
        """Extract all citations from an answer string."""
        citations = []
        for match in CITATION_PATTERN.finditer(answer):
            citation = Citation(
                domain=match.group(1), entity_id=match.group(2), raw=match.group(0)
            )
            citations.append(citation)
        return citations

    def validate_citation(self, citation: Citation) -> ValidationResult:
        """Validate a single citation against cached entities."""
        # Check if domain is known
        if citation.domain not in self._entity_cache:
            # Allow unknown domains (docs, etc.) - they can't be validated
            return ValidationResult(
                citation=citation,
                valid=True,  # Assume valid for unknown domains
                error=None,
            )

        # Check if entity exists
        known_entities = self._entity_cache[citation.domain]
        if citation.entity_id in known_entities:
            return ValidationResult(citation=citation, valid=True)

        # Entity not found - this is a hallucination
        return ValidationResult(
            citation=citation,
            valid=False,
            error=f"Entity '{citation.entity_id}' not found in {citation.domain}",
        )

    def validate_answer(self, answer: str) -> AnswerValidationResult:
        """Validate all citations in an answer.

        Returns:
            AnswerValidationResult with validation status for each citation.
        """
        citations = self.extract_citations(answer)
        result = AnswerValidationResult(has_citations=len(citations) > 0)

        for citation in citations:
            validation = self.validate_citation(citation)
            result.citations.append(validation)

        return result

    def get_known_entities(self, domain: str) -> set[str]:
        """Get the set of known entities for a domain."""
        return self._entity_cache.get(domain, set())

    def add_entities(self, domain: str, entities: set[str]) -> None:
        """Manually add entities to the cache (for testing or custom sources)."""
        if domain not in self._entity_cache:
            self._entity_cache[domain] = set()
        self._entity_cache[domain].update(entities)

    def load_entities_from_pairs(self, pairs: list) -> None:
        """Load entities from existing Q&A pairs.

        This is useful when validating against the same data that was extracted,
        ensuring all cited entities are known.
        """
        for pair in pairs:
            if len(pair.messages) < 2:
                continue
            answer = pair.messages[1].content
            for citation in self.extract_citations(answer):
                if citation.domain not in self._entity_cache:
                    self._entity_cache[citation.domain] = set()
                self._entity_cache[citation.domain].add(citation.entity_id)
