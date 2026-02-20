"""Configuration for MCP server connections and extraction parameters."""

import os

from pydantic import BaseModel


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    url: str
    tools: list[str]


class ExtractionConfig(BaseModel):
    """Controls how much data an extractor fetches and how the LLM processes it.

    The 5 extractors use 3 different strategies, so not every field applies to every
    extractor. Here's how they map:

    LIST-ALL extractors (compute-resources, affinity-groups):
        These servers return all entities with a single empty query, so search_limit
        and max_queries don't apply. The extractor fetches everything, then calls the
        LLM once per entity.
        - compute-resources: 1 call to search_resources (all ~26 HPC systems), then
          1 call to get_resource_hardware per system.
        - affinity-groups: 1 call to search_affinity_groups (all ~54 groups), then
          1 detail call per group for events + knowledge base. max_detail_items caps
          how many events/KB entries are included in the LLM prompt per group.

    SEARCH-TERMS extractor (software-discovery):
        Has a curated list of ~34 search terms (python, cuda, gcc, tensorflow, ...).
        For each term, it calls search_software with that term as the query.
        - max_queries: how many terms from the list to use. None = all 34.
          Set to 2 for cheap test runs, ramp up for production.
        - search_limit: max results per search term. The MCP server returns up to
          this many software packages per query. Deduplication by software name
          happens across all queries.

    BROAD-QUERIES extractors (allocations, nsf-awards):
        These servers require a search parameter (no list-all). Each extractor has
        a list of ~26 queries organized by dimension (fields of science, HPC topics,
        resource names, institutions, etc.).
        - max_queries: how many queries from the list to use. None = all ~26.
          Set to 1 for cheap test runs.
        - search_limit: max results per query. The MCP server returns up to this
          many projects/awards per query. Deduplication by project ID / award number
          happens across all queries.

    SHARED across all extractors:
        - max_entities: cap on how many entities get sent to the LLM, regardless
          of strategy. Applied after fetch and dedup, before LLM calls. None = no
          limit (process all). Set to 1 for cheap single-entity test runs. Works
          for list-all, search-terms, and broad-queries strategies alike.
        - max_tokens: token limit for each LLM generation call. Every extractor
          calls the LLM once per entity, asking it to produce a JSON array of Q&A
          pairs. 2048 is enough for ~5-8 Q&A pairs per entity.
    """

    # Cap on how many entities get sent to the LLM for Q&A generation.
    # Applied after fetching and deduplication, before any LLM calls.
    # None = no limit (process all). Set to 1 for cheap single-entity test runs.
    max_entities: int | None = None

    # How many queries to use from the extractor's search term / query list.
    # Only affects search-terms and broad-queries extractors.
    # None = use all queries in the list. Set low (1-3) for cheap test runs.
    max_queries: int | None = None

    # Max results the MCP server should return per query.
    # Only affects search-terms and broad-queries extractors.
    # software-discovery default was 20, allocations/nsf-awards default was 50.
    search_limit: int = 20

    # Token limit for each LLM call (one call per entity).
    # 2048 produces ~5-8 Q&A pairs. Increase if answers are getting truncated.
    max_tokens: int = 2048

    # Max events/knowledge-base items included per affinity group in the LLM prompt.
    # Only affects affinity-groups extractor. Keeps the prompt from getting huge
    # for groups with many events.
    max_detail_items: int = 5

    # Skip LLM judge evaluation (no quality scores on pairs). Set via --no-judge CLI flag.
    no_judge: bool = False

    # Only process these specific entity IDs. None = process all.
    # Set via --entity-ids CLI flag. Useful for targeted comparison runs.
    entity_ids: list[str] | None = None

    # Prompt strategy for Q&A generation. Controls how the freeform prompt is structured.
    # "baseline" = current freeform (categories as loose guidance)
    # "field-aware" = one-shot with required pairs per data field + freeform bonus
    # "two-shot" = battery (required fields) then discovery (what's unique/missing?)
    prompt_strategy: str = "baseline"


class Config(BaseModel):
    """Application configuration."""

    servers: dict[str, MCPServerConfig]
    extraction: dict[str, ExtractionConfig]
    output_dir: str = "data/output"

    def get_extraction_config(self, server_name: str) -> ExtractionConfig:
        """Get extraction config for a server, falling back to defaults."""
        return self.extraction.get(server_name, ExtractionConfig())

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        servers = {
            "compute-resources": MCPServerConfig(
                name="compute-resources",
                url=os.getenv("MCP_COMPUTE_RESOURCES_URL", "http://localhost:3002"),
                tools=["search_resources", "get_resource_hardware"],
            ),
            "software-discovery": MCPServerConfig(
                name="software-discovery",
                url=os.getenv("MCP_SOFTWARE_DISCOVERY_URL", "http://localhost:3004"),
                tools=["search_software"],
            ),
            "allocations": MCPServerConfig(
                name="allocations",
                url=os.getenv("MCP_ALLOCATIONS_URL", "http://localhost:3006"),
                tools=["search_projects", "get_allocation_statistics"],
            ),
            "affinity-groups": MCPServerConfig(
                name="affinity-groups",
                url=os.getenv("MCP_AFFINITY_GROUPS_URL", "http://localhost:3011"),
                tools=["search_affinity_groups"],
            ),
            "nsf-awards": MCPServerConfig(
                name="nsf-awards",
                url=os.getenv("MCP_NSF_AWARDS_URL", "http://localhost:3007"),
                tools=["search_nsf_awards"],
            ),
        }

        # Env var overrides apply to ALL extractors. Per-server overrides can be
        # added later if needed, but for now a global knob is simpler.
        env_max_entities = os.getenv("EXTRACT_MAX_ENTITIES")
        env_max_queries = os.getenv("EXTRACT_MAX_QUERIES")
        env_search_limit = os.getenv("EXTRACT_SEARCH_LIMIT")

        shared = ExtractionConfig(
            max_entities=int(env_max_entities) if env_max_entities else None,
            max_queries=int(env_max_queries) if env_max_queries else None,
            search_limit=int(env_search_limit) if env_search_limit else 20,
        )

        # Every server gets the same extraction config by default.
        # The fields that don't apply to a given extractor type are simply ignored
        # (e.g., compute-resources ignores max_queries and search_limit).
        extraction = {name: shared.model_copy() for name in servers}

        return cls(
            servers=servers,
            extraction=extraction,
            output_dir=os.getenv("QA_OUTPUT_DIR", "data/output"),
        )
