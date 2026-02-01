"""Configuration for MCP server connections."""

import os
from pydantic import BaseModel


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    url: str
    tools: list[str]


class Config(BaseModel):
    """Application configuration."""

    servers: dict[str, MCPServerConfig]
    output_dir: str = "data/output"

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
        return cls(
            servers=servers,
            output_dir=os.getenv("QA_OUTPUT_DIR", "data/output"),
        )
