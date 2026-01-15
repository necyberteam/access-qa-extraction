"""HTTP client for calling MCP server tools."""

import json
from typing import Any

import httpx

from .config import MCPServerConfig


class MCPClient:
    """Client for calling MCP server tool endpoints."""

    def __init__(self, config: MCPServerConfig, timeout: float = 30.0):
        self.config = config
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MCPClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call an MCP tool and return the parsed response.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Parsed response data

        Raises:
            httpx.HTTPError: On HTTP errors
            ValueError: On invalid response format
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        url = f"{self.config.url}/tools/{tool_name}"
        response = await self._client.post(url, json={"arguments": arguments or {}})
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> Any:
        """Parse MCP response format.

        MCP responses come in format:
        {"content": [{"type": "text", "text": "...json or text..."}]}
        """
        if "content" not in data:
            return data

        content = data["content"]
        if not content or not isinstance(content, list):
            return data

        first_item = content[0]
        if first_item.get("type") == "text":
            text = first_item.get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        return data
