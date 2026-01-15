"""ACCESS Q&A Extraction - Generate training data from MCP servers."""

__version__ = "0.1.0"

from .llm_client import (
    BaseLLMClient,
    AnthropicClient,
    LocalLLMClient,
    TransformersClient,
    get_llm_client,
)

from .citation_validator import (
    Citation,
    CitationValidator,
    ValidationResult,
    AnswerValidationResult,
)
