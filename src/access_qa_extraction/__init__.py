"""ACCESS Q&A Extraction - Generate training data from MCP servers."""

__version__ = "0.1.0"

from .citation_validator import (
    AnswerValidationResult,
    Citation,
    CitationValidator,
    ValidationResult,
)
from .llm_client import (
    AnthropicClient,
    BaseLLMClient,
    LocalLLMClient,
    TransformersClient,
    get_llm_client,
)
