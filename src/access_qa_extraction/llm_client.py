"""LLM client abstraction supporting Anthropic and local models."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM."""
    text: str
    model: str
    usage: Optional[dict] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system: System prompt
            user: User message
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text
        """
        pass


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it in your environment or create a .env file."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model or os.getenv("QA_EXTRACTION_MODEL", "claude-sonnet-4-20250514")

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}]
        )

        return LLMResponse(
            text=response.content[0].text,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )


class LocalLLMClient(BaseLLMClient):
    """Client for local LLM via OpenAI-compatible API (vLLM, ollama, etc.)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str | None = None,
    ):
        from openai import OpenAI

        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")
        self.client = OpenAI(base_url=self.base_url, api_key=resolved_key)

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            } if response.usage else None
        )


class TransformersClient(BaseLLMClient):
    """Client for local LLM using transformers directly (no server needed)."""

    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path or os.getenv(
            "LOCAL_LLM_PATH",
            "/home/apasquale/models/Qwen3-235B-A22B-Instruct-2507-FP8"
        )
        self.device = device

        # Use class-level caching to avoid reloading the model
        if TransformersClient._model is None:
            print(f"Loading model from {self.model_path}...")
            TransformersClient._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            TransformersClient._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model loaded!")

        self.model = TransformersClient._model
        self.tokenizer = TransformersClient._tokenizer

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        import torch

        # Format as chat messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking for extraction
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_ids = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return LLMResponse(
            text=response_text,
            model=self.model_path,
            usage={
                "input_tokens": input_length,
                "output_tokens": len(generated_ids),
            }
        )


def get_llm_client(backend: Optional[str] = None) -> BaseLLMClient:
    """Get an LLM client based on configuration.

    Args:
        backend: "anthropic", "local", or "transformers". If None, uses LLM_BACKEND env var.

    Returns:
        Configured LLM client
    """
    backend = backend or os.getenv("LLM_BACKEND", "anthropic")

    if backend == "anthropic":
        return AnthropicClient()
    elif backend == "local":
        return LocalLLMClient()
    elif backend == "transformers":
        return TransformersClient()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Use 'anthropic', 'local', or 'transformers'.")
