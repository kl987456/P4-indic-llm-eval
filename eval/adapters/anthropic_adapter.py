"""
Anthropic adapter — wraps anthropic>=0.25 SDK.
Supports claude-3-5-sonnet, claude-3-opus, claude-3-haiku, etc.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class AnthropicAdapter(ModelAdapter):
    """
    Adapter for Anthropic Claude models.

    Usage::

        adapter = AnthropicAdapter("claude-3-5-sonnet-20241022", api_key="sk-ant-...")
        response = adapter.complete("भारत की राजधानी क्या है?")
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
    ) -> str:
        kwargs = dict(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop

        response = self._client.messages.create(**kwargs)
        # Extract text from first content block
        return response.content[0].text.strip()

    def complete_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """
        Anthropic does not support n>1 in a single call.
        Fall back to sequential calls — still batched for SelfCheckGPT.
        """
        results = []
        for _ in range(n):
            results.append(
                self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            )
        return results

    def complete_with_system(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return self.complete(prompt, temperature=temperature, max_tokens=max_tokens, system=system)

    def count_tokens(self, prompt: str) -> int:
        """Estimate token count using Anthropic tokenizer."""
        return self._client.count_tokens(prompt)
