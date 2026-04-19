"""
OpenAI adapter — wraps openai>=1.0 SDK.
Supports gpt-3.5-turbo, gpt-4o, gpt-4-turbo, etc.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI chat models.

    Usage::

        adapter = OpenAIAdapter("gpt-4o", api_key="sk-...")
        response = adapter.complete("What is the capital of India?")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package not installed. Run: pip install openai") from exc

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
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
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content.strip()

    def complete_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """
        Request n completions in a single API call using the `n` parameter
        (more efficient than n separate calls).
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return [choice.message.content.strip() for choice in response.choices]

    def complete_with_system(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Send a system + user message pair."""
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def embed(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Return embedding vector for *text*."""
        response = self._client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
