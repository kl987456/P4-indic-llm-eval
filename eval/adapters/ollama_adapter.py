"""
Ollama adapter — calls the local Ollama HTTP API.
Ollama must be running at http://localhost:11434 (or custom base_url).
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import requests

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(ModelAdapter):
    """
    Adapter for models served via Ollama (llama3, mistral, gemma, etc.).

    Usage::

        adapter = OllamaAdapter("llama3:8b")
        response = adapter.complete("What is the Mahabharata about?")
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self._base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        logger.debug("POST %s payload=%s", url, payload)
        resp = self._session.post(url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_stream(self, endpoint: str, payload: dict) -> str:
        """Stream response from Ollama (ndjson) and collect full text."""
        url = f"{self._base_url}{endpoint}"
        payload["stream"] = True
        resp = self._session.post(url, json=payload, timeout=self._timeout, stream=True)
        resp.raise_for_status()

        full_text = ""
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
        return full_text.strip()

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        data = self._post("/api/generate", payload)
        return data.get("response", "").strip()

    def complete_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """Ollama does not natively support n>1; run n sequential calls."""
        return [
            self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            for _ in range(n)
        ]

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Use Ollama /api/chat endpoint for multi-turn conversations."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "").strip()

    def list_models(self) -> List[str]:
        """Return list of locally available Ollama models."""
        resp = self._session.get(f"{self._base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            resp = self._session.get(f"{self._base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
