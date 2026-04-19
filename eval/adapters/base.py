"""
Abstract base class for all model adapters.
Every concrete adapter must implement `complete` and `complete_n`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Base interface for LLM adapters used in the evaluation harness."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._kwargs = kwargs
        # In-process cache: key -> response string
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a single completion for *prompt*.
        Temperature=0 for deterministic eval; higher for sampling.
        """
        ...

    @abstractmethod
    def complete_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        """
        Generate *n* independent completions — used by SelfCheckGPT.
        """
        ...

    # ------------------------------------------------------------------
    # Caching layer — O(1) dict lookup by hash of (prompt, kwargs)
    # ------------------------------------------------------------------

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        raw = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def complete_cached(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        Wrapper around `complete` with in-memory caching.
        Identical (prompt, params) pairs are served from cache in O(1).
        """
        key = self._cache_key(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if key not in self._cache:
            logger.debug("Cache MISS — calling model %s", self.model_name)
            self._cache[key] = self.complete(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
        else:
            logger.debug("Cache HIT — skipping model call")
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "adapter": self.__class__.__name__,
            "model_name": self.model_name,
            "cache_size": len(self._cache),
        }
