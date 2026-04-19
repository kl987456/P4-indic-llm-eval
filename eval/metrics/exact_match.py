"""
Exact-match metric with Unicode normalisation and script-aware tokenisation.
Handles Hindi (Devanagari), Telugu, Tamil, and Hinglish.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


class ExactMatchMetric:
    """
    Computes exact match between predicted and reference answers.

    Normalisation steps applied before comparison:
    1. Unicode NFC normalisation
    2. Lowercase (for Latin-script tokens)
    3. Strip leading/trailing whitespace
    4. Collapse internal whitespace
    5. Remove punctuation (configurable)
    6. Strip common Indic articles / particles (optional)
    """

    INDIC_NOISE = re.compile(r"[\u0964\u0965।॥]+")  # Devanagari danda
    PUNCT = re.compile(r"[^\w\s\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF]")

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punct: bool = True,
        strip_articles: bool = False,
    ) -> None:
        self.ignore_case = ignore_case
        self.ignore_punct = ignore_punct
        self.strip_articles = strip_articles
        # Common Hindi/English articles
        self._articles = {"the", "a", "an", "एक", "यह", "वह", "ये", "वे"}

    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        # Unicode NFC
        text = unicodedata.normalize("NFC", text)
        # Remove Indic dandas
        text = self.INDIC_NOISE.sub(" ", text)
        if self.ignore_punct:
            text = self.PUNCT.sub(" ", text)
        if self.ignore_case:
            text = text.lower()
        # Collapse whitespace
        text = " ".join(text.split())
        if self.strip_articles:
            tokens = [t for t in text.split() if t not in self._articles]
            text = " ".join(tokens)
        return text.strip()

    def score(self, prediction: str, reference: str) -> float:
        """Return 1.0 if strings match after normalisation, else 0.0."""
        return 1.0 if self.normalize(prediction) == self.normalize(reference) else 0.0

    def score_batch(
        self, predictions: List[str], references: List[str]
    ) -> dict:
        """
        Score a batch of (prediction, reference) pairs.
        Returns dict with per-item scores and aggregate accuracy.
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have equal length")

        scores = [self.score(p, r) for p, r in zip(predictions, references)]
        return {
            "scores": scores,
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "num_correct": int(sum(scores)),
            "num_total": len(scores),
        }

    def top_k_score(
        self,
        predictions: List[List[str]],
        references: List[str],
        k: int = 5,
    ) -> float:
        """
        Pass@k exact match: 1.0 if any of the top-k predictions matches.
        """
        hits = 0
        for preds, ref in zip(predictions, references):
            if any(self.score(p, ref) == 1.0 for p in preds[:k]):
                hits += 1
        return hits / len(references) if references else 0.0
