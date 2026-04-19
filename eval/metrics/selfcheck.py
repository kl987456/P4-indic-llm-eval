"""
SelfCheckGPT hallucination detector.

Algorithm:
  1. Sample N completions from the model for the same prompt.
  2. Run an NLI model over all pairs.
  3. Hallucination score = mean contradiction rate across pairs.
  4. Score > 0.4 → likely hallucination.

Reference: Manakul et al. (2023) "SelfCheckGPT: Zero-Resource
           Black-Box Hallucination Detection for Generative LLMs"
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SelfCheckGPT:
    """
    Zero-resource hallucination detector using NLI consistency checking.

    Parameters
    ----------
    nli_model_name : str
        Cross-encoder NLI model from HuggingFace.
        Default: cross-encoder/nli-deberta-v3-base (SOTA NLI, ~180MB).
    batch_size : int
        Pairs sent to the NLI model per batch (tune for GPU memory).
    threshold : float
        Contradiction score above which a response is flagged.
    """

    NLI_LABEL_IDX = {
        "contradiction": 2,
        "neutral": 1,
        "entailment": 0,
    }

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        batch_size: int = 64,
        threshold: float = 0.4,
    ) -> None:
        self.nli_model_name = nli_model_name
        self.batch_size = batch_size
        self.threshold = threshold
        self._nli: Optional[object] = None  # Lazy load

    # ------------------------------------------------------------------

    def _load_nli(self) -> None:
        if self._nli is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            logger.info("Loading NLI model: %s", self.nli_model_name)
            self._nli = CrossEncoder(
                self.nli_model_name,
                num_labels=3,  # entailment / neutral / contradiction
            )

    # ------------------------------------------------------------------

    def score(self, completions: List[str]) -> float:
        """
        Compute hallucination score from a list of sampled completions.

        Parameters
        ----------
        completions : List[str]
            Multiple independent samples for the *same* prompt.

        Returns
        -------
        float
            Score in [0, 1]. Higher = more likely hallucination.
        """
        if len(completions) < 2:
            return 0.0

        self._load_nli()

        # All ordered pairs (i < j) — O(n²) pairs
        pairs = list(itertools.combinations(completions, 2))

        # Run NLI in batches
        all_scores: List[np.ndarray] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            logits = self._nli.predict(batch, apply_softmax=True)
            all_scores.append(np.array(logits))

        if not all_scores:
            return 0.0

        scores_matrix = np.concatenate(all_scores, axis=0)  # shape (num_pairs, 3)
        # Index 2 = contradiction in DeBERTa NLI (entail=0, neutral=1, contradict=2)
        contradiction_scores = scores_matrix[:, self.NLI_LABEL_IDX["contradiction"]]
        return float(np.mean(contradiction_scores))

    def score_symmetric(self, completions: List[str]) -> float:
        """
        Bidirectional NLI: check A→B and B→A, take max contradiction.
        More conservative (catches more hallucinations).
        """
        if len(completions) < 2:
            return 0.0

        self._load_nli()

        # All ordered pairs including both directions
        all_pairs = [
            (completions[i], completions[j])
            for i in range(len(completions))
            for j in range(len(completions))
            if i != j
        ]

        all_scores: List[np.ndarray] = []
        for i in range(0, len(all_pairs), self.batch_size):
            batch = all_pairs[i : i + self.batch_size]
            logits = self._nli.predict(batch, apply_softmax=True)
            all_scores.append(np.array(logits))

        scores_matrix = np.concatenate(all_scores, axis=0)
        contradiction_scores = scores_matrix[:, self.NLI_LABEL_IDX["contradiction"]]
        return float(np.mean(contradiction_scores))

    def is_hallucination(self, completions: List[str]) -> bool:
        """Return True if the hallucination score exceeds the threshold."""
        return self.score(completions) > self.threshold

    def score_with_explanation(self, completions: List[str]) -> dict:
        """
        Full diagnostic output including per-pair contradiction scores.
        """
        if len(completions) < 2:
            return {"score": 0.0, "flagged": False, "pairs": []}

        self._load_nli()
        pairs = list(itertools.combinations(range(len(completions)), 2))
        text_pairs = [(completions[i], completions[j]) for i, j in pairs]

        all_scores: List[np.ndarray] = []
        for i in range(0, len(text_pairs), self.batch_size):
            batch = text_pairs[i : i + self.batch_size]
            logits = self._nli.predict(batch, apply_softmax=True)
            all_scores.append(np.array(logits))

        scores_matrix = np.concatenate(all_scores, axis=0)

        pair_results = []
        for idx, (i, j) in enumerate(pairs):
            pair_results.append({
                "sample_a": i,
                "sample_b": j,
                "entailment": float(scores_matrix[idx, 0]),
                "neutral": float(scores_matrix[idx, 1]),
                "contradiction": float(scores_matrix[idx, 2]),
            })

        mean_contradiction = float(np.mean(scores_matrix[:, 2]))
        return {
            "score": mean_contradiction,
            "flagged": mean_contradiction > self.threshold,
            "num_samples": len(completions),
            "num_pairs": len(pairs),
            "pairs": pair_results,
        }
