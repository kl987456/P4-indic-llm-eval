"""
BERTScore metric wrapper using the bert-score library.
Supports multilingual evaluation with language-specific model selection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Best models per script/language for BERTScore
LANG_MODEL_MAP = {
    "hi": "l3cube-pune/hindi-bert-v2",
    "te": "l3cube-pune/telugu-bert",
    "ta": "l3cube-pune/tamil-bert",
    "en": "roberta-large",
    "hinglish": "google/muril-base-cased",
    "multilingual": "google/muril-base-cased",
}


class BertScoreMetric:
    """
    Computes BERTScore (Precision, Recall, F1) between predictions and references.

    Language-specific BERT models are automatically selected for Indic languages.
    Falls back to multilingual MuRIL when no specific model is available.
    """

    def __init__(
        self,
        language: str = "hi",
        model_type: Optional[str] = None,
        num_layers: Optional[int] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        rescale_with_baseline: bool = True,
    ) -> None:
        self.language = language
        self.model_type = model_type or LANG_MODEL_MAP.get(language, LANG_MODEL_MAP["multilingual"])
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline

        try:
            from bert_score import BERTScorer
            import torch
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError as exc:
            raise ImportError("bert-score not installed. Run: pip install bert-score") from exc

        logger.info("Initialising BERTScorer model=%s device=%s", self.model_type, self._device)
        from bert_score import BERTScorer
        self._scorer = BERTScorer(
            model_type=self.model_type,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            lang=language,
            device=self._device,
            rescale_with_baseline=self.rescale_with_baseline,
        )

    # ------------------------------------------------------------------

    def score(self, prediction: str, reference: str) -> Dict[str, float]:
        """Score a single prediction-reference pair. Returns P, R, F1."""
        P, R, F = self._scorer.score([prediction], [reference])
        return {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F[0]),
        }

    def score_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        """
        Score a batch of (prediction, reference) pairs.
        Returns aggregated dict with per-sample and mean scores.
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have equal length")

        P, R, F = self._scorer.score(predictions, references)

        per_sample = [
            {"precision": float(p), "recall": float(r), "f1": float(f)}
            for p, r, f in zip(P, R, F)
        ]
        return {
            "per_sample": per_sample,
            "mean_precision": float(P.mean()),
            "mean_recall": float(R.mean()),
            "mean_f1": float(F.mean()),
            "num_total": len(predictions),
        }

    def score_corpus(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Convenience method — returns only corpus-level F1."""
        result = self.score_batch(predictions, references)
        return result["mean_f1"]
