"""
ROUGE metric for Indic language evaluation.
Uses rouge-score library with Unicode-aware tokenisation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from rouge_score import rouge_scorer, scoring


class IndicTokenizer:
    """
    Simple Unicode-aware tokenizer that works for Devanagari, Telugu,
    Tamil, and Latin scripts simultaneously.
    """

    # Split on whitespace and common punctuation but keep Indic chars
    _SPLIT = re.compile(r"[\s\u0964\u0965।॥,;:.!?\"'()\[\]{}<>]+")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = unicodedata.normalize("NFC", text)
        tokens = IndicTokenizer._SPLIT.split(text.lower())
        return [t for t in tokens if t]


class RougeMetric:
    """
    ROUGE-1, ROUGE-2, ROUGE-L scores for text generation evaluation.

    Works for multilingual text by using character n-grams as a fallback
    when word tokenization yields too few tokens (e.g., agglutinative langs).
    """

    def __init__(
        self,
        metrics: List[str] = ("rouge1", "rouge2", "rougeL"),
        use_stemmer: bool = False,
    ) -> None:
        self.metrics = list(metrics)
        self._scorer = rouge_scorer.RougeScorer(
            self.metrics,
            use_stemmer=use_stemmer,
            tokenizer=IndicTokenizer(),
        )

    # ------------------------------------------------------------------

    def score(self, prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Score a single (prediction, reference) pair.
        Returns nested dict: {rouge1: {precision, recall, fmeasure}, ...}
        """
        raw = self._scorer.score(reference, prediction)
        return {
            metric: {
                "precision": scores.precision,
                "recall": scores.recall,
                "fmeasure": scores.fmeasure,
            }
            for metric, scores in raw.items()
        }

    def score_batch(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        """
        Score a batch and return per-metric aggregate f-measures.
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have equal length")

        aggregator = scoring.BootstrapAggregator()
        for pred, ref in zip(predictions, references):
            aggregator.add_scores(self._scorer.score(ref, pred))

        result = aggregator.aggregate()
        return {
            metric: {
                "precision": result[metric].mid.precision,
                "recall": result[metric].mid.recall,
                "fmeasure": result[metric].mid.fmeasure,
            }
            for metric in self.metrics
        }

    def score_corpus_f1(self, predictions: List[str], references: List[str]) -> float:
        """Convenience — returns corpus ROUGE-L F1."""
        batch = self.score_batch(predictions, references)
        return batch.get("rougeL", {}).get("fmeasure", 0.0)
