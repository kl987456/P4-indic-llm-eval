"""
LLM-as-a-Judge evaluator.
Sends (question, reference, prediction) to a judge LLM and extracts a score.

Inspired by MT-Bench / Alpaca-Eval methodology.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

JUDGE_PROMPT_EN = """\
You are a strict and impartial judge evaluating the quality of an AI assistant's answer.

[Question]
{question}

[Reference Answer]
{reference}

[Assistant's Answer]
{prediction}

Evaluate the assistant's answer on a scale from 1 to 10 based on:
- Factual correctness (does it match the reference?)
- Completeness (does it cover all key points?)
- Fluency and clarity

Respond ONLY in this JSON format:
{{"score": <integer 1-10>, "reasoning": "<one sentence>"}}
"""

JUDGE_PROMPT_HI = """\
आप एक निष्पक्ष मूल्यांकनकर्ता हैं। नीचे दिए गए प्रश्न, संदर्भ उत्तर, और AI सहायक के उत्तर का मूल्यांकन करें।

[प्रश्न]
{question}

[संदर्भ उत्तर]
{reference}

[AI सहायक का उत्तर]
{prediction}

उत्तर को 1-10 के पैमाने पर रेटिंग दें। केवल JSON में उत्तर दें:
{{"score": <1-10>, "reasoning": "<एक वाक्य>"}}
"""

JUDGE_PROMPTS = {
    "en": JUDGE_PROMPT_EN,
    "hi": JUDGE_PROMPT_HI,
}


@dataclass
class JudgeResult:
    score: float          # Normalised 0-1
    raw_score: int        # Raw 1-10
    reasoning: str
    prediction: str
    reference: str
    parse_error: bool = False


class LLMJudge:
    """
    Uses a strong LLM (default: GPT-4o) to score predictions 1-10.
    Parses JSON response and normalises to [0, 1].
    """

    def __init__(
        self,
        judge_adapter,           # Any ModelAdapter instance
        language: str = "en",
        score_min: int = 1,
        score_max: int = 10,
    ) -> None:
        self.judge = judge_adapter
        self.language = language
        self.score_min = score_min
        self.score_max = score_max
        self._prompt_template = JUDGE_PROMPTS.get(language, JUDGE_PROMPT_EN)

    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> tuple[int, str]:
        """Extract score and reasoning from judge response."""
        # Try strict JSON first
        try:
            data = json.loads(text)
            return int(data["score"]), data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: regex extraction
        match = re.search(r'"score"\s*:\s*(\d+)', text)
        if match:
            score = int(match.group(1))
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
            reasoning = reason_match.group(1) if reason_match else ""
            return score, reasoning

        # Last resort: find any standalone integer
        nums = re.findall(r'\b([1-9]|10)\b', text)
        if nums:
            return int(nums[0]), text
        raise ValueError(f"Cannot parse judge response: {text[:200]}")

    def _normalise(self, raw_score: int) -> float:
        """Normalise raw_score from [min, max] to [0, 1]."""
        return (raw_score - self.score_min) / (self.score_max - self.score_min)

    # ------------------------------------------------------------------

    def judge_single(
        self,
        question: str,
        prediction: str,
        reference: str,
    ) -> JudgeResult:
        prompt = self._prompt_template.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )
        response = self.judge.complete(prompt, temperature=0.0, max_tokens=256)
        parse_error = False
        try:
            raw_score, reasoning = self._parse_response(response)
            # Clamp to valid range
            raw_score = max(self.score_min, min(self.score_max, raw_score))
        except ValueError:
            logger.warning("Failed to parse judge response: %s", response[:100])
            raw_score = self.score_min
            reasoning = response
            parse_error = True

        return JudgeResult(
            score=self._normalise(raw_score),
            raw_score=raw_score,
            reasoning=reasoning,
            prediction=prediction,
            reference=reference,
            parse_error=parse_error,
        )

    def judge_batch(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str],
    ) -> List[JudgeResult]:
        """Score a batch; returns list of JudgeResult in order."""
        if not (len(questions) == len(predictions) == len(references)):
            raise ValueError("All input lists must have the same length")

        results = []
        for q, pred, ref in zip(questions, predictions, references):
            results.append(self.judge_single(q, pred, ref))
        return results

    def mean_score(self, results: List[JudgeResult]) -> float:
        """Aggregate mean normalised score across a batch."""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
