"""
BenchmarkRunner — orchestrates evaluation of a model adapter across registered tasks.

Workflow
--------
1. Load tasks from the registry (optionally filter by language / category).
2. Run them in topological dependency order.
3. For each task: load the dataset, format prompts, call the adapter, score with
   the configured metric.
4. Aggregate results into a report dict.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_registry import Task, TaskRegistry

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_name: str
    language: str
    category: str
    metric: str
    score: float
    num_examples: int
    elapsed_sec: float
    errors: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    model_name: str
    timestamp: str
    task_results: List[TaskResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.task_results:
            return {}
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "num_tasks": len(self.task_results),
            "mean_score": sum(r.score for r in self.task_results) / len(self.task_results),
            "per_task": {r.task_name: round(r.score, 4) for r in self.task_results},
            "by_language": self._agg_by("language"),
            "by_category": self._agg_by("category"),
        }

    def _agg_by(self, attr: str) -> Dict[str, float]:
        groups: Dict[str, List[float]] = {}
        for r in self.task_results:
            key = getattr(r, attr)
            groups.setdefault(key, []).append(r.score)
        return {k: round(sum(v) / len(v), 4) for k, v in groups.items()}

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "tasks": [
                {
                    "name": r.task_name,
                    "language": r.language,
                    "category": r.category,
                    "metric": r.metric,
                    "score": r.score,
                    "num_examples": r.num_examples,
                    "elapsed_sec": r.elapsed_sec,
                    "errors": r.errors,
                }
                for r in self.task_results
            ],
        }
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(text, encoding="utf-8")
            logger.info("Report written to %s", path)
        return text


# ── Runner ────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs evaluation tasks against a model adapter.

    Parameters
    ----------
    adapter : ModelAdapter
        Any adapter implementing complete() and complete_n().
    language_filter : str, optional
        If given, only tasks matching this language code are run.
    category_filter : str, optional
        If given, only tasks matching this category are run.
    max_examples : int, optional
        Override the per-task max_examples limit globally.
    output_path : str, optional
        If given, the final JSON report is written to this path.
    """

    def __init__(
        self,
        adapter,
        language_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        max_examples: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self.adapter = adapter
        self.language_filter = language_filter
        self.category_filter = category_filter
        self.max_examples = max_examples
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Run all matching tasks in dependency order and return a report."""
        import datetime

        tasks = self._select_tasks()
        if not tasks:
            logger.warning("No tasks matched the current filters.")

        report = BenchmarkReport(
            model_name=self.adapter.model_name,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

        for task in tasks:
            logger.info("Running task: %s [lang=%s]", task.name, task.language)
            result = self._run_task(task)
            report.task_results.append(result)
            logger.info(
                "  %s → %s = %.4f  (%d examples, %.1f s, %d errors)",
                task.name, task.metric, result.score,
                result.num_examples, result.elapsed_sec, result.errors,
            )

        if self.output_path:
            report.to_json(self.output_path)

        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_tasks(self) -> List[Task]:
        order = TaskRegistry.topological_order()
        all_tasks = TaskRegistry.all_tasks()
        tasks = [all_tasks[name] for name in order if name in all_tasks]

        if self.language_filter:
            tasks = [t for t in tasks if t.language == self.language_filter]
        if self.category_filter:
            tasks = [t for t in tasks if t.category == self.category_filter]
        return tasks

    def _run_task(self, task: Task) -> TaskResult:
        t0 = time.perf_counter()
        examples = self._load_dataset(task)
        limit = self.max_examples or task.max_examples
        if limit:
            examples = examples[:limit]

        evaluator = self._build_evaluator(task)
        predictions, references, questions = [], [], []
        errors = 0

        for ex in examples:
            prompt = self._format_prompt(task, ex)
            try:
                pred = self.adapter.complete_cached(
                    prompt, temperature=0.0, max_tokens=512
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Adapter error on task %s: %s", task.name, exc)
                pred = ""
                errors += 1

            predictions.append(pred)
            references.append(str(ex.get("answer", ex.get("reference", ""))))
            questions.append(str(ex.get("question", ex.get("input", ""))))

        score, details = self._score(
            task, evaluator, questions, predictions, references
        )
        elapsed = time.perf_counter() - t0

        return TaskResult(
            task_name=task.name,
            language=task.language,
            category=task.category,
            metric=task.metric,
            score=score,
            num_examples=len(examples),
            elapsed_sec=elapsed,
            errors=errors,
            details=details,
        )

    def _load_dataset(self, task: Task) -> List[Dict[str, Any]]:
        """Load dataset from the path specified in task config."""
        path = Path(task.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {task.dataset_path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else data.get("data", data.get("examples", []))

        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]

        raise ValueError(f"Unsupported dataset format: {suffix}. Use .json or .jsonl")

    def _format_prompt(self, task: Task, example: Dict[str, Any]) -> str:
        """Fill the task's prompt template with few-shot examples + the current example."""
        few_shot_block = ""
        for ex in task.few_shot_examples:
            q = ex.get("question", ex.get("input", ""))
            a = ex.get("answer", ex.get("output", ""))
            few_shot_block += f"Q: {q}\nA: {a}\n\n"

        question = example.get("question", example.get("input", ""))
        try:
            prompt = task.prompt_template.format(
                question=question,
                few_shot_examples=few_shot_block,
                **{k: v for k, v in example.items() if k not in ("question", "answer")},
            )
        except KeyError:
            # Fall back: simple question-answer format
            prompt = f"{few_shot_block}Q: {question}\nA:"
        return prompt

    def _build_evaluator(self, task: Task):
        """Instantiate the metric evaluator specified in task.evaluator."""
        evaluator_name = task.evaluator.lower()

        if evaluator_name == "exact_match":
            from .metrics.exact_match import ExactMatchMetric
            return ExactMatchMetric()

        if evaluator_name == "rouge":
            from .metrics.rouge import RougeMetric
            return RougeMetric()

        if evaluator_name == "bertscore":
            from .metrics.bertscore import BertScoreMetric
            return BertScoreMetric(lang=task.language)

        if evaluator_name == "llm_judge":
            from .metrics.llm_judge import LLMJudge
            return LLMJudge(judge_adapter=self.adapter, language=task.language)

        if evaluator_name == "selfcheck":
            from .metrics.selfcheck import SelfCheckMetric
            return SelfCheckMetric(adapter=self.adapter)

        raise ValueError(f"Unknown evaluator: {task.evaluator!r}")

    def _score(
        self,
        task: Task,
        evaluator,
        questions: List[str],
        predictions: List[str],
        references: List[str],
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Dispatch to the correct evaluator API and return (aggregate_score, details)."""
        from .metrics.exact_match import ExactMatchMetric
        from .metrics.rouge import RougeMetric
        from .metrics.llm_judge import LLMJudge

        details: List[Dict[str, Any]] = []

        if isinstance(evaluator, ExactMatchMetric):
            result = evaluator.score_batch(predictions, references)
            return result["accuracy"], []

        if isinstance(evaluator, RougeMetric):
            result = evaluator.score_batch(predictions, references)
            metric_key = task.metric if task.metric in result else "rougeL"
            return result.get(metric_key, result.get("rougeL", {})).get("fmeasure", 0.0), []

        if isinstance(evaluator, LLMJudge):
            results = evaluator.judge_batch(questions, predictions, references)
            details = [
                {
                    "question": q,
                    "prediction": r.prediction,
                    "reference": r.reference,
                    "score": r.score,
                    "reasoning": r.reasoning,
                }
                for q, r in zip(questions, results)
            ]
            return evaluator.mean_score(results), details

        # Generic fallback: call score_batch if it exists
        if hasattr(evaluator, "score_batch"):
            batch = evaluator.score_batch(predictions, references)
            if isinstance(batch, dict):
                return batch.get("score", batch.get("f1", 0.0)), []
            return float(batch), []

        return 0.0, []
