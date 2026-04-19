"""
Tests for P4 evaluation metrics and adapters:
  - ExactMatchMetric
  - RougeMetric
  - LLMJudge
  - TaskRegistry
  - BenchmarkRunner
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── ExactMatchMetric ───────────────────────────────────────────────────────────

from eval.metrics.exact_match import ExactMatchMetric


def test_exact_match_identical():
    m = ExactMatchMetric()
    assert m.score("दिल्ली", "दिल्ली") == 1.0


def test_exact_match_different():
    m = ExactMatchMetric()
    assert m.score("मुम्बई", "दिल्ली") == 0.0


def test_exact_match_case_insensitive():
    m = ExactMatchMetric()
    assert m.score("Delhi", "delhi") == 1.0


def test_exact_match_punct_ignored():
    m = ExactMatchMetric()
    assert m.score("Delhi!", "Delhi") == 1.0


def test_exact_match_whitespace_normalised():
    m = ExactMatchMetric()
    assert m.score("  Delhi  ", "Delhi") == 1.0


def test_exact_match_devanagari_danda_stripped():
    m = ExactMatchMetric()
    assert m.score("दिल्ली।", "दिल्ली") == 1.0


def test_exact_match_batch():
    m = ExactMatchMetric()
    result = m.score_batch(["Delhi", "Mumbai"], ["Delhi", "Mumbai"])
    assert result["accuracy"] == 1.0
    assert result["num_correct"] == 2


def test_exact_match_batch_partial():
    m = ExactMatchMetric()
    result = m.score_batch(["Delhi", "wrong"], ["Delhi", "Mumbai"])
    assert result["accuracy"] == 0.5
    assert result["num_correct"] == 1
    assert result["num_total"] == 2


def test_exact_match_batch_length_mismatch():
    m = ExactMatchMetric()
    with pytest.raises(ValueError):
        m.score_batch(["a", "b"], ["a"])


def test_exact_match_top_k():
    m = ExactMatchMetric()
    predictions = [["Delhi", "Mumbai", "Chennai"], ["Paris", "London", "Berlin"]]
    references = ["Mumbai", "London"]
    score = m.top_k_score(predictions, references, k=3)
    assert score == 1.0


def test_exact_match_top_k_miss():
    m = ExactMatchMetric()
    predictions = [["Wrong1", "Wrong2"]]
    references = ["Delhi"]
    assert m.top_k_score(predictions, references, k=2) == 0.0


# ── RougeMetric ────────────────────────────────────────────────────────────────

from eval.metrics.rouge import RougeMetric


def test_rouge_perfect_match():
    m = RougeMetric()
    result = m.score("भारत की राजधानी दिल्ली है", "भारत की राजधानी दिल्ली है")
    assert result["rouge1"]["fmeasure"] > 0.99


def test_rouge_partial_overlap():
    m = RougeMetric()
    result = m.score("भारत की राजधानी दिल्ली है", "दिल्ली भारत की राजधानी है")
    assert 0.0 < result["rouge1"]["fmeasure"] <= 1.0


def test_rouge_no_overlap():
    m = RougeMetric()
    result = m.score("hello world", "xyz abc")
    assert result["rouge1"]["fmeasure"] == 0.0


def test_rouge_returns_all_metrics():
    m = RougeMetric()
    result = m.score("The cat sat on the mat", "The cat sat on the mat")
    for key in ["rouge1", "rouge2", "rougeL"]:
        assert key in result
        assert "precision" in result[key]
        assert "recall" in result[key]
        assert "fmeasure" in result[key]


def test_rouge_batch():
    m = RougeMetric()
    preds = ["The capital of India is Delhi", "Mumbai is a city"]
    refs = ["The capital of India is Delhi", "Mumbai is a city"]
    result = m.score_batch(preds, refs)
    assert result["rouge1"]["fmeasure"] > 0.95


def test_rouge_corpus_f1():
    m = RougeMetric()
    preds = ["hello world"] * 5
    refs = ["hello world"] * 5
    f1 = m.score_corpus_f1(preds, refs)
    assert f1 > 0.95


def test_rouge_hindi_text():
    m = RougeMetric()
    result = m.score("भारत एक देश है", "भारत एक देश है")
    assert result["rouge1"]["fmeasure"] > 0.9


# ── LLMJudge ──────────────────────────────────────────────────────────────────

from eval.metrics.llm_judge import LLMJudge, JudgeResult


def _make_judge(response_text: str) -> LLMJudge:
    mock_adapter = MagicMock()
    mock_adapter.complete.return_value = response_text
    return LLMJudge(judge_adapter=mock_adapter)


def test_judge_parse_json_response():
    judge = _make_judge('{"score": 8, "reasoning": "Good answer"}')
    result = judge.judge_single("Q?", "pred", "ref")
    assert result.raw_score == 8
    assert abs(result.score - (8 - 1) / 9) < 1e-6
    assert result.parse_error is False


def test_judge_parse_fallback_regex():
    judge = _make_judge('score: 7, this is a decent answer')
    result = judge.judge_single("Q?", "pred", "ref")
    assert result.raw_score == 7


def test_judge_clamps_score():
    judge = _make_judge('{"score": 15, "reasoning": "out of range"}')
    result = judge.judge_single("Q?", "pred", "ref")
    assert result.raw_score <= 10


def test_judge_handles_parse_error():
    judge = _make_judge("I cannot evaluate this")
    result = judge.judge_single("Q?", "pred", "ref")
    assert result.parse_error is True
    assert result.raw_score == 1  # fallback to min


def test_judge_batch():
    judge = _make_judge('{"score": 9, "reasoning": "Excellent"}')
    results = judge.judge_batch(["Q1", "Q2"], ["p1", "p2"], ["r1", "r2"])
    assert len(results) == 2
    assert all(isinstance(r, JudgeResult) for r in results)


def test_judge_batch_length_mismatch():
    judge = _make_judge('{"score": 5, "reasoning": "ok"}')
    with pytest.raises(ValueError):
        judge.judge_batch(["Q1"], ["p1", "p2"], ["r1", "r2"])


def test_judge_mean_score():
    judge = _make_judge('{"score": 10, "reasoning": "perfect"}')
    results = judge.judge_batch(["Q"] * 3, ["p"] * 3, ["r"] * 3)
    mean = judge.mean_score(results)
    assert abs(mean - 1.0) < 1e-6


def test_judge_mean_score_empty():
    judge = _make_judge("")
    assert judge.mean_score([]) == 0.0


def test_judge_hindi_prompt():
    judge = _make_judge('{"score": 7, "reasoning": "ठीक है"}')
    judge._prompt_template = judge._prompt_template   # stays as EN default
    result = judge.judge_single("भारत की राजधानी?", "दिल्ली", "दिल्ली")
    assert result.raw_score == 7


# ── TaskRegistry ──────────────────────────────────────────────────────────────

from eval.task_registry import Task, TaskRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    TaskRegistry.reset()
    yield
    TaskRegistry.reset()


def test_register_task():
    task = Task(name="hindi-qa", language="hi", category="qa",
                dataset_path="data.json", prompt_template="{question}",
                evaluator="exact_match", metric="accuracy")
    TaskRegistry.register(task)
    assert TaskRegistry.get("hindi-qa") is task


def test_register_duplicate_raises():
    task = Task(name="task1", language="en", category="qa",
                dataset_path="data.json", prompt_template="{question}",
                evaluator="exact_match", metric="accuracy")
    TaskRegistry.register(task)
    with pytest.raises(ValueError):
        TaskRegistry.register(task)


def test_get_nonexistent_raises():
    with pytest.raises(KeyError):
        TaskRegistry.get("nonexistent-task")


def test_register_from_yaml(tmp_path):
    config = {
        "name": "telugu-qa",
        "language": "te",
        "category": "qa",
        "dataset_path": "te_data.json",
        "prompt_template": "Q: {question}\nA:",
        "evaluator": "exact_match",
        "metric": "accuracy",
    }
    import yaml
    yaml_file = tmp_path / "telugu_qa.yaml"
    yaml_file.write_text(yaml.dump(config))
    task = TaskRegistry.register_from_yaml(str(yaml_file))
    assert task.name == "telugu-qa"
    assert task.language == "te"


def test_register_from_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        TaskRegistry.register_from_yaml("/nonexistent/config.yaml")


def test_topological_order_no_deps():
    for name in ["task-a", "task-b", "task-c"]:
        TaskRegistry.register(Task(name=name, language="en", category="qa",
                                   dataset_path="d.json", prompt_template="{question}",
                                   evaluator="exact_match", metric="accuracy"))
    order = TaskRegistry.topological_order()
    assert set(order) == {"task-a", "task-b", "task-c"}


def test_topological_order_with_deps():
    TaskRegistry.register(Task(name="base", language="en", category="qa",
                               dataset_path="d.json", prompt_template="{question}",
                               evaluator="exact_match", metric="accuracy"))
    TaskRegistry.register(Task(name="derived", language="en", category="qa",
                               dataset_path="d.json", prompt_template="{question}",
                               evaluator="exact_match", metric="accuracy",
                               dependencies=["base"]))
    order = TaskRegistry.topological_order()
    assert order.index("base") < order.index("derived")


def test_filter_by_language():
    for i, lang in enumerate(["hi", "hi", "te"]):
        TaskRegistry.register(Task(name=f"task-flang-{i}", language=lang,
                                   category="qa", dataset_path="d.json",
                                   prompt_template="{question}", evaluator="exact_match",
                                   metric="accuracy"))
    hi_tasks = TaskRegistry.filter_by_language("hi")
    assert len([t for t in hi_tasks if t.name.startswith("task-flang-") and t.language == "hi"]) == 2


# ── BenchmarkRunner ───────────────────────────────────────────────────────────

from eval.runner import BenchmarkRunner


def test_runner_empty_registry(reset_registry):
    adapter = MagicMock()
    adapter.model_name = "test-model"
    runner = BenchmarkRunner(adapter=adapter)
    report = runner.run()
    assert report.model_name == "test-model"
    assert report.task_results == []


def test_runner_runs_task(tmp_path, reset_registry):
    # Create dataset file
    dataset = [{"question": "What is 2+2?", "answer": "4"}]
    ds_path = tmp_path / "dataset.json"
    ds_path.write_text(json.dumps(dataset))

    TaskRegistry.register(Task(
        name="math-test", language="en", category="math",
        dataset_path=str(ds_path),
        prompt_template="Q: {question}\nA:",
        evaluator="exact_match", metric="accuracy",
    ))

    adapter = MagicMock()
    adapter.model_name = "gpt-test"
    adapter.complete_cached.return_value = "4"

    runner = BenchmarkRunner(adapter=adapter)
    report = runner.run()
    assert len(report.task_results) == 1
    assert report.task_results[0].task_name == "math-test"
    assert report.task_results[0].score == 1.0


def test_runner_report_summary(tmp_path, reset_registry):
    dataset = [{"question": "Q?", "answer": "A"}]
    ds_path = tmp_path / "data.json"
    ds_path.write_text(json.dumps(dataset))

    TaskRegistry.register(Task(
        name="t1", language="hi", category="qa",
        dataset_path=str(ds_path),
        prompt_template="{question}",
        evaluator="exact_match", metric="accuracy",
    ))

    adapter = MagicMock()
    adapter.model_name = "test"
    adapter.complete_cached.return_value = "A"

    report = BenchmarkRunner(adapter=adapter).run()
    summary = report.summary()
    assert "mean_score" in summary
    assert "by_language" in summary
    assert "hi" in summary["by_language"]
