# P4 — Indic LLM Evaluation Framework

A benchmark and evaluation framework for assessing large language models on **Indic languages** — covering multiple scripts, diverse NLP tasks, hallucination detection, and automated red-teaming.

## Features

- **Multi-adapter support** — evaluate any model via OpenAI, Anthropic, HuggingFace, or Ollama adapters
- **Task registry** — declarative task configs for classification, QA, summarization, translation across Indic languages
- **Rich metrics** — Exact Match, ROUGE, BERTScore, LLM-as-judge, SelfCheckGPT (hallucination)
- **Hallucination detection** — SelfCheckGPT-style consistency checking across multiple samples
- **Red-teaming module** — automated adversarial prompt generation and safety evaluation
- **Clustering analysis** — groups model errors by linguistic pattern
- **Dashboard** — visual benchmark leaderboard and per-task breakdowns
- **REST API** — trigger runs, query results, and export reports programmatically

## Supported Model Adapters

| Adapter | Models |
|---|---|
| `OpenAIAdapter` | GPT-4o, GPT-4, GPT-3.5 |
| `AnthropicAdapter` | Claude 3/4 family |
| `HuggingFaceAdapter` | Any local HF model |
| `OllamaAdapter` | Any Ollama-served model |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env

# Run benchmark on a model
python -m eval.runner \
  --adapter openai \
  --model gpt-4o \
  --languages hi,ta,te,bn \
  --output results/
```

## Metrics

| Metric | Use case |
|---|---|
| Exact Match | Classification, short-answer QA |
| ROUGE-L | Summarization |
| BERTScore | Generation quality (multilingual) |
| LLM Judge | Open-ended generation, reasoning |
| SelfCheckGPT | Hallucination / factual consistency |

## Project Structure

```
├── eval/
│   ├── runner.py          # BenchmarkRunner — orchestrates task execution
│   ├── task_registry.py   # Task definitions and dependency ordering
│   ├── adapters/          # Model adapter implementations
│   ├── metrics/           # Scoring functions
│   └── tasks/             # Task configs and datasets per language
├── hallucination/         # SelfCheckGPT-style consistency evaluation
├── redteaming/            # Adversarial prompt generation and safety checks
├── clustering/            # Error pattern clustering
├── dashboard/             # Leaderboard and result visualization
├── api/                   # REST API for triggering runs and querying results
├── scripts/               # Data download and preprocessing
└── tests/
```

## Output

Each benchmark run produces a `BenchmarkReport` with per-task scores and an aggregated summary:

```json
{
  "model_name": "gpt-4o",
  "task_results": [
    { "task": "hi_qa", "language": "hi", "metric": "exact_match", "score": 0.82 },
    { "task": "ta_summarization", "language": "ta", "metric": "rouge_l", "score": 0.61 }
  ]
}
```
