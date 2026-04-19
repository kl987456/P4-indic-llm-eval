# P4 — Indic LLM Evaluation Framework

Benchmark and evaluation framework for measuring large language model performance on **Indic languages** — supports any model provider, multiple NLP task types, hallucination detection, and automated red-teaming.

## Architecture

```
Benchmark config (YAML)
  └─► TaskRegistry         loads + dependency-orders tasks
        └─► BenchmarkRunner  iterates tasks, calls adapter, scores output
              ├─► ModelAdapter   OpenAI / Anthropic / HuggingFace / Ollama
              └─► Metric         ExactMatch / ROUGE-L / BERTScore / LLMJudge / SelfCheckGPT

  ├─► hallucination/        SelfCheckGPT consistency check across N samples
  ├─► redteaming/           adversarial prompt generation + safety evaluation
  └─► clustering/           groups model errors by linguistic pattern
```

## Key features

- **4 model adapters**: OpenAI (GPT-4o+), Anthropic (Claude 3/4), HuggingFace (any local model), Ollama (self-hosted)
- **5 metrics**: Exact Match, ROUGE-L, BERTScore (language-specific models per Indic script), LLM-as-judge, SelfCheckGPT
- **Hallucination detection**: runs N independent samples and checks consistency with SelfCheckGPT
- **Red-teaming**: generates adversarial prompts and evaluates safety refusals
- **Task registry**: YAML task configs with dependency ordering via `graphlib.TopologicalSorter`
- **REST API**: trigger benchmark runs, query results, export reports
- **Dashboard**: leaderboard and per-task score visualisation

## Supported languages

Hindi (hi), Telugu (te), Tamil (ta), Kannada (kn), Malayalam (ml), Bengali (bn), Gujarati (gu), Marathi (mr), Punjabi (pa), Odia (or), Hinglish (code-switch), and more via task configs.

## Tech stack

| Layer | Package |
|---|---|
| API | fastapi 0.111, uvicorn |
| Database | sqlalchemy 2 |
| LLM providers | anthropic 0.28, openai 1.30, ollama 0.2 |
| Local models | transformers 4.41, torch 2.3, datasets, accelerate |
| ROUGE | rouge-score 0.1.2 |
| BERTScore | bert-score 0.3.13, evaluate 0.4.2 |
| Task config | pyyaml 6 |
| HTTP | requests 2.32, httpx 0.27 |

## Project files

```
P4-indic-llm-eval/
├── eval/
│   ├── runner.py                BenchmarkRunner — loads tasks, calls adapter, aggregates BenchmarkReport
│   ├── task_registry.py         Task dataclass, TaskRegistry, topological dependency ordering
│   ├── adapters/
│   │   ├── base.py              ModelAdapter ABC
│   │   ├── anthropic_adapter.py Anthropic Claude (claude-3-5-sonnet, claude-opus-4 …)
│   │   ├── openai_adapter.py    OpenAI GPT-4o, GPT-4, GPT-3.5
│   │   ├── hf_adapter.py        HuggingFace transformers (any local checkpoint)
│   │   └── ollama_adapter.py    Ollama REST API (self-hosted models)
│   ├── metrics/
│   │   ├── exact_match.py       Normalised exact match with Unicode handling
│   │   ├── rouge.py             ROUGE-L via rouge-score
│   │   ├── bertscore.py         BERTScore with per-language model selection
│   │   ├── llm_judge.py         LLM-as-judge scoring with structured rubric
│   │   └── selfcheck.py         SelfCheckGPT consistency-based hallucination score
│   └── tasks/
│       ├── configs/             YAML task configs (one file per task/language)
│       └── datasets/            Dataset loaders per task
├── hallucination/               SelfCheckGPT pipeline — sample N responses, score consistency
├── redteaming/                  Adversarial prompt generation + safety evaluation harness
├── clustering/                  Error pattern clustering by linguistic feature
├── dashboard/
│   ├── components/              Leaderboard table, per-task bar charts
│   └── pages/                   Dashboard pages
├── api/
│   ├── db/
│   │   ├── database.py          SQLAlchemy engine (SQLite default)
│   │   └── models.py            Run and TaskResult ORM models
│   ├── models/
│   │   └── schemas.py           Pydantic schemas
│   └── routes/                  REST routes for triggering runs + querying results
├── scripts/                     Dataset download, preprocessing, result export
├── tests/
│   └── test_metrics.py          Unit tests for all metric implementations
├── docker-compose.yml           api + runner services
├── Dockerfile                   Multi-stage: base / api / runner targets
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick start

```bash
# Install
pip install -r requirements.txt
cp .env.example .env   # add your API keys

# Run a benchmark from the CLI
python -m eval.runner \
  --adapter anthropic \
  --model claude-sonnet-4-6 \
  --languages hi,te,ta \
  --output ./eval_results/

# Or via Docker
docker compose up api
docker compose run --rm runner python -m eval.runner --adapter openai --model gpt-4o
```

## Benchmark output

```json
{
  "model_name": "claude-sonnet-4-6",
  "timestamp": "2026-04-19T10:00:00Z",
  "task_results": [
    { "task_name": "hi_qa", "language": "hi", "metric": "exact_match", "score": 0.83, "num_examples": 500 },
    { "task_name": "ta_summarization", "language": "ta", "metric": "rouge_l", "score": 0.61, "num_examples": 200 }
  ]
}
```

## Metrics

| Metric | Best for |
|---|---|
| Exact Match | Classification, short-answer QA |
| ROUGE-L | Summarisation |
| BERTScore | Open-ended generation (multilingual models selected per script) |
| LLM Judge | Reasoning, open-ended generation |
| SelfCheckGPT | Factual consistency / hallucination detection |

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required for AnthropicAdapter |
| `OPENAI_API_KEY` | Required for OpenAIAdapter |
| `HF_TOKEN` | Required for gated HuggingFace models |
| `OLLAMA_BASE_URL` | Ollama server URL (default: http://localhost:11434) |
| `EVAL_WORKERS` | Parallel evaluation threads |
| `EVAL_OUTPUT_DIR` | Where to write benchmark reports |
