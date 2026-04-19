FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── API target (REST interface for triggering runs) ────────────────────────────
FROM base AS api
EXPOSE 8003
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]

# ── CLI eval runner target ─────────────────────────────────────────────────────
FROM base AS runner
# Usage: docker run ... runner python -m eval.runner --adapter openai --model gpt-4o
CMD ["python", "-m", "eval.runner", "--help"]
