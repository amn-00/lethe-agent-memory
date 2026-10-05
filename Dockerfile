# Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does:
#   docker build -t lethe . && docker run -p 7860:7860 -e GROQ_API_KEY=... lethe
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LETHE_DATA=/app/data \
    LETHE_RESULTS=/app/evals/results \
    LETHE_EMBED_CACHE=/app/.fastembed \
    LETHE_BUDGET=6 \
    LETHE_GRACE=2 \
    LETHE_DEMO_PER_CHAT=30 \
    LETHE_DEMO_PER_DAY=150

WORKDIR /app

# dependencies first, so code-only changes rebuild fast
COPY pyproject.toml ./
COPY lethe ./lethe
RUN pip install .

# download the embedding model at build time, not on every start
RUN python -c "from lethe.index import FastEmbedder; FastEmbedder()"

# eval results power the Results tab
COPY evals ./evals

# Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user && mkdir -p /app/data && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "lethe.api:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
