---
title: LexAgent
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: French lease assistant with citations and compliance
---

# LexAgent

AI assistant for French residential leases: it extracts the key terms, answers questions with
cited passages, and flags illegal or missing clauses with the law article.

> Demo project, not legal advice.

## Features

- **Extract**: parties, rent, deposit, duration, notice periods... as validated JSON
- **Ask**: questions in French or English, each answer linked to a verified quote from the contract
- **Check**: compliance report against loi n° 89-462 (deposit cap, minimum duration, notice
  periods, rent indexation, mandatory mentions, abusive clauses)
- **Measure**: evaluation suite on fictional test leases, run in CI

## Stack

LangGraph · FastAPI · Streamlit · hybrid retrieval (BM25 + multilingual embeddings) ·
any OpenAI-compatible LLM (Groq by default) · MCP server · optional Langfuse tracing

## Run locally

```bash
cp .env.example .env              # add your LLM_API_KEY
pip install -r requirements.txt
uvicorn app.main:app              # API: http://localhost:8000/docs
streamlit run ui/streamlit_app.py # UI:  http://localhost:8501
```

Tests: `pytest -q`

## Deploy (Hugging Face Spaces)

Create a Docker Space, add the secrets `LLM_API_KEY` and `DEMO_PASSWORD` (optional, protects
your LLM quota), then push this repo to it. The root `Dockerfile` runs everything in one container.

## Results

On two fictional leases (`data/contracts/`): 100% of fields extracted correctly, all 14 planted
legal problems flagged with no false alarms, and 83 of 86 test questions passed (answer found
and cited from the right page). Details: `evals/`.
