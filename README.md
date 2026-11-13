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

Multi-agent assistant for French residential leases. It extracts key clauses, flags illegal, abusive or missing ones with the law article, and answers questions with exact citations. Every answer is measured by an evaluation suite.

> Status: week 3. Compliance agent, MCP server, Langfuse tracing, CI evals, deploy-ready container.
> Not legal advice: a lawyer should confirm every flag.

## What it does

- **Extract**: parties, rent, deposit, duration, notice periods, indexation, surface... as validated JSON.
- **Ask**: questions in French or English, every claim linked to a passage of the contract.
- **Check**: compliance report against loi n° 89-462 (deposit cap, minimum duration, notice periods,
  IRL indexation, mandatory mentions, clauses deemed unwritten), each flag with its legal basis.
- **Measure**: eval suite with hand-labelled gold answers, LLM judge, compliance precision/recall;
  CI fails the build if quality drops.

## Architecture

```
PDF -> ingest (sentence-aware chunks with page numbers)
    -> index once per document: BM25 (accent-insensitive) + multilingual embeddings
    -> LangGraph: router (1 LLM call: route + French keywords)
         -> extractor                         (fiche du contrat, validated JSON)
         -> hybrid retriever (RRF) -> answerer (citations checked against the chunks)
         -> compliance: deterministic rules on the extraction + LLM clause scan (quotes verified)
    -> FastAPI (/upload, /extract, /ask, /check)      -> Langfuse traces (optional)
    -> MCP server (load/extract/ask/check tools)      -> Claude Desktop, IDEs...
    -> Streamlit UI (workspace, risk report, quality)
```

Design choices:

- **Hybrid retrieval.** BM25 catches exact legal terms and numbers ("IRL", "750"); multilingual
  embeddings (`paraphrase-multilingual-MiniLM-L12-v2` via fastembed, CPU/ONNX, no torch) catch
  paraphrases and English questions on French contracts. Rankings are merged with Reciprocal
  Rank Fusion, so no score calibration is needed. Set `EMBEDDING_MODEL=` to fall back to BM25.
- **Router does double duty.** One LLM call returns the route *and* French search keywords
  for BM25, so routing adds no extra latency. If the call fails, a keyword rule takes over.
- **Citations are verified, not trusted.** Citations to chunks that were not retrieved are
  dropped, page numbers come from the chunk, and each quote is fuzzy-matched against the chunk
  text (`verified: true/false`, shown in the UI).
- **Explicit abstention.** The answerer returns `found: false` when the passages do not
  contain the answer; the eval measures it instead of string-matching "je ne trouve pas".
- **Compliance: rules where the law is a number, LLM only where it is wording.** Deposit,
  duration, notice periods and indexation are checked by code on the extracted fields, so they
  are exact and testable. The LLM only scans for clauses deemed unwritten (art. 4: imposed direct
  debit, penalties, pet bans, visits on public holidays...) and must pick from a fixed
  checklist: the server attaches the law reference itself and drops any finding whose quote is
  not found in the contract. No flag can cite an invented article or an invented clause.

## Run it

```bash
cp .env.example .env          # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
pip install -r requirements.txt
uvicorn app.main:app --reload # API on http://localhost:8000/docs
streamlit run ui/streamlit_app.py
```

Or with Docker: `docker compose up --build`.

Any OpenAI-compatible endpoint works: Ollama locally (`http://localhost:11434/v1`), Groq, Mistral, OpenAI.
If the first embedding-model download stalls, set `HF_HUB_DISABLE_XET=1`.

### MCP server

Expose the agents as tools to any MCP client (`load_contract`, `extract_contract`,
`ask_contract`, `check_contract`):

```bash
python -m app.mcp_server                      # stdio
python -m app.mcp_server --http --port 8001   # streamable HTTP
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"lexagent": {
  "command": "C:/path/to/lexagent/.venv/Scripts/python.exe",
  "args": ["-m", "app.mcp_server"],
  "cwd": "C:/path/to/lexagent"}}}
```

### Tracing

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST` if self-hosted). Each API
request becomes a trace (`ask`, `extract`, `check`) with every LLM call nested as a generation
(prompt, output, tokens, latency). Without keys, tracing is off and costs nothing.

## Evaluation

```bash
python evals/run_evals.py --contract data/contracts/bail_exemple.pdf --gold evals/gold/bail_exemple.yaml --judge
python evals/run_evals.py --contract data/contracts/bail_abusif.pdf  --gold evals/gold/bail_abusif.yaml  --judge
python evals/run_evals.py ... --no-embeddings          # BM25-only baseline
python evals/run_evals.py ... --fail-under answer_accuracy=0.9   # exit 1 below threshold (CI)
```

Test contracts (fictional):

- `bail_exemple.pdf`: lawful lease, 3 pages. The compliance agent must raise **no** flag.
- `bail_abusif.pdf`: 5 pages, 14 planted problems (deposit of 2 months, 1-year unfurnished
  lease, 6-month tenant notice, ICC indexation, no surface, imposed direct debit, late
  penalties, receipt fees, imposed insurer, pet ban, guest ban, visits on public holidays,
  termination for breaking building rules...) mixed with lawful clauses that must not be
  flagged. Regenerate with `python data/make_test_contracts.py`.

Eval set: 50 questions (`evals/questions.yaml`) with gold answers per contract, 22 routing cases
(`evals/router.yaml`), and the expected compliance flags in each gold file. Questions go through
the same path as the API. The runner stops after 3 errors in a row (quota exhausted, endpoint
down) instead of scoring failures, and evaluates the router in strict mode so a silent
keyword fallback cannot pass for the LLM router.

Week 2 results on `bail_exemple.pdf`, LLM `openai/gpt-oss-120b` via Groq, top_k=5 (before the
week 3 chunker change):

| Metric | BM25 only | Hybrid (BM25 + embeddings) |
| --- | --- | --- |
| Extraction field accuracy (7 fields) | 100% | 100% |
| Answer accuracy, LLM judge (43 answerable) | 95.3% | 95.3% |
| Abstention on hallucination traps (7) | 100% | 100% |
| False abstention on answerable | 0% | 0% |
| Router accuracy (16 cases + 50 questions) | 100% | 100% |
| Retrieval recall@5 | 100% | 100% |
| Retrieval hit@1 | **81.4%** | 74.4% |
| Citation on the gold page | 100% | 100% |
| Quotes verified in the cited chunk | 100% | 100% |
| Median / p95 answer latency | 12.1 s / 20.7 s | 11.6 s / 15.0 s |

What this says:

- **The 3-page contract is too small to separate retrievers.** With 6 chunks and top_k=5 the
  answerer almost always sees the right passage. Hit@1 is the only discriminating metric there,
  and the small MiniLM model *lowers* it. `bail_abusif.pdf` (longer, 9 chunks) was added to
  measure this properly.
- **Remaining failures are judge strictness, not wrong answers** (q28 omits a secondary
  condition; q34 paraphrases "commandement de payer infructueux").
- **Latency** is two LLM calls in series (router, then answer) on a reasoning model.

Week 3 results (2026-11-13, Groq free tier, hybrid retrieval, top_k=5, **LLM judge off** to fit
the free quota: an answer passes when it is found and cites the gold page, so answer accuracy
is not measured here):

| Metric | `bail_exemple.pdf` (lawful, 3 pages, 5 chunks) | `bail_abusif.pdf` (14 planted problems, 5 pages, 9 chunks) |
| --- | --- | --- |
| Model | `gpt-oss-120b` | Q&A: `gpt-oss-20b`; extraction + compliance: `gpt-oss-120b` |
| Extraction field accuracy (10 fields) | 100% | 100% |
| Compliance precision / recall | 100% / 100% (0 flags expected, 0 raised) | 100% / 100% (14 of 14, no false flag) |
| Questions passed (judge off) | 49 / 50 | 34 / 36 |
| Abstention on hallucination traps | 100% (7) | 100% (7; the 8th, q23, hit a connection error) |
| False abstention on answerable | 0% | 3.6% (1 of 28) |
| Router accuracy (22 cases + questions) | 100% | 100% |
| Retrieval recall@5 / hit@1 | 100% / 73.8% | 96.4% / 71.4% (BM25 only: 67.9%) |
| Citation on the gold page | 100% | 96.4% |
| Quotes verified in the cited chunk | 100% | 100% |
| Median / p95 answer latency | 12.2 s / 20.0 s | 14.6 s / 20.0 s |

What this says:

- **Compliance works on both sides:** every planted problem is flagged with its article, and
  neither the lawful lease nor the lawful clauses of the abusive one get a false flag.
- **Two of the three non-passes were network errors, not wrong answers** (a request timeout on
  q43, a connection error on q23). Groq was slow that afternoon: the first `bail_abusif` run
  also lost its extraction and compliance steps to timeouts, so those two steps were rerun alone
  on `gpt-oss-120b` (`evals/results_120b_abusif_compliance.json`).
- **The one real failure is a retrieval miss on the longer contract.** q06 asks for "l'adresse
  du logement", but the contract never uses the word "adresse" ("situé au 17 rue Paul Bert"):
  BM25 ranks the right chunk 5th because "logement" is on every page, the MiniLM embeddings miss
  it, and rank fusion pushes it out of the top 5. The answerer then correctly says it cannot find
  it. Fix candidates: a stronger embedding model, or top_k 6-8 on long contracts.
- **Hybrid vs BM25 is still a wash.** On the longer contract hybrid gets hit@1 71.4% vs 67.9%
  for BM25 alone, a single question out of 28. A real comparison needs a much longer lease.

### Cost and CI

The `evals` CI job runs on pushes to `main` and on manual dispatch, with thresholds on answer
accuracy, abstention, extraction, routing and compliance precision/recall; results are uploaded
as a build artifact. It needs the `LLM_API_KEY` repo secret (optional repo variables
`LLM_BASE_URL`, `LLM_MODEL`) and skips with a warning without it.

Measured cost: one judged run on one contract (extraction, compliance, 22 routing cases,
~50 questions x 3 LLM calls) needs **more than the 200k tokens/day** of Groq's free tier for a
reasoning model. On the free tier, run one contract per day with a fresh model quota, skip the
judge, or use `--skip-questions` for compliance only; for CI, use a paid tier. Per-minute rate
limits (8k tokens/min on the free tier) are handled automatically: the client waits and retries.
A daily-quota error stops the run, and the partial results are still written.

## Deploy (demo)

The repo deploys as-is to a **Hugging Face Space** (Docker SDK, free CPU): the header at the top
of this README is the Space config, and the root `Dockerfile` runs the API and the UI in one
container. Only the UI is public; the API listens on localhost inside the container.

1. Create a Space on huggingface.co: SDK **Docker**, blank template, visibility of your choice.
2. In the Space **Settings -> Variables and secrets**, add the secrets:
   - `LLM_API_KEY`: your LLM key (Groq by default; set `LLM_BASE_URL` / `LLM_MODEL` to change)
   - `DEMO_PASSWORD`: recommended for a public Space, so strangers cannot use up your LLM quota
3. Push this repo to the Space (`git push` to `https://huggingface.co/spaces/<user>/<space>`,
   with a Hugging Face access token as the password). The first build takes ~5-10 minutes.

Demo limits: contracts live in memory only (lost when the Space restarts or sleeps after
inactivity), uploads are capped at 10 MB, and every visitor shares the same LLM quota.
It is not built for real users' data: no accounts, no storage, and contracts are sent to a US LLM provider.

Other hosts (Render, Railway, Fly.io...): build the root `Dockerfile` and route traffic to `$PORT`.
Locally: `docker compose up --build` (API on :8000, UI on :8501).

## Tests

```bash
pytest -q   # 64 tests, no LLM and no model download needed
```

## Roadmap

- [x] Week 1: skeleton, schemas, ingest, extractor, eval runner
- [x] Week 2: hybrid retrieval (BM25 + embeddings), Q&A with verified citations, LLM router, 50 eval questions, LLM-as-judge
- [x] Week 3: compliance agent, MCP server, Langfuse tracing, CI evals, deploy-ready container, risk report and quality pages
- [ ] Next: deploy the demo, retrieval fix for long contracts (q06), judged runs on a paid tier, persistent storage (pgvector), OCR for scanned leases, more test contracts (furnished, bail mobilité)
