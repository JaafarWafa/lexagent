"""FastAPI service. Docs at /docs."""
import tempfile
from contextlib import asynccontextmanager
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import openai
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.agents.compliance import check
from app.agents.extractor import extract
from app.agents.graph import build_graph
from app.ingest import ingest
from app.retrieval import Retriever, default_embedder
from app.schemas import AskRequest, Chunk, LeaseExtraction
from app.tracing import flush, observe

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    flush()  # send pending Langfuse traces before exit (no-op when tracing is off)


app = FastAPI(title="LexAgent", version="0.3.0", lifespan=lifespan)
graph = build_graph()


@dataclass
class Document:
    chunks: list[Chunk]
    retriever: Retriever  # built once at upload: embedding the contract is the slow part
    extraction: LeaseExtraction | None = None  # cached: /check reuses it


DOCS: dict[str, Document] = {}  # in-memory store; later -> Qdrant / pgvector


@app.exception_handler(openai.RateLimitError)
def _llm_rate_limited(_req: Request, exc: openai.RateLimitError):
    retry = exc.response.headers.get("retry-after") if exc.response is not None else None
    wait = f" Réessayez dans {int(float(retry)) // 60 + 1} min." if retry else ""
    return JSONResponse(
        status_code=429,
        content={"detail": f"Quota du modèle de langage atteint.{wait}"},
        headers={"Retry-After": retry} if retry else None,
    )


@app.exception_handler(openai.APIError)
def _llm_unavailable(_req: Request, exc: openai.APIError):
    return JSONResponse(status_code=502, content={"detail": f"Modèle de langage indisponible : {type(exc).__name__}."})


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(DOCS)}


def _index(data: bytes) -> list[Chunk]:
    # Windows cannot reopen a temp file that is still open, so write it in a temp folder first
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.pdf"
        path.write_bytes(data)
        return ingest(path)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "PDF only")
    data = await file.read()
    chunks = await run_in_threadpool(_index, data)
    if not chunks:
        raise HTTPException(422, "No text found (scanned PDF? OCR comes later)")
    retriever = await run_in_threadpool(Retriever, chunks, default_embedder())
    doc_id = uuid.uuid4().hex[:8]
    DOCS[doc_id] = Document(chunks, retriever)
    return {
        "doc_id": doc_id,
        "pages": max(c.page for c in chunks),
        "chunks": len(chunks),
        "retrieval": retriever.mode,
    }


def _get(doc_id: str) -> Document:
    if doc_id not in DOCS:
        raise HTTPException(404, "Unknown doc_id")
    return DOCS[doc_id]


@app.post("/extract/{doc_id}")
@observe("extract")
def extract_endpoint(doc_id: str):
    t0 = time.perf_counter()
    # Explicit request: no routing needed, call the extractor agent directly.
    doc = _get(doc_id)
    doc.extraction = extract(doc.chunks)
    return {"extraction": doc.extraction, "latency_s": round(time.perf_counter() - t0, 2)}


@app.post("/check/{doc_id}")
@observe("check")
def check_endpoint(doc_id: str):
    """Compliance report: illegal, abusive or missing clauses, with the law article."""
    t0 = time.perf_counter()
    doc = _get(doc_id)
    report = check(doc.chunks, doc.extraction)
    doc.extraction = report.extraction
    return {"report": report, "latency_s": round(time.perf_counter() - t0, 2)}


@app.post("/ask")
@observe("ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    doc = _get(req.doc_id)
    out = graph.invoke({"request": req.question, "chunks": doc.chunks, "retriever": doc.retriever})
    result = out.get("answer") or out.get("extraction") or out.get("report")
    return {"result": result, "route": out["route"], "latency_s": round(time.perf_counter() - t0, 2)}


@app.get("/chunks/{doc_id}")
def chunks(doc_id: str):
    return _get(doc_id).chunks
