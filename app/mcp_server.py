"""MCP server: exposes LexAgent's agents as tools to any MCP client (Claude Desktop, IDEs...).

    python -m app.mcp_server                      # stdio (for local clients)
    python -m app.mcp_server --http --port 8001   # streamable HTTP

Claude Desktop config (claude_desktop_config.json):

    {"mcpServers": {"lexagent": {
        "command": "<path to>/.venv/Scripts/python.exe",
        "args": ["-m", "app.mcp_server"],
        "cwd": "<path to>/lexagent"}}}
"""
import argparse
import uuid
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message is shown to the client

from app.agents.answerer import answer
from app.agents.compliance import check
from app.agents.extractor import extract
from app.agents.graph import plan_request
from app.ingest import ingest
from app.retrieval import Retriever, default_embedder
from app.schemas import Answer, ComplianceReport, LeaseExtraction

mcp = MCPServer(
    name="lexagent",
    instructions=(
        "Tools for French residential leases. Call load_contract with a local PDF path first, "
        "then use the returned doc_id with extract_contract, ask_contract or check_contract. "
        "Answers cite the contract (chunk id, page, verified quote). Not legal advice."
    ),
)
_DOCS: dict[str, dict] = {}


def _doc(doc_id: str) -> dict:
    if doc_id not in _DOCS:
        raise ToolError(f"Unknown doc_id {doc_id!r}: call load_contract first.")
    return _DOCS[doc_id]


@mcp.tool()
def load_contract(path: str) -> dict:
    """Load and index a contract PDF from a local path. Returns its doc_id."""
    pdf = Path(path).expanduser()
    if pdf.suffix.lower() != ".pdf" or not pdf.is_file():
        raise ToolError(f"Not a PDF file: {path}")
    chunks = ingest(pdf)
    if not chunks:
        raise ToolError("No text found in this PDF (scanned? OCR is not supported yet).")
    retriever = Retriever(chunks, default_embedder())
    doc_id = uuid.uuid4().hex[:8]
    _DOCS[doc_id] = {"name": pdf.name, "chunks": chunks, "retriever": retriever, "extraction": None}
    return {"doc_id": doc_id, "file": pdf.name, "pages": max(c.page for c in chunks), "chunks": len(chunks)}


@mcp.tool()
def extract_contract(doc_id: str) -> LeaseExtraction:
    """Extract the key fields of a lease: parties, rent, deposit, duration, notice periods..."""
    doc = _doc(doc_id)
    doc["extraction"] = extract(doc["chunks"])
    return doc["extraction"]


@mcp.tool()
def ask_contract(doc_id: str, question: str) -> Answer:
    """Answer a question (French or English) about the contract, with cited passages."""
    doc = _doc(doc_id)
    keywords = plan_request(question).keywords
    return answer(question, doc["retriever"].search(question, keywords=keywords))


@mcp.tool()
def check_contract(doc_id: str) -> ComplianceReport:
    """Flag illegal, abusive or missing clauses, each with the French law article."""
    doc = _doc(doc_id)
    report = check(doc["chunks"], doc["extraction"])
    doc["extraction"] = report.extraction
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    if args.http:
        mcp.run("streamable-http", port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
