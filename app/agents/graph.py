"""LangGraph orchestration: router -> extractor | retriever -> answerer | compliance."""
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.answerer import answer
from app.agents.compliance import check
from app.agents.extractor import extract
from app.llm import chat_json
from app.retrieval import Retriever, default_embedder
from app.schemas import Answer, Chunk, ComplianceReport, LeaseExtraction, RoutePlan

EXTRACT_WORDS = ("extrais", "extraire", "extract", "résume", "résumé", "summary", "summarize", "fiche")
CHECK_WORDS = (
    "conforme", "conformité", "légal", "illégal", "abusi", "risque", "vérifie le bail",
    "compliance", "compliant", "legal", "illegal", "unfair", "check the lease", "red flag",
)


class State(TypedDict, total=False):
    request: str
    chunks: list[Chunk]
    retriever: Any  # optional prebuilt Retriever (the API caches one per document)
    route: Literal["extract", "ask", "check"]
    keywords: str
    retrieved: list[Chunk]
    extraction: Optional[LeaseExtraction]
    answer: Optional[Answer]
    report: Optional[ComplianceReport]


def keyword_route(text: str) -> Literal["extract", "ask", "check"]:
    """Keyword rule, kept as the fallback when the LLM router is unavailable."""
    text = text.lower()
    if any(w in text for w in CHECK_WORDS):
        return "check"
    return "extract" if any(w in text for w in EXTRACT_WORDS) else "ask"


ROUTER_SYSTEM = (
    "Tu es le routeur d'un assistant pour contrats de location français.\n"
    "route = 'extract' si l'utilisateur demande la fiche complète / un résumé / l'extraction "
    "de toutes les informations clés du contrat ; 'check' s'il demande si le bail est "
    "conforme à la loi, s'il contient des clauses abusives, illégales ou risquées (le bail "
    "entier) ; 'ask' pour toute question précise (même sur un seul champ, comme le loyer, "
    "ou sur la légalité d'une seule clause).\n"
    "keywords = 4 à 8 mots-clés juridiques en français pour un moteur de recherche par "
    "mots-clés dans le contrat (traduis si la question est en anglais ; ex : menues "
    "réparations, entretien courant, sous-location, préavis, dépôt de garantie)."
)


def plan_request(request: str, strict: bool = False) -> RoutePlan:
    """One LLM call decides the route AND rewrites the query, so routing adds no latency.
    strict=True raises instead of falling back (evals must not score the fallback rule
    as if it were the LLM router)."""
    try:
        return chat_json(ROUTER_SYSTEM, request, RoutePlan, retries=1)
    except Exception:  # never let routing break the request
        if strict:
            raise
        return RoutePlan(route=keyword_route(request), keywords="")


def router(state: State) -> State:
    plan = plan_request(state["request"])
    return {"route": plan.route, "keywords": plan.keywords[:300]}


def extract_node(state: State) -> State:
    return {"extraction": extract(state["chunks"])}


def retrieve_node(state: State) -> State:
    retriever = state.get("retriever") or Retriever(state["chunks"], default_embedder())
    return {"retrieved": retriever.search(state["request"], keywords=state.get("keywords", ""))}


def check_node(state: State) -> State:
    return {"report": check(state["chunks"])}


def answer_node(state: State) -> State:
    return {"answer": answer(state["request"], state["retrieved"])}


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("extract", extract_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("check", check_node)
    g.set_entry_point("router")
    g.add_conditional_edges(
        "router", lambda s: s["route"], {"extract": "extract", "ask": "retrieve", "check": "check"}
    )
    g.add_edge("retrieve", "answer")
    g.add_edge("extract", END)
    g.add_edge("answer", END)
    g.add_edge("check", END)
    return g.compile()
