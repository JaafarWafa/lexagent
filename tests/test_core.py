"""Fast tests that need no LLM and no embedding model: schemas, chunking, retrieval,
fusion, routing, citation checks, API."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.agents.answerer as answerer_mod
import app.agents.graph as g
from app.agents.answerer import answer, check_citations, quote_in_passage
from app.ingest import chunk_pages
from app.main import app
from app.retrieval import Retriever, rrf
from app.schemas import Answer, Chunk, Citation, LeaseExtraction, RoutePlan

PAGES = [
    (1, "Contrat de location. Le bailleur Mme Martin loue au locataire M. Alaoui un logement vide."),
    (2, "Le loyer mensuel est fixé à 750 euros. Le dépôt de garantie est de 750 euros."),
]
REPAIRS = (3, "Le locataire prend en charge l'entretien courant et les menues réparations.")


def test_chunks_keep_page_numbers():
    chunks = chunk_pages(PAGES, size=40, overlap=10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.id.startswith(f"p{c.page}-") for c in chunks)


def test_retriever_finds_deposit():
    r = Retriever(chunk_pages(PAGES))
    hits = r.search("dépôt de garantie")
    assert r.mode == "bm25"
    assert hits and hits[0].page == 2


def test_retriever_empty_document():
    assert Retriever([]).search("loyer") == []


# --- hybrid retrieval -------------------------------------------------------------


def test_rrf_rewards_agreement():
    # a chunk found by both retrievers beats chunks found by only one of them
    assert rrf([[0, 1], [1, 2]])[0] == 1
    assert rrf([[3, 1], [3]]) == [3, 1]


class FakeEmbedder:
    """Maps texts to concept vectors, like a multilingual model would: 'minor repairs'
    and 'menues réparations' land on the same axis."""

    CONCEPTS = [("repair", "réparation"), ("rent", "loyer"), ("landlord", "bailleur")]

    def embed(self, texts):
        v = np.array(
            [[float(any(w in t.lower() for w in c)) for c in self.CONCEPTS] + [0.01] for t in texts]
        )
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hybrid_finds_english_question_without_keywords():
    """BM25 alone cannot match 'minor repairs' to a French clause; embeddings can."""
    chunks = chunk_pages(PAGES + [REPAIRS])
    assert 3 not in [c.page for c in Retriever(chunks).search("Who pays for minor repairs?", k=1)]
    hybrid = Retriever(chunks, FakeEmbedder())
    assert hybrid.mode == "hybrid"
    assert hybrid.search("Who pays for minor repairs?", k=1)[0].page == 3


def test_keywords_feed_bm25():
    chunks = chunk_pages(PAGES + [REPAIRS])
    hits = Retriever(chunks).search("Who pays?", k=1, keywords="menues réparations entretien courant")
    assert hits[0].page == 3


# --- routing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,route",
    [("Extrais les clauses du bail", "extract"), ("Quel est le dépôt de garantie ?", "ask")],
)
def test_keyword_route(text, route):
    assert g.keyword_route(text) == route


def test_router_uses_llm_plan(monkeypatch):
    monkeypatch.setattr(g, "chat_json", lambda *a, **k: RoutePlan(route="extract", keywords="bail"))
    assert g.router({"request": "Donne-moi une vue d'ensemble"}) == {"route": "extract", "keywords": "bail"}


def test_router_falls_back_to_keywords(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(g, "chat_json", boom)
    assert g.router({"request": "Extrais les infos"}) == {"route": "extract", "keywords": ""}
    assert g.router({"request": "Qui paie ?"})["route"] == "ask"


def test_english_question_reaches_french_passage():
    """Cross-lingual retrieval: the router's French keywords let BM25 find page 3."""
    chunks = chunk_pages(PAGES + [REPAIRS])
    out = g.retrieve_node(
        {"request": "Who pays for minor repairs?", "chunks": chunks, "keywords": "menues réparations entretien courant"}
    )
    assert out["retrieved"] and out["retrieved"][0].page == 3


def test_retrieve_node_uses_cached_retriever():
    class Spy:
        def search(self, query, keywords=""):
            return ["hit"]

    assert g.retrieve_node({"request": "x", "chunks": [], "retriever": Spy()})["retrieved"] == ["hit"]


# --- answers and citations -------------------------------------------------------

CHUNK = Chunk(id="p3-c0", page=3, text="La sous-location est interdite sauf accord écrit du bailleur.")


@pytest.mark.parametrize(
    "quote,ok",
    [
        ("La sous-location est interdite", True),
        ("la sous location est interdite sauf accord ecrit", True),  # accents/punctuation lost
        ("La sous-location est autorisée sans accord", False),
        ("Le loyer est de 900 euros", False),
        ("", False),
    ],
)
def test_quote_in_passage(quote, ok):
    assert quote_in_passage(quote, CHUNK.text) is ok


def test_check_citations_drops_invented_ids_and_fixes_pages():
    cits = [
        Citation(chunk_id="p3-c0", page=1, quote="sous-location est interdite"),
        Citation(chunk_id="p9-c9", page=9, quote="invented"),
    ]
    out = check_citations(cits, [CHUNK])
    assert [(c.chunk_id, c.page, c.verified) for c in out] == [("p3-c0", 3, True)]


def test_answer_without_passages_abstains():
    a = answer("Can the tenant sublet?", [])
    assert a.found is False and a.citations == [] and "cannot find" in a.answer


def test_not_found_answer_has_no_citations(monkeypatch):
    fake = Answer(answer="Pas d'info.", found=False, citations=[Citation(chunk_id="p3-c0", page=3, quote="x")])
    monkeypatch.setattr(answerer_mod, "chat_json", lambda *a, **k: fake)
    assert answer("Parking ?", [CHUNK]).citations == []


def test_extraction_schema_rejects_bad_lease_type():
    with pytest.raises(Exception):
        LeaseExtraction(lease_type="château")
    assert LeaseExtraction(monthly_rent_eur=750).monthly_rent_eur == 750


# --- API --------------------------------------------------------------------------


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    r = TestClient(app).post("/upload", files={"file": ("bail.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_unknown_doc_is_404():
    assert TestClient(app).post("/ask", json={"doc_id": "nope", "question": "?"}).status_code == 404


# --- week 3 additions ---------------------------------------------------------------


def test_bm25_ignores_accents():
    chunks = chunk_pages(PAGES)
    assert Retriever(chunks).search("depot de garantie", k=1)[0].page == 2


def test_chunks_end_on_sentence_boundaries():
    text = "Premier article du bail. " * 30 + "Le dépôt de garantie est de 750 euros."
    chunks = chunk_pages([(1, text)], size=200, overlap=50)
    assert all(c.text.endswith(".") and len(c.text) <= 200 for c in chunks)
    assert any("dépôt de garantie est de 750 euros." in c.text for c in chunks)


def test_long_sentence_is_split_without_cutting_words():
    chunks = chunk_pages([(1, " ".join(["mot"] * 200))], size=100, overlap=0)
    assert all(len(c.text) <= 100 and set(c.text.split()) == {"mot"} for c in chunks)


@pytest.mark.parametrize("text", ["Ce bail est-il conforme ?", "Is this lease legal?", "Y a-t-il des clauses abusives ?"])
def test_keyword_route_check(text):
    assert g.keyword_route(text) == "check"


def test_check_endpoint_unknown_doc():
    assert TestClient(app).post("/check/nope").status_code == 404


def test_llm_quota_error_becomes_429(monkeypatch):
    import httpx
    import openai

    import app.main as main_mod

    resp = httpx.Response(429, headers={"retry-after": "600"}, request=httpx.Request("POST", "http://llm"))

    def quota(*a, **k):
        raise openai.RateLimitError("quota", response=resp, body=None)

    monkeypatch.setattr(main_mod, "check", quota)
    main_mod.DOCS["q"] = main_mod.Document(chunks=[], retriever=None)
    r = TestClient(app).post("/check/q")
    assert r.status_code == 429 and "Quota" in r.json()["detail"] and "11 min" in r.json()["detail"]
