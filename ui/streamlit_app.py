"""LexAgent UI: Workspace (extraction, questions with citations), Risk report, Quality.

Two modes:
- API_URL set (docker-compose, deploy/start.sh): talks to the FastAPI service over HTTP.
- API_URL unset (Streamlit Community Cloud, plain `streamlit run`): runs the same FastAPI app
  in-process, so a single process serves everything.
"""
import hmac
import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import app` when Streamlit runs this file from ui/
EVALS = ROOT / "evals"
SEVERITY = {"high": ("🔴", "Élevé"), "medium": ("🟠", "Moyen"), "low": ("🟡", "Faible")}
KIND = {"illegal": "Clause illégale", "missing": "Mention manquante", "warning": "Avertissement"}


def _eur(v) -> str:
    return f"{v:,.0f} €".replace(",", " ") if float(v).is_integer() else f"{v:,.2f} €".replace(",", " ")


def _date(v: str) -> str:
    parts = str(v).split("-")
    return "/".join(reversed(parts)) if len(parts) == 3 else str(v)


def _parties(parties) -> str:
    return " · ".join(f"{p['role'].capitalize()} : {p['name']}" for p in parties)


FIELD_LABELS = {  # extraction field -> (label shown, formatter)
    "parties": ("Parties", _parties),
    "property_address": ("Adresse du logement", str),
    "lease_type": ("Type de bail", lambda v: {"vide": "vide (non meublé)"}.get(v, v)),
    "start_date": ("Prise d'effet", _date),
    "duration_months": ("Durée", lambda v: f"{v} mois"),
    "monthly_rent_eur": ("Loyer mensuel hors charges", _eur),
    "charges_eur": ("Provision sur charges", _eur),
    "deposit_eur": ("Dépôt de garantie", _eur),
    "notice_period_tenant_months": ("Préavis du locataire", lambda v: f"{v} mois"),
    "notice_period_landlord_months": ("Préavis du bailleur", lambda v: f"{v} mois"),
    "rent_indexation": ("Révision du loyer", str),
    "landlord_type": ("Type de bailleur", str),
    "surface_m2": ("Surface habitable", lambda v: f"{v:g} m²"),
}


def load_secrets_into_env():
    """Streamlit Cloud secrets -> environment variables, read by app.config (env wins)."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # no secrets file: local run, nothing to load
        pass


load_secrets_into_env()


class HttpApi:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path: str, **kw):
        return requests.get(self.base + path, **kw)

    def post(self, path: str, **kw):
        return requests.post(self.base + path, **kw)


@st.cache_resource(show_spinner="Démarrage du moteur...")
def in_process_api():
    """The FastAPI app itself, called without a network hop. Cached: one document store
    shared by all reruns and visitors of this process."""
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def api():
    url = os.getenv("API_URL")
    return HttpApi(url) if url else in_process_api()


def ok(r) -> bool:
    return 200 <= r.status_code < 300


def api_error(r) -> str:
    try:
        return r.json().get("detail") or r.text
    except ValueError:
        return r.text or f"Erreur {r.status_code}"


st.set_page_config(page_title="LexAgent", page_icon="⚖️", layout="wide")
st.title("LexAgent")
st.caption("Assistant multi-agents pour baux d'habitation français : extraction, questions avec citations, conformité.")


def require_password():
    """Demo gate: when DEMO_PASSWORD is set, nothing (and no LLM call) runs before it is entered."""
    expected = os.getenv("DEMO_PASSWORD", "")
    if not expected or st.session_state.get("authenticated"):
        return
    with st.form("login"):
        st.write("Démo privée : entrez le mot de passe pour continuer.")
        given = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Entrer", type="primary"):
            if hmac.compare_digest(given.encode(), expected.encode()):
                st.session_state.authenticated = True
                st.rerun()
            st.error("Mot de passe incorrect.")
    st.stop()


require_password()

with st.sidebar:
    st.header("Contrat")
    pdf = st.file_uploader("Déposez un bail (PDF)", type="pdf")
    if pdf and st.button("Analyser", type="primary"):
        with st.spinner("Lecture et indexation..."):
            r = api().post("/upload", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
        if ok(r):
            st.session_state.doc = r.json() | {"name": pdf.name}
            st.session_state.chunks = {c["id"]: c for c in api().get(f"/chunks/{r.json()['doc_id']}").json()}
            st.session_state.history = []
            st.session_state.extraction = None
            st.session_state.report = None
        else:
            st.error(api_error(r))
    if "doc" in st.session_state:
        d = st.session_state.doc
        st.success(f"{d['name']} : {d['pages']} pages, {d['chunks']} passages")
        st.caption(f"Recherche : {'hybride (BM25 + embeddings)' if d.get('retrieval') == 'hybrid' else 'BM25'}")
    st.divider()
    st.caption("Outil d'aide à la lecture, pas un conseil juridique.")


def show_passage(chunk_id: str | None):
    full = st.session_state.chunks.get(chunk_id) if chunk_id else None
    if full:
        st.caption(f"Passage {chunk_id}, page {full['page']} : {full['text']}")


def workspace():
    left, right = st.columns([1, 1], gap="large")
    doc_id = st.session_state.doc["doc_id"]
    with left:
        st.subheader("Fiche du contrat")
        if st.button("Extraire les clauses clés"):
            with st.spinner("Extraction..."):
                r = api().post(f"/extract/{doc_id}")
            st.session_state.extraction = r.json() if ok(r) else {"error": api_error(r)}
        ex = st.session_state.get("extraction")
        if ex and "extraction" in ex:
            fields = dict(ex["extraction"])
            evidence = fields.pop("evidence", {}) or {}
            for key, (label, fmt) in FIELD_LABELS.items():
                value = fields.get(key)
                if value in (None, [], ""):
                    st.markdown(f"**{label}** : <span style='color:gray'>non mentionné</span>", unsafe_allow_html=True)
                    continue
                chunk = st.session_state.chunks.get(evidence.get(key) or "")
                src = f"  \n<small>source : page {chunk['page']} ({chunk['id']})</small>" if chunk else ""
                st.markdown(f"**{label}** : {fmt(value)}{src}", unsafe_allow_html=True)
            st.caption(f"Latence : {ex['latency_s']} s")
        elif ex:
            st.error(ex["error"])

    with right:
        st.subheader("Questions")
        for turn in st.session_state.history:
            st.chat_message("user").write(turn["q"])
            with st.chat_message("assistant"):
                st.write(turn["a"]["answer"])
                if turn["a"].get("found") is False:
                    st.caption("Information absente du contrat.")
                for c in turn["a"].get("citations", []):
                    mark = "✓ citation vérifiée" if c.get("verified") else "⚠ citation non retrouvée mot pour mot"
                    with st.expander(f"Page {c['page']} · {c['chunk_id']} · {mark}"):
                        st.write(f"« {c['quote']} »")
                        show_passage(c["chunk_id"])
        q = st.chat_input("Ex : Quel est le montant du dépôt de garantie ?")
        if q:
            with st.spinner("Recherche..."):
                r = api().post("/ask", json={"doc_id": doc_id, "question": q})
            if not ok(r):
                st.error(api_error(r))
                return
            out = r.json()
            if out["route"] == "ask":
                st.session_state.history.append({"q": q, "a": out["result"]})
            elif out["route"] == "extract":
                st.session_state.extraction = {"extraction": out["result"], "latency_s": out["latency_s"]}
            else:
                st.session_state.report = {"report": out["result"], "latency_s": out["latency_s"]}
                st.toast("Rapport de conformité prêt : onglet « Rapport de risques ».")
            st.rerun()


def risk_report():
    st.subheader("Rapport de conformité")
    st.caption(
        "Règles déterministes (dépôt, durée, préavis, indexation, mentions obligatoires) et "
        "analyse des clauses réputées non écrites (art. 4, loi n° 89-462). Chaque clause signalée "
        "par l'analyse est vérifiée mot pour mot dans le contrat."
    )
    if st.button("Analyser la conformité", type="primary"):
        with st.spinner("Contrôle des clauses (environ 30 s)..."):
            r = api().post(f"/check/{st.session_state.doc['doc_id']}")
        st.session_state.report = r.json() if ok(r) else {"error": api_error(r)}
    rep = st.session_state.get("report")
    if not rep:
        return
    if "error" in rep:
        st.error(rep["error"])
        return
    flags = rep["report"]["flags"]
    counts = {s: sum(f["severity"] == s for f in flags) for s in SEVERITY}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points relevés", len(flags))
    c2.metric("Élevés", counts["high"])
    c3.metric("Moyens", counts["medium"])
    c4.metric("Faibles", counts["low"])
    if not flags:
        st.success("Aucun problème détecté sur les règles contrôlées.")
    for f in flags:
        icon, label = SEVERITY[f["severity"]]
        where = f" · page {f['page']}" if f.get("page") else ""
        with st.expander(f"{icon} {KIND[f['kind']]} · {f['clause'][:80]}{where}", expanded=f["severity"] == "high"):
            st.write(f["issue"])
            st.markdown(f"**Base légale :** {f['law_reference']}  \n**Gravité :** {label}"
                        f"  \n**Détecté par :** {'règle' if f['source'] == 'rule' else 'analyse des clauses (LLM, citation vérifiée)'}")
            if f.get("quote"):
                st.write(f"« {f['quote']} »")
            show_passage(f.get("chunk_id"))
    with st.expander("Règles contrôlées"):
        st.write(", ".join(rep["report"]["checked_rules"]))
    st.caption(f"Latence : {rep['latency_s']} s")


def pct(v):
    return "–" if v is None else f"{v:.0%}"


def quality():
    st.subheader("Qualité mesurée")
    st.caption(
        "Scores des tests automatiques : des questions dont la bonne réponse est connue sont posées "
        "à l'application sur des baux fictifs (`data/contracts`). Ils mesurent la fiabilité de "
        "l'outil, pas le contrat que vous avez déposé."
    )
    files = sorted(EVALS.glob("results*.json"))
    if not files:
        st.info("Aucun résultat. Lancez : python evals/run_evals.py --contract ... --gold ...")
        return
    runs = {}
    for f in files:
        try:
            runs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    metrics = [
        ("answer_accuracy", "Réponses correctes (juge)"),
        ("abstention_rate", "Abstention sur pièges"),
        ("false_abstention_rate", "Abstentions à tort"),
        ("retrieval_hit_at_1", "Bon passage en 1re position"),
        ("citation_page_accuracy", "Citation sur la bonne page"),
        ("quote_verified_rate", "Citations vérifiées"),
        ("router_accuracy", "Routage correct"),
        ("extraction_field_accuracy", "Extraction (champs exacts)"),
        ("compliance_precision", "Conformité : précision"),
        ("compliance_recall", "Conformité : rappel"),
    ]
    if all(r.get("answer_accuracy") is None for r in runs.values()):
        metrics = metrics[1:]  # no judged run: the row would only show dashes
    rows = []
    row = {"Métrique": "Contrat testé"}
    for name, r in runs.items():
        row[name] = Path(r.get("contract", "")).name or "–"
    rows.append(row)
    row = {"Métrique": "Questions réussies"}
    for name, r in runs.items():
        done = [q for q in r.get("questions", []) if "error" not in q]
        row[name] = f"{sum(q['ok'] for q in done)} / {len(r.get('questions', []))}" if r.get("questions") else "–"
    rows.append(row)
    row = {"Métrique": "Juge LLM"}
    for name, r in runs.items():
        row[name] = "activé" if r.get("answer_accuracy") is not None else "désactivé"
    rows.append(row)
    for label_key, label in metrics:
        row = {"Métrique": label}
        for name, r in runs.items():
            row[name] = pct(r.get(label_key))
        rows.append(row)
    row = {"Métrique": "Latence médiane"}
    for name, r in runs.items():
        v = r.get("median_answer_latency_s")
        row[name] = "–" if v is None else f"{v} s"
    rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    name = st.selectbox("Détail d'une exécution", list(runs))
    r = runs[name]
    st.caption(f"Modèle : {r.get('llm_model')} · recherche : {r.get('retrieval_mode')} · top_k : {r.get('top_k')}")
    if r.get("aborted"):
        st.warning(f"Exécution interrompue ({r.get('questions_answered')} questions traitées) : {r['aborted'][:200]}")
    if r.get("by_tag"):
        st.markdown("**Taux de réussite par type de question**")
        st.bar_chart({t: v["pass"] or 0 for t, v in r["by_tag"].items()}, horizontal=True)
    fails = [q for q in r.get("questions", []) if not q["ok"]]
    if fails:
        st.markdown("**Échecs**")
        for q in fails:
            with st.expander(f"{q['id']} · {q['question']}"):
                st.write(f"Attendu : {q['expected']}")
                st.write(f"Obtenu : {q.get('answer') or q.get('error')}")
                if q.get("judge_reason"):
                    st.caption(f"Juge : {q['judge_reason']}")
    if r.get("compliance_missed") or r.get("compliance_false_positives"):
        st.markdown(f"**Conformité** · manqués : {r.get('compliance_missed')} · faux positifs : {r.get('compliance_false_positives')}")


tab_work, tab_risk, tab_quality = st.tabs(["Espace de travail", "Rapport de risques", "Qualité"])
with tab_quality:
    quality()
if "doc" not in st.session_state:
    for tab in (tab_work, tab_risk):
        with tab:
            st.info("Commencez par déposer un contrat dans la barre latérale.")
else:
    with tab_work:
        workspace()
    with tab_risk:
        risk_report()
