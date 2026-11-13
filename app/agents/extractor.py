"""Extractor agent: contract chunks -> LeaseExtraction (validated JSON)."""
import unicodedata

from app.llm import chat_json
from app.schemas import Chunk, LeaseExtraction

SYSTEM = (
    "Tu es un juriste spécialisé en baux d'habitation français. "
    "Extrais les informations demandées du contrat. "
    "N'invente rien : si une information est absente, mets null. "
    "landlord_type : 'personne morale' si le bailleur est une société, une SCI (non familiale), "
    "un organisme ; 'personne physique' si c'est un particulier. "
    "Les durées et préavis sont en mois (trois ans = 36). "
    "Dans 'evidence', indique pour chaque champ rempli l'id du passage (ex: p2-c1) où tu l'as trouvé."
)

# Words a passage must contain to be the source of a field (any group, all words of a group).
# The LLM's evidence ids are sometimes off by a passage; these let the server check and fix them.
FIELD_KEYWORDS: dict[str, list[tuple[str, ...]]] = {
    "parties": [("bailleur",), ("locataire",)],
    "property_address": [("situe",), ("adresse",), ("logement",)],
    "lease_type": [("meuble",), ("vide",), ("mobilite",)],
    "start_date": [("prend effet",), ("prise d'effet",), ("a compter du",)],
    "duration_months": [("duree",)],
    "monthly_rent_eur": [("loyer",)],
    "charges_eur": [("charges",)],
    "deposit_eur": [("depot de garantie",)],
    "notice_period_tenant_months": [("preavis", "locataire")],
    "notice_period_landlord_months": [("preavis", "bailleur")],
    "rent_indexation": [("revis",), ("indice",), ("irl",)],
    "landlord_type": [("bailleur",)],
    "surface_m2": [("surface",)],
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _mentions(chunk: Chunk, field: str) -> bool:
    text = _fold(chunk.text)
    return any(all(word in text for word in group) for group in FIELD_KEYWORDS.get(field, []))


def fix_evidence(ex: LeaseExtraction, chunks: list[Chunk]) -> LeaseExtraction:
    """Keep the LLM's source passage when it mentions the field, otherwise point to the first
    passage that does; drop it when none does or the id is invented."""
    by_id = {c.id: c for c in chunks}
    fixed = {}
    for field, chunk_id in ex.evidence.items():
        if field not in FIELD_KEYWORDS:
            if chunk_id in by_id:
                fixed[field] = chunk_id
            continue
        chunk = by_id.get(chunk_id)
        if chunk is not None and _mentions(chunk, field):
            fixed[field] = chunk_id
            continue
        match = next((c for c in chunks if _mentions(c, field)), None)
        if match is not None:
            fixed[field] = match.id
    ex.evidence = fixed
    return ex


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id} | page {c.page}]\n{c.text}" for c in chunks)


# Fields the compliance rules depend on: an empty one would silently become a false
# "missing mention" flag, so they get a second, focused look before being reported absent.
KEY_FIELDS = (
    "start_date", "duration_months", "monthly_rent_eur", "deposit_eur",
    "notice_period_tenant_months", "notice_period_landlord_months", "surface_m2", "lease_type",
)


def _fill_missing(ex: LeaseExtraction, chunks: list[Chunk]) -> LeaseExtraction:
    missing = [f for f in KEY_FIELDS if getattr(ex, f) is None]
    relevant = [c for c in chunks if any(_mentions(c, f) for f in missing)]
    if not missing or not relevant:
        return ex
    focus = (
        f"{SYSTEM}\nConcentre-toi UNIQUEMENT sur ces champs : {', '.join(missing)}. "
        "Lis chaque passage attentivement ; laisse null seulement si l'information est vraiment absente."
    )
    try:
        second = chat_json(focus, format_chunks(relevant), LeaseExtraction)
    except Exception:  # the first answer stands
        return ex
    for f in missing:
        value = getattr(second, f)
        if value is not None:
            setattr(ex, f, value)
            if f in second.evidence:
                ex.evidence[f] = second.evidence[f]
    return ex


def extract(chunks: list[Chunk]) -> LeaseExtraction:
    ex = chat_json(SYSTEM, format_chunks(chunks), LeaseExtraction)
    return fix_evidence(_fill_missing(ex, chunks), chunks)
