"""Extractor agent: contract chunks -> LeaseExtraction (validated JSON)."""
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


def format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id} | page {c.page}]\n{c.text}" for c in chunks)


def extract(chunks: list[Chunk]) -> LeaseExtraction:
    return chat_json(SYSTEM, format_chunks(chunks), LeaseExtraction)
