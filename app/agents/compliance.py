"""Compliance agent: flags illegal, abusive or missing clauses in a French residential lease.

Two layers, so that no flag rests on an invented law article:

1. Deterministic rules on the extracted fields (deposit cap, minimum duration, notice
   periods, rent indexation, mandatory mentions). Each rule carries its own legal reference.
2. An LLM scan of the clauses against a fixed checklist of clauses that the law deems
   unwritten ("réputées non écrites", art. 4 loi n° 89-462). The LLM only picks checklist
   ids and quotes the clause; the server attaches the law reference and drops any finding
   whose quote is not found in the cited chunk.

Scope: residential leases (loi n° 89-462 du 6 juillet 1989), unfurnished, furnished and
bail mobilité. Not legal advice: a lawyer should confirm every flag.
"""
from dataclasses import dataclass

from app.agents.answerer import quote_in_passage
from app.agents.extractor import extract, format_chunks
from app.llm import chat_json
from app.schemas import Chunk, ClauseScan, ComplianceReport, Flag, LeaseExtraction

LOI = "loi n° 89-462 du 6 juillet 1989"


# --- 1. Deterministic rules ------------------------------------------------------


def _where(ex: LeaseExtraction, field: str, chunks: list[Chunk]) -> dict:
    chunk_id = ex.evidence.get(field)
    page = next((c.page for c in chunks if c.id == chunk_id), None)
    return {"chunk_id": chunk_id if page else None, "page": page}


def rule_flags(ex: LeaseExtraction, chunks: list[Chunk] | None = None) -> tuple[list[Flag], list[str]]:
    """Return (flags, ids of the rules that could be checked)."""
    chunks = chunks or []
    flags: list[Flag] = []
    checked: list[str] = []
    t = ex.lease_type

    def flag(rule_id, field, **kw):
        flags.append(Flag(rule_id=rule_id, **kw, **_where(ex, field, chunks)))

    if t == "commercial":
        flag(
            "out_of_scope", "lease_type", kind="warning", clause="Type de bail", severity="low",
            issue="Bail commercial : les règles des baux d'habitation ne s'appliquent pas, contrôle non effectué.",
            law_reference="Code de commerce, art. L145-1 et s.",
        )
        return flags, ["out_of_scope"]

    # Dépôt de garantie
    if ex.deposit_eur is not None and t in ("vide", "meublé", "mobilité"):
        checked.append("deposit_cap")
        if t == "mobilité" and ex.deposit_eur > 0:
            flag(
                "deposit_cap", "deposit_eur", kind="illegal", clause="Dépôt de garantie", severity="high",
                issue=f"Un dépôt de garantie ({ex.deposit_eur:g} €) est exigé alors qu'il est interdit pour un bail mobilité.",
                law_reference=f"Art. 25-13, {LOI}",
            )
        elif ex.monthly_rent_eur:
            months_max = 1 if t == "vide" else 2
            if ex.deposit_eur > months_max * ex.monthly_rent_eur + 0.01:
                flag(
                    "deposit_cap", "deposit_eur", kind="illegal", clause="Dépôt de garantie", severity="high",
                    issue=(
                        f"Dépôt de {ex.deposit_eur:g} € pour un loyer de {ex.monthly_rent_eur:g} € hors charges : "
                        f"le maximum légal est {months_max} mois de loyer hors charges "
                        f"({months_max * ex.monthly_rent_eur:g} €)."
                    ),
                    law_reference=f"Art. {'22' if t == 'vide' else '25-6'}, {LOI}",
                )

    # Durée minimale
    d = ex.duration_months
    if d is not None and t in ("vide", "meublé", "mobilité"):
        checked.append("duration_min")
        if t == "vide":
            minimum = 72 if ex.landlord_type == "personne morale" else 36
            if d < minimum:
                who = "une personne morale" if minimum == 72 else "une personne physique"
                flag(
                    "duration_min", "duration_months", kind="illegal", clause="Durée du bail", severity="high",
                    issue=f"Durée de {d} mois : un bail vide consenti par {who} dure au moins {minimum // 12} ans.",
                    law_reference=f"Art. 10, {LOI}",
                )
        elif t == "meublé" and d < 12:
            flag(
                "duration_min", "duration_months", kind="illegal", clause="Durée du bail", severity="medium",
                issue=f"Durée de {d} mois : un bail meublé dure au moins 1 an (9 mois seulement pour un étudiant).",
                law_reference=f"Art. 25-7, {LOI}",
            )
        elif t == "mobilité" and not 1 <= d <= 10:
            flag(
                "duration_min", "duration_months", kind="illegal", clause="Durée du bail", severity="high",
                issue=f"Durée de {d} mois : un bail mobilité dure de 1 à 10 mois.",
                law_reference=f"Art. 25-14, {LOI}",
            )

    # Préavis du locataire : un préavis plus long que la loi est réputé non écrit
    n = ex.notice_period_tenant_months
    if n is not None and t in ("vide", "meublé", "mobilité"):
        checked.append("tenant_notice_max")
        legal = 3 if t == "vide" else 1
        if n > legal:
            flag(
                "tenant_notice_max", "notice_period_tenant_months", kind="illegal", clause="Préavis du locataire",
                severity="high",
                issue=(
                    f"Préavis de {n} mois imposé au locataire : la loi le limite à {legal} mois"
                    + (" (1 mois en zone tendue ou dans les cas prévus par la loi)." if t == "vide" else ".")
                ),
                law_reference=f"Art. {'15' if t == 'vide' else '25-8'}, {LOI}",
            )

    # Préavis du bailleur : il ne peut pas être raccourci
    n = ex.notice_period_landlord_months
    if n is not None and t in ("vide", "meublé"):
        checked.append("landlord_notice_min")
        legal = 6 if t == "vide" else 3
        if n < legal:
            flag(
                "landlord_notice_min", "notice_period_landlord_months", kind="illegal",
                clause="Préavis du bailleur", severity="medium",
                issue=f"Préavis du bailleur de {n} mois : il doit donner congé au moins {legal} mois avant l'échéance.",
                law_reference=f"Art. {'15' if t == 'vide' else '25-8'}, {LOI}",
            )

    # Révision du loyer : seul l'IRL est autorisé
    if ex.rent_indexation:
        checked.append("indexation_irl")
        text = ex.rent_indexation.lower()
        if "irl" not in text and "indice de référence des loyers" not in text:
            flag(
                "indexation_irl", "rent_indexation", kind="illegal", clause="Révision du loyer", severity="medium",
                issue=f"Révision selon « {ex.rent_indexation} » : seule la variation de l'IRL publié par l'INSEE est autorisée.",
                law_reference=f"Art. 17-1, {LOI}",
            )

    # Mentions obligatoires
    checked.append("mandatory_mentions")
    roles = {p.role for p in ex.parties}
    required = {
        "bailleur": ("bailleur" in roles, "Identité du bailleur"),
        "locataire": ("locataire" in roles, "Identité du locataire"),
        "property_address": (bool(ex.property_address), "Adresse du logement"),
        "start_date": (bool(ex.start_date), "Date de prise d'effet"),
        "duration_months": (ex.duration_months is not None, "Durée du bail"),
        "monthly_rent_eur": (ex.monthly_rent_eur is not None, "Montant du loyer"),
        "surface_m2": (ex.surface_m2 is not None, "Surface habitable"),
    }
    for field, (present, label) in required.items():
        if not present:
            flags.append(Flag(
                rule_id=f"missing_{field}", kind="missing", clause=label,
                severity="high" if field in ("bailleur", "locataire", "monthly_rent_eur") else "medium",
                issue=f"Mention obligatoire absente : {label.lower()}."
                + (" Le locataire peut demander une diminution de loyer si la surface est inexacte." if field == "surface_m2" else ""),
                law_reference=f"Art. 3{' et 3-1' if field == 'surface_m2' else ''}, {LOI}",
            ))
    return flags, checked


# --- 2. LLM scan against a fixed checklist ----------------------------------------


@dataclass(frozen=True)
class ClauseRule:
    id: str
    description: str  # what the LLM looks for
    issue: str  # what the report says
    law_reference: str
    severity: str = "high"


CLAUSE_RULES = [
    ClauseRule(
        "art4_visits", "obliger le locataire à laisser visiter le logement les jours fériés, ou plus de deux heures par jour ouvrable",
        "Clause réputée non écrite : visites imposées les jours fériés ou plus de deux heures par jour ouvrable.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_insurer_choice", "obliger le locataire à souscrire l'assurance auprès d'une compagnie choisie par le bailleur",
        "Clause réputée non écrite : le bailleur ne peut pas imposer l'assureur.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_direct_debit", "imposer le prélèvement automatique, le prélèvement sur salaire, ou des traites signées d'avance comme mode de paiement du loyer",
        "Clause réputée non écrite : le prélèvement automatique ne peut pas être imposé.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_collective_liability", "rendre les locataires collectivement responsables de la dégradation d'un équipement commun",
        "Clause réputée non écrite : responsabilité collective des locataires.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_landlord_judge", "prévoir que le bailleur est seul juge des réparations ou qu'il retient sur le dépôt des sommes forfaitaires sans justificatif",
        "Clause réputée non écrite : le bailleur ne peut pas être seul juge ni fixer des retenues forfaitaires.",
        f"Art. 4 et 22, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_termination", "prévoir la résiliation de plein droit du bail pour un motif autre que : impayé de loyer, de charges ou du dépôt de garantie, défaut d'assurance, ou troubles de voisinage constatés par décision de justice",
        "Clause réputée non écrite : clause résolutoire pour un motif non prévu par la loi.",
        f"Art. 4 et 24, {LOI}",
    ),
    ClauseRule(
        "art4_penalties", "autoriser le bailleur à percevoir des amendes ou pénalités (par ex. par jour de retard ou en cas d'infraction au bail)",
        "Clause réputée non écrite : le bailleur ne peut pas infliger d'amendes ou de pénalités.",
        f"Art. 4, {LOI}",
    ),
    ClauseRule(
        "art4_activities", "interdire au locataire une activité politique, syndicale, associative ou confessionnelle",
        "Clause réputée non écrite : interdiction d'activités politiques, syndicales, associatives ou confessionnelles.",
        f"Art. 4, {LOI}",
    ),
    ClauseRule(
        "art4_receipt_fees", "faire payer au locataire des frais d'envoi ou d'établissement de quittance, des frais de relance ou de mise en demeure",
        "Clause réputée non écrite : frais de quittance, de relance ou de mise en demeure à la charge du locataire.",
        f"Art. 4 et 21, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_waiver", "exonérer le bailleur de toute responsabilité, ou faire renoncer le locataire à un recours contre lui",
        "Clause réputée non écrite : exonération de responsabilité du bailleur.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "art4_guests", "interdire au locataire d'héberger des personnes ne vivant pas habituellement avec lui",
        "Clause réputée non écrite : le locataire peut héberger qui il veut.",
        f"Art. 4, {LOI}", "medium",
    ),
    ClauseRule(
        "pets_ban", "interdire de détenir un animal familier (chien, chat...) dans le logement",
        "Clause réputée non écrite : la détention d'un animal familier ne peut pas être interdite (sauf chiens d'attaque).",
        "Art. 10, loi n° 70-598 du 9 juillet 1970", "medium",
    ),
]
RULES_BY_ID = {r.id: r for r in CLAUSE_RULES}

SCAN_SYSTEM = (
    "Tu es juriste en droit du logement français. Tu cherches dans un bail d'habitation les "
    "clauses qui correspondent à la liste ci-dessous. Ne signale une clause QUE si elle "
    "correspond clairement à une entrée de la liste ; une clause conforme ou favorable au "
    "locataire ne doit pas être signalée. Pour chaque clause trouvée : rule_id (de la liste), "
    "chunk_id du passage, quote = la clause copiée mot pour mot, explanation en une phrase. "
    "Si rien ne correspond, renvoie findings = [].\n\nListe :\n"
    + "\n".join(f"- {r.id} : {r.description}" for r in CLAUSE_RULES)
)


def clause_flags(chunks: list[Chunk]) -> list[Flag]:
    scan = chat_json(SCAN_SYSTEM, format_chunks(chunks), ClauseScan)
    by_id = {c.id: c for c in chunks}
    flags, seen = [], set()
    for f in scan.findings:
        rule, chunk = RULES_BY_ID.get(f.rule_id), by_id.get(f.chunk_id)
        if rule is None or chunk is None or rule.id in seen:
            continue  # unknown rule, invented chunk, or duplicate
        if not quote_in_passage(f.quote, chunk.text):
            continue  # the clause is not really in the contract: drop it
        seen.add(rule.id)
        flags.append(Flag(
            rule_id=rule.id, kind="illegal", clause=f.quote[:200], issue=f"{rule.issue} {f.explanation}".strip(),
            law_reference=rule.law_reference, severity=rule.severity, source="llm",
            chunk_id=chunk.id, page=chunk.page, quote=f.quote,
        ))
    return flags


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def check(chunks: list[Chunk], extraction: LeaseExtraction | None = None) -> ComplianceReport:
    extraction = extraction or extract(chunks)
    flags, checked = rule_flags(extraction, chunks)
    if extraction.lease_type != "commercial":
        flags += clause_flags(chunks)
        checked += [r.id for r in CLAUSE_RULES]
    flags.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.page or 99))
    return ComplianceReport(flags=flags, checked_rules=checked, extraction=extraction)
