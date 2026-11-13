"""Compliance agent tests, no LLM: deterministic rules, and the checks applied to the LLM scan."""
import pytest

import app.agents.compliance as comp
from app.agents.compliance import check, clause_flags, rule_flags
from app.schemas import Chunk, ClauseFinding, ClauseScan, LeaseExtraction, Party

PARTIES = [Party(role="bailleur", name="Mme A"), Party(role="locataire", name="M. B")]


def lease(**kw) -> LeaseExtraction:
    base = dict(
        parties=PARTIES, property_address="1 rue X, Lyon", lease_type="vide", start_date="2026-01-01",
        duration_months=36, monthly_rent_eur=750, deposit_eur=750, notice_period_tenant_months=3,
        notice_period_landlord_months=6, rent_indexation="IRL", landlord_type="personne physique", surface_m2=42,
    )
    return LeaseExtraction(**(base | kw))


def ids(ex: LeaseExtraction) -> set[str]:
    return {f.rule_id for f in rule_flags(ex)[0]}


def test_lawful_lease_has_no_flags():
    assert ids(lease()) == set()


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"deposit_eur": 1500}, "deposit_cap"),                              # vide: 1 month max
        ({"lease_type": "meublé", "deposit_eur": 1600, "duration_months": 12, "notice_period_tenant_months": 1,
          "notice_period_landlord_months": 3}, "deposit_cap"),               # meublé: 2 months max
        ({"lease_type": "mobilité", "deposit_eur": 100, "duration_months": 6}, "deposit_cap"),  # forbidden
        ({"duration_months": 12}, "duration_min"),
        ({"duration_months": 36, "landlord_type": "personne morale"}, "duration_min"),  # 6 years
        ({"lease_type": "mobilité", "deposit_eur": 0, "duration_months": 12}, "duration_min"),
        ({"notice_period_tenant_months": 6}, "tenant_notice_max"),
        ({"notice_period_landlord_months": 3}, "landlord_notice_min"),
        ({"rent_indexation": "indice du coût de la construction (ICC)"}, "indexation_irl"),
        ({"surface_m2": None}, "missing_surface_m2"),
        ({"parties": [Party(role="locataire", name="M. B")]}, "missing_bailleur"),
    ],
)
def test_rule_flags(changes, expected):
    assert expected in ids(lease(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"deposit_eur": 750.0},                                              # exactly 1 month
        {"lease_type": "meublé", "deposit_eur": 1500, "duration_months": 12, "notice_period_tenant_months": 1,
         "notice_period_landlord_months": 3},                                # 2 months, lawful
        {"duration_months": 72, "landlord_type": "personne morale"},
        {"notice_period_tenant_months": 1},                                  # shorter than legal: fine
        {"notice_period_landlord_months": 8},                                # longer than legal: fine
        {"rent_indexation": "indice de référence des loyers publié par l'INSEE"},
        {"deposit_eur": None, "rent_indexation": None},                      # unknown: not checked
    ],
)
def test_no_false_positive(changes):
    assert ids(lease(**changes)) == set()


def test_commercial_lease_is_out_of_scope():
    flags, checked = rule_flags(lease(lease_type="commercial", deposit_eur=5000, duration_months=12))
    assert [f.rule_id for f in flags] == ["out_of_scope"] and checked == ["out_of_scope"]


def test_rule_flag_points_to_evidence_chunk():
    chunks = [Chunk(id="p2-c0", page=2, text="Le dépôt de garantie est fixé à 1600 euros.")]
    flags, _ = rule_flags(lease(deposit_eur=1600, evidence={"deposit_eur": "p2-c0"}), chunks)
    assert (flags[0].chunk_id, flags[0].page) == ("p2-c0", 2)
    flags, _ = rule_flags(lease(deposit_eur=1600, evidence={"deposit_eur": "p9-c9"}), chunks)
    assert (flags[0].chunk_id, flags[0].page) == (None, None)  # invented evidence id is ignored


# --- LLM clause scan: the server never trusts it blindly ------------------------------

CHUNKS = [
    Chunk(id="p3-c0", page=3, text="La détention de tout animal, quel qu'il soit, est formellement interdite dans le logement."),
    Chunk(id="p3-c1", page=3, text="La sous-location est interdite sauf accord écrit du bailleur."),
]


def scan_returns(monkeypatch, *findings):
    monkeypatch.setattr(comp, "chat_json", lambda *a, **k: ClauseScan(findings=list(findings)))


def test_clause_scan_keeps_verified_findings_and_sets_law_reference(monkeypatch):
    scan_returns(monkeypatch, ClauseFinding(
        rule_id="pets_ban", chunk_id="p3-c0", quote="La détention de tout animal, quel qu'il soit, est formellement interdite",
        explanation="Interdit tout animal.", law_reference="invented by the model",
    ))
    [f] = clause_flags(CHUNKS)
    assert f.rule_id == "pets_ban" and f.page == 3 and f.source == "llm"
    assert f.law_reference == "Art. 10, loi n° 70-598 du 9 juillet 1970"


@pytest.mark.parametrize(
    "finding",
    [
        ClauseFinding(rule_id="pets_ban", chunk_id="p3-c0", quote="Les chiens sont interdits dans la cour", explanation=""),
        ClauseFinding(rule_id="pets_ban", chunk_id="p7-c7", quote="La détention de tout animal", explanation=""),
        ClauseFinding(rule_id="made_up_rule", chunk_id="p3-c0", quote="La détention de tout animal", explanation=""),
    ],
    ids=["quote-not-in-chunk", "invented-chunk", "unknown-rule"],
)
def test_clause_scan_drops_unverifiable_findings(monkeypatch, finding):
    scan_returns(monkeypatch, finding)
    assert clause_flags(CHUNKS) == []


def test_clause_scan_deduplicates(monkeypatch):
    f = ClauseFinding(rule_id="pets_ban", chunk_id="p3-c0", quote="tout animal, quel qu'il soit", explanation="")
    scan_returns(monkeypatch, f, f)
    assert len(clause_flags(CHUNKS)) == 1


def test_check_sorts_by_severity(monkeypatch):
    scan_returns(monkeypatch, ClauseFinding(
        rule_id="pets_ban", chunk_id="p3-c0", quote="tout animal, quel qu'il soit", explanation=""))
    report = check(CHUNKS, lease(deposit_eur=1600))
    assert [f.severity for f in report.flags] == ["high", "medium"]
    assert "deposit_cap" in report.checked_rules and "pets_ban" in report.checked_rules
