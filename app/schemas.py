"""Data contracts. Everything the agents return is validated here."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    page: int
    text: str


class Party(BaseModel):
    role: Literal["bailleur", "locataire", "garant", "vendeur", "acquéreur", "autre"]
    name: str


class LeaseExtraction(BaseModel):
    """Key fields of a French residential lease (bail d'habitation)."""

    parties: list[Party] = Field(default_factory=list)
    property_address: Optional[str] = None
    lease_type: Optional[Literal["vide", "meublé", "mobilité", "commercial", "autre"]] = None
    start_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD")
    duration_months: Optional[int] = None
    monthly_rent_eur: Optional[float] = None
    charges_eur: Optional[float] = None
    deposit_eur: Optional[float] = None
    notice_period_tenant_months: Optional[int] = None
    rent_indexation: Optional[str] = Field(None, description="e.g. IRL, annual date")
    landlord_type: Optional[Literal["personne physique", "personne morale"]] = Field(
        None, description="personne morale = société, SCI non familiale, bailleur social..."
    )
    notice_period_landlord_months: Optional[int] = None
    surface_m2: Optional[float] = Field(None, description="surface habitable")
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description="field name -> chunk id where the value was found",
    )


class Citation(BaseModel):
    chunk_id: str
    page: int
    quote: str = Field(description="short exact passage copied from the contract")
    verified: bool = Field(
        False, description="set by the server: the quote really appears in that passage"
    )


class Answer(BaseModel):
    answer: str
    found: bool = Field(
        True, description="false when the passages do not contain the answer"
    )
    citations: list[Citation] = Field(default_factory=list)


class RoutePlan(BaseModel):
    """Router output: which agent handles the request, plus French search keywords."""

    route: Literal["extract", "ask", "check"]
    keywords: str = Field("", description="4-8 French legal keywords for the search engine")


class AskRequest(BaseModel):
    doc_id: str
    question: str


class Flag(BaseModel):
    """Compliance agent output: one problem in the contract, with its legal basis."""

    rule_id: str = Field(description="stable id, e.g. deposit_cap, art4_pets_ban")
    kind: Literal["illegal", "missing", "warning"]
    clause: str = Field(description="the contract term concerned")
    issue: str
    law_reference: str
    severity: Literal["low", "medium", "high"]
    source: Literal["rule", "llm"] = "rule"
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    quote: Optional[str] = None


class ClauseFinding(BaseModel):
    """What the LLM clause scan returns, before the server checks it."""

    rule_id: str
    chunk_id: str
    quote: str = Field(description="the clause, copied word for word")
    explanation: str


class ClauseScan(BaseModel):
    findings: list[ClauseFinding] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    flags: list[Flag] = Field(default_factory=list)
    checked_rules: list[str] = Field(default_factory=list)
    extraction: Optional[LeaseExtraction] = None
