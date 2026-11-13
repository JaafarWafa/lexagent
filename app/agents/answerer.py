"""Answer agent: question + retrieved chunks -> Answer with verified citations."""
import re
from difflib import SequenceMatcher

from app.agents.extractor import format_chunks
from app.llm import chat_json
from app.schemas import Answer, Chunk, Citation

SYSTEM = (
    "You answer questions about a French contract using ONLY the passages given. "
    "Reply in the language of the question. Every claim must cite a passage: "
    "chunk_id, page, and a short quote copied word for word from that passage. "
    "If the passages do not contain the answer, set found=false, say so in one sentence "
    "and return no citations. Never use outside knowledge to fill a gap."
)
NOT_FOUND = {
    "fr": "Je ne trouve pas cette information dans le contrat.",
    "en": "I cannot find this information in the contract.",
}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def quote_in_passage(quote: str, passage: str, min_ratio: float = 0.85) -> bool:
    """True if the quote appears in the passage, tolerating small LLM copy errors
    (accents, punctuation, a changed word)."""
    q, p = _norm(quote), _norm(passage)
    if not q:
        return False
    if q in p:
        return True
    m = SequenceMatcher(None, p, q, autojunk=False).find_longest_match(0, len(p), 0, len(q))
    if m.size == len(q):
        return True
    # Compare the quote with the passage window that best aligns with it.
    start = max(m.a - m.b, 0)
    window = p[start : start + len(q)]
    return SequenceMatcher(None, window, q, autojunk=False).ratio() >= min_ratio


def check_citations(citations: list[Citation], chunks: list[Chunk]) -> list[Citation]:
    """Drop citations to chunks that were not retrieved, fix their page number from the
    chunk itself, and flag whether each quote really appears in that chunk."""
    by_id = {c.id: c for c in chunks}
    checked: dict[str, Citation] = {}
    for cit in citations:
        chunk = by_id.get(cit.chunk_id)
        if chunk is None:  # invented id
            continue
        cit.page = chunk.page
        cit.verified = quote_in_passage(cit.quote, chunk.text)
        # Overlapping chunks contain the same sentence: show each quote once, verified first.
        key = _norm(cit.quote)
        if key not in checked or (cit.verified and not checked[key].verified):
            checked[key] = cit
    return list(checked.values())


ENGLISH_WORDS = {
    "the", "what", "who", "is", "are", "does", "do", "can", "how", "where", "when", "which",
    "why", "much", "many", "tenant", "landlord", "rent", "lease", "deposit",
}


def _looks_english(text: str) -> bool:
    return bool(set(_norm(text).split()) & ENGLISH_WORDS)


def answer(question: str, chunks: list[Chunk]) -> Answer:
    if not chunks:
        msg = NOT_FOUND["en" if _looks_english(question) else "fr"]
        return Answer(answer=msg, found=False, citations=[])
    # The model tends to answer in the contract's language (French): say the language explicitly.
    language = "English" if _looks_english(question) else "French"
    user = (
        f"Question: {question}\n\nPassages:\n{format_chunks(chunks)}\n\n"
        f"Write the answer in {language} (the quotes stay word for word in French)."
    )
    result = chat_json(SYSTEM, user, Answer)
    result.citations = check_citations(result.citations, chunks)
    if not result.found:
        result.citations = []
    return result
