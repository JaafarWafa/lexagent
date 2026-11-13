"""PDF -> text chunks that keep their page number (needed for citations)."""
import re
from pathlib import Path

from pypdf import PdfReader

from app.config import settings
from app.schemas import Chunk


def read_pdf(path: str | Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")


def _split_long(sentence: str, size: int) -> list[str]:
    """Hard-split a sentence longer than `size`, at word boundaries."""
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return parts + ([current] if current else [])


def chunk_pages(
    pages: list[tuple[int, str]],
    size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Pack whole sentences into chunks of at most `size` characters, page by page.
    The last sentence(s) of a chunk, up to `overlap` characters, are repeated at the start
    of the next one so a clause cut between two chunks is still found whole."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        text = " ".join(text.split())
        sentences = [p for s in _SENTENCE_END.split(text) if s for p in _split_long(s, size)]
        current: list[str] = []
        n = 0

        def flush():
            nonlocal n
            chunks.append(Chunk(id=f"p{page_no}-c{n}", page=page_no, text=" ".join(current)))
            n += 1

        for sentence in sentences:
            if current and len(" ".join(current + [sentence])) > size:
                flush()
                tail: list[str] = []
                for prev in reversed(current):
                    if len(" ".join([prev] + tail)) > overlap:
                        break
                    tail.insert(0, prev)
                current = tail
                if current and len(" ".join(current + [sentence])) > size:
                    current = []
            current.append(sentence)
        if current:
            flush()
    return chunks


def ingest(path: str | Path) -> list[Chunk]:
    return chunk_pages(read_pdf(path))
