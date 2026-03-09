"""Document parsing utilities for PDF, DOCX, and plain text files."""

from __future__ import annotations

import re
from pathlib import Path


def parse_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def parse_text(path: Path) -> str:
    """Read a plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


PARSERS = {
    ".docx": parse_docx,
    ".pdf": parse_pdf,
    ".txt": parse_text,
    ".md": parse_text,
}

SUPPORTED_EXTENSIONS = set(PARSERS.keys())


def parse_document(path: Path) -> str:
    """Parse a document file, dispatching by extension.

    Returns the extracted text content.
    Raises ValueError for unsupported file types.
    """
    ext = path.suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext} ({path.name})")
    return parser(path)


def chunk_text(
    text: str,
    max_words: int = 6000,
    overlap_words: int = 500,
) -> list[str]:
    """Split text into chunks by word count with overlap.

    Returns a list of text chunks. If the text is short enough,
    returns a single-element list with the original text.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Advance past the chunk minus the overlap
        start = end - overlap_words
        # Avoid tiny trailing chunks
        if start + overlap_words >= len(words):
            break

    return chunks


def clean_extracted_text(text: str) -> str:
    """Clean up common artifacts from PDF/docx extraction."""
    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page break artifacts
    text = re.sub(r"\f", "\n\n", text)
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()
