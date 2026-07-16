"""
pdf_utils.py
Handles PDF text extraction and chunking for the Research Paper Summarizer + QA app.
"""

from typing import List
import re
from pypdf import PdfReader


def extract_text_from_pdf(file) -> str:
    """
    Extract raw text from a PDF file object (as given by Streamlit's file_uploader).
    Returns a single cleaned string with page breaks marked.
    """
    reader = PdfReader(file)
    pages_text = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages_text.append(f"\n[Page {i + 1}]\n{text}")

    full_text = "\n".join(pages_text)
    return clean_text(full_text)


def clean_text(text: str) -> str:
    """Basic cleanup: collapse excess whitespace, fix hyphenated line breaks."""
    text = re.sub(r"-\n", "", text)          # join hyphenated words split across lines
    text = re.sub(r"[ \t]+", " ", text)       # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse excess blank lines
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for embedding + retrieval.

    chunk_size: approx number of characters per chunk
    overlap: number of characters shared between consecutive chunks,
             which helps preserve context across chunk boundaries.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # try to break on a sentence or paragraph boundary near the end
        if end < text_length:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end - overlap > start else end

    return chunks


def estimate_token_count(text: str) -> int:
    """Rough token estimate (≈4 chars/token) used for UI display, not billing."""
    return max(1, len(text) // 4)
