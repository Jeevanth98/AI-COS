"""
AI-COS — Semantic Chunker
Splits cleaned text into overlapping character-level chunks that respect
sentence and paragraph boundaries wherever possible.
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Any

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_SECTION_HEADING_RE = re.compile(
    r"^(\d+[\.\d]*\.?\s+[A-Z]|\b(?:Section|Article|Chapter|Part|Clause)\b\s+\d+)",
    re.MULTILINE,
)


def _split_into_sentences(text: str) -> List[str]:
    """Coarse sentence splitter (no spaCy dependency here to keep chunker fast)."""
    # Split on '. ', '! ', '? ', '.\n'
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    page_map: List[Dict] | None = None,
) -> List[Dict[str, Any]]:
    """
    Split *text* into overlapping chunks, trying to respect paragraph breaks.

    Parameters
    ----------
    text         : cleaned full text
    chunk_size   : target maximum characters per chunk
    chunk_overlap: characters of overlap between consecutive chunks
    page_map     : optional list of {page_num, text} dicts from extractor
                   used to approximate page numbers for each chunk

    Returns
    -------
    list of dicts:
        chunk_index    : int
        text           : str
        char_start     : int
        char_end       : int
        page_number    : int | None
        section_heading: str | None
    """
    if not text or not text.strip():
        return []

    # Build a char-offset → page_number map for approximate page attribution
    page_offsets: List[tuple[int, int]] = []  # (start_char, page_num)
    if page_map:
        offset = 0
        for page in page_map:
            page_offsets.append((offset, page["page_num"]))
            offset += len(page["text"])

    def _get_page(char_start: int) -> int | None:
        if not page_offsets:
            return None
        pn = page_offsets[0][1]
        for start, page_num in page_offsets:
            if char_start >= start:
                pn = page_num
            else:
                break
        return pn

    # ── Paragraph-level splitting ──────────────────────────────────────────
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[Dict[str, Any]] = []
    current_chars: List[str] = []
    current_start = 0
    global_offset = 0
    current_section: str | None = None

    idx = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            global_offset += 2  # blank lines
            continue

        # Track section headings
        if _SECTION_HEADING_RE.match(para):
            current_section = para[:120]

        if len(" ".join(current_chars) + " " + para) > chunk_size and current_chars:
            # Flush current buffer
            chunk_text_str = " ".join(current_chars)
            end = current_start + len(chunk_text_str)
            chunks.append({
                "chunk_index": idx,
                "text": chunk_text_str,
                "char_start": current_start,
                "char_end": end,
                "page_number": _get_page(current_start),
                "section_heading": current_section,
            })
            idx += 1

            # Overlap: keep last overlap chars worth of content
            overlap_text = chunk_text_str[-chunk_overlap:]
            current_chars = [overlap_text] if overlap_text.strip() else []
            current_start = max(0, end - chunk_overlap)

        current_chars.append(para)
        global_offset += len(para) + 2

    # Flush remaining
    if current_chars:
        chunk_text_str = " ".join(current_chars)
        end = current_start + len(chunk_text_str)
        chunks.append({
            "chunk_index": idx,
            "text": chunk_text_str,
            "char_start": current_start,
            "char_end": end,
            "page_number": _get_page(current_start),
            "section_heading": current_section,
        })

    # Ensure no chunk exceeds 2×chunk_size (split large ones)
    final_chunks: List[Dict[str, Any]] = []
    final_idx = 0
    for chunk in chunks:
        if len(chunk["text"]) <= chunk_size * 2:
            chunk["chunk_index"] = final_idx
            final_chunks.append(chunk)
            final_idx += 1
        else:
            # Hard split
            text_slice = chunk["text"]
            start = chunk["char_start"]
            while text_slice:
                piece = text_slice[:chunk_size]
                final_chunks.append({
                    "chunk_index": final_idx,
                    "text": piece,
                    "char_start": start,
                    "char_end": start + len(piece),
                    "page_number": chunk["page_number"],
                    "section_heading": chunk["section_heading"],
                })
                final_idx += 1
                text_slice = text_slice[chunk_size - chunk_overlap:]
                start += chunk_size - chunk_overlap

    logger.debug("Chunked text: %d chunks  (size=%d, overlap=%d)",
                 len(final_chunks), chunk_size, chunk_overlap)
    return final_chunks
