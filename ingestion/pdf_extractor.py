"""
AI-COS — PDF Text Extractor
Uses PyMuPDF (fitz) to extract text with page numbers and section detection.
Detects scanned/empty-text PDFs and reports them rather than silently failing.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_SCANNED_THRESHOLD = 100  # chars per page: below this → likely scanned


def extract_pdf(file_path: str | Path) -> Dict[str, Any]:
    """
    Extract text from a PDF file.

    Returns
    -------
    dict with keys:
        full_text       : str   – concatenated text
        pages           : list  – [{page_num, text, char_count}]
        total_chars     : int
        is_likely_scanned: bool
        warnings        : list[str]
        error           : str | None
    """
    file_path = Path(file_path)
    result: Dict[str, Any] = {
        "full_text": "",
        "pages": [],
        "total_chars": 0,
        "is_likely_scanned": False,
        "warnings": [],
        "error": None,
    }

    try:
        import fitz  # PyMuPDF
    except ImportError:
        result["error"] = "PyMuPDF (fitz) is not installed."
        return result

    if not file_path.exists():
        result["error"] = f"File not found: {file_path}"
        return result

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        result["error"] = f"Cannot open PDF: {exc}"
        return result

    pages = []
    all_text_parts: List[str] = []
    scanned_page_count = 0

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")          # plain text extraction
            char_count = len(text.strip())
            if char_count < _SCANNED_THRESHOLD:
                scanned_page_count += 1
            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "char_count": char_count,
            })
            all_text_parts.append(text)
        except Exception as exc:
            result["warnings"].append(f"Page {page_num + 1}: extraction error – {exc}")

    doc.close()

    full_text = "\n".join(all_text_parts)
    result["full_text"] = full_text
    result["pages"] = pages
    result["total_chars"] = len(full_text.strip())

    # Scanned detection: if >50% of pages have very little text
    if pages and scanned_page_count / len(pages) > 0.5:
        result["is_likely_scanned"] = True
        result["warnings"].append(
            "Document appears to be a scanned PDF. Text extraction may be incomplete. "
            "Consider enabling OCR if Tesseract is installed."
        )

    if result["total_chars"] == 0:
        result["warnings"].append("No text was extracted from this PDF.")

    logger.debug("PDF extracted: %s  chars=%d  pages=%d",
                 file_path.name, result["total_chars"], len(pages))
    return result
