"""
AI-COS — Text Cleaner
Normalises raw extracted text before NLP processing.
"""

from __future__ import annotations
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise raw extracted text:
    1. Unicode normalisation (NFC).
    2. Replace non-breaking and unusual whitespace with regular spaces.
    3. Collapse multiple blank lines to at most two.
    4. Strip leading/trailing whitespace from each line.
    5. Remove form-feed characters.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Non-standard whitespace → space (but keep newlines)
    text = re.sub(r"[ \t\r\xa0\u200b\u2009\u202f]+", " ", text)

    # 3. Form feeds → newline
    text = text.replace("\f", "\n")

    # 4. Strip per line
    lines = [line.strip() for line in text.splitlines()]

    # 5. Collapse runs of blank lines to at most 2
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_headers_footers_heuristic(text: str) -> str:
    """
    Very light heuristic: remove lines that look like page-number-only lines
    (e.g. '--- 3 ---', 'Page 3 of 12') from the text.
    Does NOT remove content; only obvious artifacts.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    page_pattern = re.compile(
        r"^(\s*[-–—]+\s*)?\d+(\s*[-–—]+\s*)?$"   # --- 3 ---
        r"|^[Pp]age\s+\d+(\s+of\s+\d+)?$"
    )
    for line in lines:
        if page_pattern.match(line.strip()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
