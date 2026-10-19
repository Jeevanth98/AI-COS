"""
AI-COS — NLP Preprocessing
Sentence segmentation, tokenisation, and basic linguistic analysis.
Uses spaCy en_core_web_sm; falls back gracefully if the model is unavailable.
"""

from __future__ import annotations
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_nlp = None  # lazy-loaded


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _nlp = False
    return _nlp if _nlp is not False else None


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences. Falls back to regex if spaCy unavailable."""
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:100_000])  # cap to avoid OOM on huge blobs
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Fallback: coarse regex
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Named entity recognition using spaCy.
    Returns list of {text, label} dicts.
    Falls back to empty list if spaCy unavailable.
    """
    nlp = _get_nlp()
    if not nlp:
        return []
    doc = nlp(text[:100_000])
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in {
            "ORG", "PERSON", "GPE", "DATE", "TIME", "PERCENT",
            "MONEY", "CARDINAL", "ORDINAL", "LAW", "NORP",
        }
    ]


def extract_dates(text: str) -> List[str]:
    """Extract date-like strings using a regex heuristic."""
    date_re = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|(?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",
        re.IGNORECASE,
    )
    return date_re.findall(text)


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Extract number+unit patterns (e.g. '3 days', '50%', '$200')."""
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*"
        r"(%|percent|days?|weeks?|months?|years?|hours?|minutes?"
        r"|USD|INR|\$|₹|km|kg|ml|GB|MB|TB)?",
        re.IGNORECASE,
    )
    results = []
    for m in pattern.finditer(text):
        value = m.group(1)
        unit = (m.group(2) or "").strip()
        results.append({"value": float(value), "unit": unit, "span": m.group(0).strip()})
    return results
