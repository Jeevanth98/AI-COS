"""
AI-COS — Policy Parser
Rule-based extraction of structured policy statements from text chunks.
Identifies modality (permission / obligation / prohibition / recommendation / factual),
actors, actions, conditions, exceptions, and numerical constraints.
"""

from __future__ import annotations
import re
import uuid
import logging
from typing import List, Dict, Any, Optional

from nlp.preprocessing import sentence_tokenize, extract_numbers, extract_dates

logger = logging.getLogger(__name__)

# ── Modality keyword patterns ─────────────────────────────────────────────────
_OBLIGATION = re.compile(
    r"\b(must|shall|required to|is required|are required|obligated to"
    r"|mandated to|need to|needs to|have to|has to|will be required)\b",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(may|can|are allowed|is allowed|are permitted|is permitted"
    r"|have the right to|has the right to|are entitled to|is entitled)\b",
    re.IGNORECASE,
)
_PROHIBITION = re.compile(
    r"\b(must not|shall not|may not|cannot|is prohibited|are prohibited"
    r"|is not permitted|are not permitted|is not allowed|are not allowed"
    r"|must never|should never|forbidden)\b",
    re.IGNORECASE,
)
_RECOMMENDATION = re.compile(
    r"\b(should|ought to|is recommended|are recommended|it is advisable"
    r"|is encouraged|are encouraged)\b",
    re.IGNORECASE,
)

# ── Actor detection ───────────────────────────────────────────────────────────
_ACTOR_PATTERNS = [
    r"\b(employee[s]?|staff|worker[s]?|personnel)\b",
    r"\b(manager[s]?|supervisor[s]?|department head[s]?|head[s]? of department)\b",
    r"\b(student[s]?|pupil[s]?|learner[s]?)\b",
    r"\b(contractor[s]?|vendor[s]?|supplier[s]?)\b",
    r"\b(administrator[s]?|it team|hr department|finance team)\b",
    r"\b(organization|company|institution)\b",
]
_ACTOR_RE = re.compile("|".join(_ACTOR_PATTERNS), re.IGNORECASE)

# ── Condition / exception patterns ───────────────────────────────────────────
_CONDITION_RE = re.compile(
    r"\b(?:if|when|provided that|subject to|upon|as long as|in cases? where"
    r"|under the condition|contingent on|on the condition that|where)\b",
    re.IGNORECASE,
)
_EXCEPTION_RE = re.compile(
    r"\b(?:except|unless|notwithstanding|with the exception|excluding"
    r"|save for|other than|provided however|however,?)\b",
    re.IGNORECASE,
)

_MIN_SENTENCE_CHARS = 20


def _detect_modality(sentence: str) -> Optional[str]:
    if _PROHIBITION.search(sentence):
        return "prohibition"
    if _OBLIGATION.search(sentence):
        return "obligation"
    if _PERMISSION.search(sentence):
        return "permission"
    if _RECOMMENDATION.search(sentence):
        return "recommendation"
    return None   # factual or unknown


def _extract_actor(sentence: str) -> Optional[str]:
    m = _ACTOR_RE.search(sentence)
    return m.group(0).strip().title() if m else None


def _extract_conditions(sentence: str) -> List[str]:
    """Return clauses that start with a condition keyword."""
    conditions: List[str] = []
    for m in _CONDITION_RE.finditer(sentence):
        # Take up to 120 chars after the keyword
        start = m.start()
        clause = sentence[start:start + 120].strip()
        conditions.append(clause)
    return conditions


def _extract_exceptions(sentence: str) -> List[str]:
    exceptions: List[str] = []
    for m in _EXCEPTION_RE.finditer(sentence):
        start = m.start()
        clause = sentence[start:start + 120].strip()
        exceptions.append(clause)
    return exceptions


def _extract_temporal(sentence: str) -> Dict[str, Any]:
    """Attempt to extract temporal constraints (frequency / deadline)."""
    temporal: Dict[str, Any] = {}
    # Frequency patterns: "three days per week", "annually", "monthly"
    freq_re = re.compile(
        r"(\d+|one|two|three|four|five|six|seven)\s+(day[s]?|week[s]?|month[s]?|year[s]?)"
        r"\s+per\s+(week|month|year|quarter)",
        re.IGNORECASE,
    )
    m = freq_re.search(sentence)
    if m:
        temporal["frequency_text"] = m.group(0)

    # Deadline patterns: "by", "before", "no later than"
    deadline_re = re.compile(
        r"(?:by|before|no later than|prior to)\s+(.{5,60}?)(?:[,;.]|$)",
        re.IGNORECASE,
    )
    m2 = deadline_re.search(sentence)
    if m2:
        temporal["deadline_text"] = m2.group(1).strip()

    # Recurrence: daily / weekly / monthly / annually
    recurrence_re = re.compile(
        r"\b(daily|weekly|monthly|quarterly|annually|bi-annually|per\s+annum)\b",
        re.IGNORECASE,
    )
    m3 = recurrence_re.search(sentence)
    if m3:
        temporal["recurrence"] = m3.group(1).lower()

    return temporal if temporal else {}


def extract_policy_statements(
    chunks: List[Dict[str, Any]],
    version_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract structured policy statements from a list of text chunks.

    Each statement has the structure described in the master prompt §6.
    Only sentences with a detectable modality are returned as policy statements;
    others are still inspected for dates/numbers but not promoted.

    Parameters
    ----------
    chunks     : list of chunk dicts from chunker.chunk_text()
    version_id : document version identifier for traceability

    Returns
    -------
    list of policy statement dicts (not yet persisted — caller saves to DB)
    """
    statements: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        sentences = sentence_tokenize(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < _MIN_SENTENCE_CHARS:
                continue

            modality = _detect_modality(sentence)
            if modality is None:
                continue  # Skip factual sentences for structured representation

            actor = _extract_actor(sentence)
            conditions = _extract_conditions(sentence)
            exceptions = _extract_exceptions(sentence)
            temporal = _extract_temporal(sentence)
            nums = extract_numbers(sentence)

            num_threshold = None
            if nums:
                # Pick the most prominent number
                num_threshold = nums[0]["span"]

            stmt: Dict[str, Any] = {
                "statement_id": f"POL-{uuid.uuid4().hex[:8].upper()}",
                "version_id": version_id,
                "chunk_id": chunk.get("chunk_id"),  # may be None until persisted
                "source_page": chunk.get("page_number"),
                "source_section": chunk.get("section_heading"),
                "original_text": sentence,
                "actor": actor,
                "action": None,          # fine-grained action extraction deferred to Phase 2
                "object": None,
                "modality": modality,
                "conditions": conditions if conditions else [],
                "exceptions": exceptions if exceptions else [],
                "scope": None,
                "temporal_constraints": temporal if temporal else {},
                "numerical_threshold": num_threshold,
                "authority": None,
                "effective_date": None,
                "extraction_method": "rule_based",
                "verification_status": "unverified",
            }
            statements.append(stmt)

    logger.debug("Extracted %d policy statements from %d chunks (version=%s)",
                 len(statements), len(chunks), version_id)
    return statements
