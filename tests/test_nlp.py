"""
AI-COS Tests — NLP Preprocessing & Policy Parsing
Tests linguistic analysis, entity recognition, and rule-based policy extraction.
"""

import pytest
from nlp.preprocessing import sentence_tokenize, extract_dates, extract_numbers
from nlp.policy_parser import extract_policy_statements, _detect_modality, _extract_actor


def test_sentence_tokenization():
    text = "All staff must wear badges. Visitors may obtain temporary passes at reception."
    sents = sentence_tokenize(text)
    assert len(sents) == 2
    assert "badges" in sents[0]
    assert "reception" in sents[1]


def test_extract_dates_and_numbers():
    text = "The policy takes effect on 2026-01-01. Reimbursements are capped at 2500 dollars within 30 days."
    dates = extract_dates(text)
    assert any("2026-01-01" in d for d in dates)

    numbers = extract_numbers(text)
    assert any(n["value"] == 2500.0 for n in numbers)
    assert any(n["value"] == 30.0 for n in numbers)


def test_modality_detection():
    assert _detect_modality("Employees must report security incidents immediately.") == "obligation"
    assert _detect_modality("Staff are permitted to work remotely on Fridays.") == "permission"
    assert _detect_modality("Employees must not store confidential files on personal thumb drives.") == "prohibition"
    assert _detect_modality("Managers should review timesheets weekly.") == "recommendation"
    assert _detect_modality("This section describes organizational overview.") is None


def test_actor_detection():
    actor = _extract_actor("Employees must submit expense claims by month-end.")
    assert actor is not None
    assert actor.lower() in ["employee", "employees"]

    actor2 = _extract_actor("Department heads shall approve all travel requests.")
    assert actor2 is not None
    assert "head" in actor2.lower() or "department" in actor2.lower()


def test_extract_policy_statements():
    chunk_text = (
        "Eligible employees may work remotely up to 3 days per week with manager approval. "
        "All employees must complete data protection training annually. "
        "Personnel must not connect unauthorized hardware to the internal corporate network."
    )
    chunks = [{"chunk_id": "c1", "text": chunk_text, "page_number": 1, "section_heading": "General"}]
    statements = extract_policy_statements(chunks, version_id="v1")
    assert len(statements) >= 3

    modalities = [s["modality"] for s in statements]
    assert "permission" in modalities
    assert "obligation" in modalities
    assert "prohibition" in modalities

    perm = next(s for s in statements if s["modality"] == "permission")
    assert perm["actor"] is not None
    assert "remote" in perm["original_text"].lower()
