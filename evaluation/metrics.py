"""
Evaluation metrics — precision, recall, F1, and entity matching.
"""
from __future__ import annotations

import re


def extract_entity_ids(text: str) -> set[str]:
    """
    Extract entity IDs (both old format like B003/D002 and new format like BAT_HS_2026_001/SUP_SHAKTI)
    from a text string.
    """
    pattern = r"\b(?:[A-Z]{2,10}_[A-Z0-9_]+|[A-Z]{1,3}\d{3,})\b"
    return set(re.findall(pattern, text))


def compute_entity_recall(
    expected_ids: list[str],
    actual_text: str,
) -> float:
    """
    Recall = |expected ∩ found| / |expected|

    Checks how many of the expected entity IDs appear in the actual answer.
    """
    if not expected_ids:
        return 1.0
    expected = set(expected_ids)
    found = extract_entity_ids(actual_text)
    matched = expected & found
    return len(matched) / len(expected)


def compute_precision(
    expected_ids: list[str],
    actual_text: str,
) -> float:
    """
    Precision = |expected ∩ found| / |found|
    """
    expected = set(expected_ids)
    found = extract_entity_ids(actual_text)
    if not found:
        return 0.0
    return len(expected & found) / len(found)


def compute_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_correct(
    expected_ids: list[str],
    actual_text: str,
    threshold: float = 0.5,
) -> bool:
    """Binary correctness: recall above threshold."""
    return compute_entity_recall(expected_ids, actual_text) >= threshold
