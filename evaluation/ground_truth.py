"""
Ground truth loader — validates and exposes the answers.json dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from settings import settings


def load_ground_truth(path: Path | None = None) -> dict[str, Any]:
    """Load and return the ground truth answers.json."""
    gt_path = path or Path(settings.ground_truth_path)
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


def get_breaking_point_query(gt: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical breaking-point query and expected answer."""
    return gt["breaking_point_query"]


def get_supporting_queries(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all supporting queries."""
    return gt.get("supporting_queries", [])
