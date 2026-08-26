"""Exports for machine analysis and blinded human theatre evaluation."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from .experiment_runner import ExperimentRecord

EXPERT_EVALUATION_COLUMNS = (
    "experiment_id",
    "project_id",
    "system",
    "script_coherence",
    "character_consistency",
    "dialogue_quality",
    "stageability",
    "blocking_quality",
    "lighting_quality",
    "cultural_relevance",
    "production_feasibility",
    "overall_score",
    "comments",
)

RATING_COLUMNS = EXPERT_EVALUATION_COLUMNS[3:-1]


def export_expert_evaluation_csv(
    records: Iterable[ExperimentRecord], path: str | Path
) -> Path:
    """Create an expert-rating sheet with every subjective field intentionally empty."""
    destination = _prepare(path)
    with destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=EXPERT_EVALUATION_COLUMNS)
        writer.writeheader()
        for record in records:
            row: dict[str, str | int] = {
                "experiment_id": record.experiment_id,
                "project_id": record.project_id if record.project_id is not None else "",
                "system": record.system.value,
                "comments": "",
            }
            row.update({column: "" for column in RATING_COLUMNS})
            writer.writerow(row)
    return destination


def export_experiment_csv(records: Iterable[ExperimentRecord], path: str | Path) -> Path:
    """Export objective experiment metadata; JSON structures remain lossless strings."""
    destination = _prepare(path)
    fieldnames = tuple(ExperimentRecord.model_fields)
    with destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            for field, value in row.items():
                if isinstance(value, (list, dict)):
                    row[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(row)
    return destination


def _prepare(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
