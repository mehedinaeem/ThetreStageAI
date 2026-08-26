"""Dataset-level validation and statistics aggregation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .loader import load_original_records, load_retrieval_records
from .schemas import DatasetInspection, DatasetStatistics, ViewType

logger = logging.getLogger(__name__)

ORIGINAL_FILENAME = "bangla_natok_500.jsonl"
VIEW_FILENAMES = (
    "scene_view.jsonl",
    "blocking_view.jsonl",
    "lighting_view.jsonl",
)


def inspect_dataset(dataset_path: str | Path) -> DatasetInspection:
    """Validate canonical source files and return records plus aggregate statistics."""
    root = Path(dataset_path).expanduser().resolve()
    statistics = DatasetStatistics()
    originals, errors = load_original_records(root / ORIGINAL_FILENAME)

    missing_counts = {"search_text": 0, "metadata": 0, "payload": 0}

    def count_missing(record: dict[str, Any]) -> None:
        for field in missing_counts:
            if field not in record or record[field] is None or record[field] == "":
                missing_counts[field] += 1

    retrieval_documents = []
    for filename in VIEW_FILENAMES:
        records, file_errors = load_retrieval_records(
            root / "retrieval_views" / filename,
            observe=count_missing,
        )
        retrieval_documents.extend(records)
        errors.extend(file_errors)

    statistics.original_records = len(originals)
    statistics.scene_records = sum(
        record.view_type is ViewType.SCENE for record in retrieval_documents
    )
    statistics.blocking_records = sum(
        record.view_type is ViewType.BLOCKING for record in retrieval_documents
    )
    statistics.lighting_records = sum(
        record.view_type is ViewType.LIGHTING for record in retrieval_documents
    )
    statistics.malformed_records = len(errors)
    statistics.missing_search_text = missing_counts["search_text"]
    statistics.missing_metadata = missing_counts["metadata"]
    statistics.missing_payload = missing_counts["payload"]

    logger.info(
        "Dataset inspected at %s: %d originals, %d retrieval documents, %d errors",
        root,
        statistics.original_records,
        statistics.retrieval_records,
        statistics.malformed_records,
    )
    return DatasetInspection(
        originals=originals,
        retrieval_documents=retrieval_documents,
        errors=errors,
        statistics=statistics,
    )
