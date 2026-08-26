"""Safe, read-only JSONL loaders for theatre dataset records."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .schemas import MalformedRecord, OriginalTheatreRecord, RetrievalViewDocument, ViewType

logger = logging.getLogger(__name__)
MAX_JSONL_LINE_BYTES = 2_000_000

RecordT = TypeVar("RecordT", bound=BaseModel)
ErrorObserver = Callable[[dict[str, Any]], None]


def _read_jsonl(
    path: Path,
    model: type[RecordT],
    *,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    observe: ErrorObserver | None = None,
) -> tuple[list[RecordT], list[MalformedRecord]]:
    """Read UTF-8 JSONL without writing to or normalizing the source file."""
    records: list[RecordT] = []
    errors: list[MalformedRecord] = []

    try:
        source = path.open("rb")
    except OSError as exc:
        error = MalformedRecord(path=path, line_number=0, reason=str(exc))
        logger.error("Cannot read dataset file %s: %s", path, exc)
        return records, [error]

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                if len(raw_line) > MAX_JSONL_LINE_BYTES:
                    raise ValueError(
                        f"JSONL record exceeds {MAX_JSONL_LINE_BYTES} bytes"
                    )
                line = raw_line.decode("utf-8")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSON value must be an object")
                if observe is not None:
                    observe(value)
                if transform is not None:
                    value = transform(value)
                records.append(model.model_validate(value))
            except (json.JSONDecodeError, UnicodeError, ValidationError, ValueError) as exc:
                error = MalformedRecord(path=path, line_number=line_number, reason=str(exc))
                errors.append(error)
                logger.warning("Skipping malformed record in %s:%d: %s", path, line_number, exc)

    return records, errors


def detect_view_type(record: dict[str, Any], source_path: Path | None = None) -> ViewType:
    """Detect a retrieval view from its field, metadata, identifier, or filename."""
    candidates = [
        record.get("view_type"),
        (record.get("metadata") or {}).get("document_type")
        if isinstance(record.get("metadata"), dict)
        else None,
        record.get("id"),
        source_path.stem if source_path else None,
    ]
    for candidate in candidates:
        normalized = str(candidate or "").lower()
        for view_type in ViewType:
            if view_type.value in normalized:
                return view_type
    raise ValueError("Unable to detect retrieval view type")


def load_original_records(
    path: str | Path,
) -> tuple[list[OriginalTheatreRecord], list[MalformedRecord]]:
    return _read_jsonl(Path(path), OriginalTheatreRecord)


def load_retrieval_records(
    path: str | Path,
    *,
    observe: ErrorObserver | None = None,
) -> tuple[list[RetrievalViewDocument], list[MalformedRecord]]:
    source_path = Path(path)

    def add_detected_view(record: dict[str, Any]) -> dict[str, Any]:
        if not record.get("view_type"):
            record = dict(record)
            record["view_type"] = detect_view_type(record, source_path).value
        return record

    return _read_jsonl(
        source_path,
        RetrievalViewDocument,
        transform=add_detected_view,
        observe=observe,
    )
