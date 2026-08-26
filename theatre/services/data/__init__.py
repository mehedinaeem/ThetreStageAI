"""Read-only loading and validation for theatre research datasets."""

from .loader import load_original_records, load_retrieval_records
from .schemas import (
    DatasetInspection,
    DatasetStatistics,
    OriginalTheatreRecord,
    RetrievalViewDocument,
    ViewType,
)
from .validator import inspect_dataset

__all__ = [
    "DatasetInspection",
    "DatasetStatistics",
    "OriginalTheatreRecord",
    "RetrievalViewDocument",
    "ViewType",
    "inspect_dataset",
    "load_original_records",
    "load_retrieval_records",
]
