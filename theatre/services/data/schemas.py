"""Typed schemas for the source dataset and inspection results."""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ViewType(StrEnum):
    SCENE = "scene"
    BLOCKING = "blocking"
    LIGHTING = "lighting"


class OriginalTheatreRecord(BaseModel):
    """An original play record, preserving optional research fields as supplied."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    theme: str | None = None
    genre: str | None = None
    scene_type: str | None = None
    location: str | None = None
    time: str | None = None
    emotion: dict[str, Any] | str | None = None
    characters: list[str] = Field(default_factory=list)
    dialogue: list[dict[str, Any]] = Field(default_factory=list)
    stage_directions: list[str] = Field(default_factory=list)
    blocking: list[dict[str, Any]] = Field(default_factory=list)
    lighting: list[dict[str, Any]] = Field(default_factory=list)
    sound: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None


class RetrievalViewDocument(BaseModel):
    """A typed retrieval representation; no embedding is created here."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    view_type: ViewType
    source_id: str = Field(min_length=1)
    search_text: str = Field(min_length=1)
    metadata: dict[str, Any]
    payload: dict[str, Any]


class MalformedRecord(BaseModel):
    path: Path
    line_number: int
    reason: str


class DatasetStatistics(BaseModel):
    original_records: int = 0
    scene_records: int = 0
    blocking_records: int = 0
    lighting_records: int = 0
    malformed_records: int = 0
    missing_search_text: int = 0
    missing_metadata: int = 0
    missing_payload: int = 0

    @property
    def retrieval_records(self) -> int:
        return self.scene_records + self.blocking_records + self.lighting_records


class DatasetInspection(BaseModel):
    originals: list[OriginalTheatreRecord] = Field(default_factory=list)
    retrieval_documents: list[RetrievalViewDocument] = Field(default_factory=list)
    errors: list[MalformedRecord] = Field(default_factory=list)
    statistics: DatasetStatistics = Field(default_factory=DatasetStatistics)
