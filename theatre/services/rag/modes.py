"""Research RAG-mode definitions and retrieval-view activation rules."""
from __future__ import annotations

from enum import StrEnum

from theatre.services.data.schemas import ViewType


class RAGMode(StrEnum):
    NO_RAG = "no_rag"
    SCENE_ONLY = "scene_only"
    SCENE_BLOCKING = "scene_blocking"
    SCENE_LIGHTING = "scene_lighting"
    SINGLE_COMBINED = "single_combined"
    FULL_MULTIVIEW = "full_multiview"

    @property
    def label(self) -> str:
        return {
            self.NO_RAG: "Mode 1 — No RAG",
            self.SCENE_ONLY: "Mode 2 — Scene-only RAG",
            self.SCENE_BLOCKING: "Mode 3 — Scene + Blocking RAG",
            self.SCENE_LIGHTING: "Mode 4 — Scene + Lighting RAG",
            self.SINGLE_COMBINED: "Mode 5 — Single combined retrieval RAG",
            self.FULL_MULTIVIEW: "Mode 6 — Full Multi-View RAG",
        }[self]

    @property
    def active_views(self) -> frozenset[ViewType]:
        if self is self.NO_RAG:
            return frozenset()
        if self is self.SCENE_ONLY:
            return frozenset({ViewType.SCENE})
        if self is self.SCENE_BLOCKING:
            return frozenset({ViewType.SCENE, ViewType.BLOCKING})
        if self is self.SCENE_LIGHTING:
            return frozenset({ViewType.SCENE, ViewType.LIGHTING})
        return frozenset(ViewType)

    @classmethod
    def choices(cls) -> tuple[tuple[str, str], ...]:
        return tuple((mode.value, mode.label) for mode in cls)


DEFAULT_RAG_MODE = RAGMode.FULL_MULTIVIEW
