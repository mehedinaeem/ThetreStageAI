"""Signed, reproducible retrieval inspection and generation workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.core import signing

from theatre.services.production_service import (
    ProductionOutcome,
    ProductionServiceError,
    build_default_service,
)
from theatre.services.rag.modes import RAGMode
from theatre.services.retrieval.base import RetrievalResult

SELECTION_SALT = "thetrestageai.research.rag.selection.v1"
MAX_SELECTION_TOKEN_CHARS = 500_000


@dataclass(frozen=True, slots=True)
class ResearchRetrieval:
    query: str
    rag_mode: RAGMode
    scene_results: list[RetrievalResult]
    blocking_results: list[RetrievalResult]
    lighting_results: list[RetrievalResult]
    top_k: dict[str, int]
    selection_token: str


def retrieve_for_research(
    query: str,
    *,
    scene_top_k: int,
    blocking_top_k: int,
    lighting_top_k: int,
    combined_top_k: int = 11,
    rag_mode: RAGMode | str = RAGMode.FULL_MULTIVIEW,
) -> ResearchRetrieval:
    query = query.strip()
    if not query or len(query) > 2_000:
        raise ValueError("Research query must contain 1 to 2000 characters")
    limits = (scene_top_k, blocking_top_k, lighting_top_k, combined_top_k)
    if any(value < 1 or value > 50 for value in limits):
        raise ValueError("Research Top-K values must be between 1 and 50")
    selected_mode = RAGMode(rag_mode)
    service = build_default_service()
    try:
        results = service.retrieve_for_mode(
            query,
            mode=selected_mode,
            scene_top_k=scene_top_k,
            blocking_top_k=blocking_top_k,
            lighting_top_k=lighting_top_k,
            combined_top_k=combined_top_k,
        )
    finally:
        service.dependencies.store.close()
    scene, blocking, lighting = results
    service.require_results_for_mode(selected_mode, scene, blocking, lighting)
    top_k = {
        "scene_top_k": scene_top_k,
        "blocking_top_k": blocking_top_k,
        "lighting_top_k": lighting_top_k,
        "combined_top_k": combined_top_k,
    }
    selection_token = signing.dumps(
        {
            "query": query,
            "rag_mode": selected_mode.value,
            "top_k": top_k,
            "scene": [item.model_dump(mode="json") for item in scene],
            "blocking": [item.model_dump(mode="json") for item in blocking],
            "lighting": [item.model_dump(mode="json") for item in lighting],
        },
        key=None,
        salt=SELECTION_SALT,
        compress=True,
    )
    if len(selection_token) > MAX_SELECTION_TOKEN_CHARS:
        raise ProductionServiceError(
            "selection_too_large",
            "The retrieved research selection is too large to preserve safely. Reduce Top-K and try again.",
        )
    return ResearchRetrieval(
        query, selected_mode, scene, blocking, lighting, top_k, selection_token
    )


def generate_from_research_selection(token: str, *, max_age: int = 3_600) -> ProductionOutcome:
    if not token or len(token) > MAX_SELECTION_TOKEN_CHARS:
        raise signing.BadSignature("Invalid research selection size")
    data = signing.loads(token, salt=SELECTION_SALT, max_age=max_age)
    if not isinstance(data, dict):
        raise signing.BadSignature("Invalid research selection")
    query = str(data.get("query", "")).strip()
    top_k = data.get("top_k")
    try:
        rag_mode = RAGMode(data.get("rag_mode"))
    except (TypeError, ValueError) as exc:
        raise signing.BadSignature("Invalid RAG mode") from exc
    if not query or not isinstance(top_k, dict):
        raise signing.BadSignature("Incomplete research selection")
    try:
        scene = [RetrievalResult.model_validate(item) for item in data["scene"]]
        blocking = [RetrievalResult.model_validate(item) for item in data["blocking"]]
        lighting = [RetrievalResult.model_validate(item) for item in data["lighting"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise signing.BadSignature("Malformed research selection") from exc
    if len(query) > 2_000 or any(len(items) > 50 for items in (scene, blocking, lighting)):
        raise signing.BadSignature("Research selection exceeds safe limits")
    try:
        clean_top_k = {key: int(value) for key, value in top_k.items()}
    except (TypeError, ValueError) as exc:
        raise signing.BadSignature("Malformed Top-K configuration") from exc
    expected_keys = {
        "scene_top_k", "blocking_top_k", "lighting_top_k", "combined_top_k"
    }
    if set(clean_top_k) != expected_keys or any(
        value < 1 or value > 50 for value in clean_top_k.values()
    ):
        raise signing.BadSignature("Invalid Top-K configuration")

    request_data: dict[str, Any] = {
        "story_idea": query,
        "theme": "research_query",
        "genre": "research_prototype",
        "language": "bn",
        "actor_count": 2,
        "duration_minutes": 15,
        "stage_size": "research",
        "available_lights": [],
        "scene_time": "unspecified",
        "desired_emotion": "specified in research query",
    }
    service = build_default_service()
    try:
        return service.generate_production(
            request_data,
            retrieved_results=(scene, blocking, lighting),
            research_query=query,
            retrieval_config=clean_top_k,
            rag_mode=rag_mode,
        )
    finally:
        service.dependencies.store.close()
