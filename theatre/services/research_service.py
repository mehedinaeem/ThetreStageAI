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
from theatre.services.retrieval.base import RetrievalResult

SELECTION_SALT = "thetrestageai.research.rag.selection.v1"


@dataclass(frozen=True, slots=True)
class ResearchRetrieval:
    query: str
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
) -> ResearchRetrieval:
    service = build_default_service()
    try:
        results = service.retrieve_sources(
            query,
            scene_top_k=scene_top_k,
            blocking_top_k=blocking_top_k,
            lighting_top_k=lighting_top_k,
        )
    finally:
        service.dependencies.store.close()
    scene, blocking, lighting = results
    empty = [
        name
        for name, values in (("scene", scene), ("blocking", blocking), ("lighting", lighting))
        if not values
    ]
    if empty:
        raise ProductionServiceError(
            "empty_retrieval",
            f"No {'/'.join(empty)} references were found. Rebuild the RAG index and try again.",
        )
    top_k = {
        "scene_top_k": scene_top_k,
        "blocking_top_k": blocking_top_k,
        "lighting_top_k": lighting_top_k,
    }
    selection_token = signing.dumps(
        {
            "query": query,
            "top_k": top_k,
            "scene": [item.model_dump(mode="json") for item in scene],
            "blocking": [item.model_dump(mode="json") for item in blocking],
            "lighting": [item.model_dump(mode="json") for item in lighting],
        },
        key=None,
        salt=SELECTION_SALT,
        compress=True,
    )
    return ResearchRetrieval(query, scene, blocking, lighting, top_k, selection_token)


def generate_from_research_selection(token: str, *, max_age: int = 3_600) -> ProductionOutcome:
    data = signing.loads(token, salt=SELECTION_SALT, max_age=max_age)
    if not isinstance(data, dict):
        raise signing.BadSignature("Invalid research selection")
    query = str(data.get("query", "")).strip()
    top_k = data.get("top_k")
    if not query or not isinstance(top_k, dict):
        raise signing.BadSignature("Incomplete research selection")
    try:
        scene = [RetrievalResult.model_validate(item) for item in data["scene"]]
        blocking = [RetrievalResult.model_validate(item) for item in data["blocking"]]
        lighting = [RetrievalResult.model_validate(item) for item in data["lighting"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise signing.BadSignature("Malformed research selection") from exc

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
            retrieval_config={key: int(value) for key, value in top_k.items()},
        )
    finally:
        service.dependencies.store.close()
