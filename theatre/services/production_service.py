"""Application service coordinating retrieval, generation, and persistence."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from django.conf import settings
from django.db import transaction

from theatre.models import GenerationRun, TheatreProject
from theatre.services.data.schemas import ViewType
from theatre.services.llm import (
    InvalidLLMResponseError,
    LLMConnectionError,
    LLMTimeoutError,
    ModelUnavailableError,
    OllamaClient,
    TheatreGenerator,
)
from theatre.services.rag import (
    COLLECTION_BY_VIEW,
    DEFAULT_RAG_MODE,
    ContextBuilder,
    EmbeddingService,
    QdrantStore,
    RAGMode,
)
from theatre.services.retrieval import BlockingRetriever, LightingRetriever, SceneRetriever
from theatre.services.retrieval.base import RetrievalResult
from theatre.services.validation import ProductionValidationError
from theatre.services.experiment_logging import (
    log_generation_run,
    safe_model_settings,
)

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, user_request: str, **kwargs: Any) -> list[RetrievalResult]: ...


@dataclass(slots=True)
class ProductionDependencies:
    store: QdrantStore
    scene_retriever: Retriever
    blocking_retriever: Retriever
    lighting_retriever: Retriever
    generator: TheatreGenerator


@dataclass(frozen=True, slots=True)
class ProductionOutcome:
    project: TheatreProject
    run: GenerationRun


class ProductionServiceError(RuntimeError):
    """Controlled application error safe to present in the web interface."""

    def __init__(
        self,
        code: str,
        user_message: str,
        *,
        project: TheatreProject | None = None,
    ) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.project = project


class ProductionService:
    def __init__(
        self,
        dependencies: ProductionDependencies,
        *,
        model_name: str,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        self.dependencies = dependencies
        self.model_name = model_name
        self.model_settings = safe_model_settings(model_settings)
        self._research_query = ""
        self._retrieval_config: dict[str, int] = {}
        self._rag_mode = DEFAULT_RAG_MODE

    def generate_production(
        self,
        request_data: dict[str, Any],
        *,
        retrieved_results: tuple[
            list[RetrievalResult], list[RetrievalResult], list[RetrievalResult]
        ] | None = None,
        research_query: str = "",
        retrieval_config: dict[str, int] | None = None,
        rag_mode: RAGMode | str = DEFAULT_RAG_MODE,
    ) -> ProductionOutcome:
        try:
            self._rag_mode = RAGMode(rag_mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported RAG mode: {rag_mode}") from exc
        self._research_query = research_query
        default_config = {
            "scene_top_k": 5,
            "blocking_top_k": 3,
            "lighting_top_k": 3,
            "combined_top_k": 11,
        }
        if retrieved_results is None:
            self._retrieval_config = default_config | dict(retrieval_config or {})
        else:
            scene, blocking, lighting = retrieved_results
            actual_config = {
                "scene_top_k": len(scene),
                "blocking_top_k": len(blocking),
                "lighting_top_k": len(lighting),
                "combined_top_k": len(scene) + len(blocking) + len(lighting),
            }
            self._retrieval_config = actual_config | dict(retrieval_config or {})
        project = self._create_project(request_data)
        started = perf_counter()
        scene_results: list[RetrievalResult] = []
        blocking_results: list[RetrievalResult] = []
        lighting_results: list[RetrievalResult] = []
        try:
            prompt = project.user_prompt
            if retrieved_results is None:
                scene_results, blocking_results, lighting_results = self.retrieve_for_mode(
                    prompt,
                    mode=self._rag_mode,
                    scene_top_k=self._retrieval_config.get("scene_top_k", 5),
                    blocking_top_k=self._retrieval_config.get("blocking_top_k", 3),
                    lighting_top_k=self._retrieval_config.get("lighting_top_k", 3),
                    combined_top_k=self._retrieval_config.get("combined_top_k", 11),
                )
            else:
                scene_results, blocking_results, lighting_results = retrieved_results
            self.require_results_for_mode(
                self._rag_mode, scene_results, blocking_results, lighting_results
            )
            generation = self.dependencies.generator.generate(
                prompt,
                scene_results,
                blocking_results,
                lighting_results,
            )
        except ProductionServiceError as exc:
            self._record_failure(
                project, started, scene_results, blocking_results, lighting_results,
                code=exc.code, message=exc.user_message,
            )
            exc.project = project
            raise
        except (LLMTimeoutError,) as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "ollama_timeout", "The local model timed out. Try a shorter brief or increase the configured timeout.", exc)
        except (ModelUnavailableError,) as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "model_unavailable", f"The configured Qwen model is unavailable. Run: ollama pull {self.model_name}", exc)
        except (LLMConnectionError,) as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "ollama_unavailable", "Ollama is not reachable. Start it with 'ollama serve' and try again.", exc)
        except ProductionValidationError as exc:
            errors = [*exc.initial_errors, *exc.final_errors]
            cause = exc.__cause__
            if isinstance(cause, LLMTimeoutError):
                code = "ollama_timeout"
                user_message = "The local model timed out during its single validation repair attempt."
            elif isinstance(cause, ModelUnavailableError):
                code = "model_unavailable"
                user_message = f"The configured Qwen model is unavailable. Run: ollama pull {self.model_name}"
            elif isinstance(cause, LLMConnectionError):
                code = "ollama_unavailable"
                user_message = "Ollama became unavailable during the validation repair attempt. Start it and try again."
            else:
                code = "validation_failed"
                user_message = "The generated production could not be validated after one repair attempt. No unsafe lighting output was saved."
            self._record_failure(
                project, started, scene_results, blocking_results, lighting_results,
                code=code, message=str(exc),
                validation_errors=errors,
                raw_output=exc.corrected_output or exc.initial_output,
                repair_attempts=1,
            )
            raise ProductionServiceError(
                code,
                user_message,
                project=project,
            ) from exc
        except InvalidLLMResponseError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "malformed_output", "The local model returned an unreadable response. Please try again.", exc)
        except Exception as exc:
            logger.error(
                "Unexpected production pipeline failure project=%s error_type=%s",
                project.pk,
                type(exc).__name__,
            )
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "pipeline_unavailable", "The local generation pipeline is unavailable. Check the index and local model services.", exc)

        elapsed = perf_counter() - started
        with transaction.atomic():
            project.generated_json = generation.production.model_dump(mode="json", by_alias=True)
            project.save(update_fields=("generated_json", "updated_at"))
            run = GenerationRun.objects.create(
                project=project,
                model_name=self.model_name,
                model_settings=self.model_settings,
                user_input=project.user_prompt,
                scene_sources=self._source_ids(scene_results),
                blocking_sources=self._source_ids(blocking_results),
                lighting_sources=self._source_ids(lighting_results),
                retrieval_trace=self._trace_payload(
                    generation.retrieval_trace,
                    [*scene_results, *blocking_results, *lighting_results],
                ),
                raw_output=generation.raw_output,
                generated_json=generation.production.model_dump(mode="json", by_alias=True),
                validated=True,
                validation_errors=generation.validation_errors,
                repair_attempts=1 if generation.repaired else 0,
                generation_time_seconds=elapsed,
                research_query=self._research_query,
                retrieval_config=self._retrieval_config,
                rag_mode=self._rag_mode.value,
            )
        log_generation_run(run)
        logger.info("Generated and saved project %s in %.3f seconds", project.pk, elapsed)
        return ProductionOutcome(project=project, run=run)

    def retrieve_sources(
        self,
        query: str,
        *,
        scene_top_k: int,
        blocking_top_k: int,
        lighting_top_k: int,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], list[RetrievalResult]]:
        self._require_index(frozenset(ViewType))
        try:
            return (
                self.dependencies.scene_retriever.retrieve(query, limit=scene_top_k),
                self.dependencies.blocking_retriever.retrieve(query, limit=blocking_top_k),
                self.dependencies.lighting_retriever.retrieve(query, limit=lighting_top_k),
            )
        except Exception as exc:
            raise ProductionServiceError(
                "qdrant_unavailable",
                "The local retrieval index could not be queried. Ensure Qdrant storage is available and rebuild the index if necessary.",
            ) from exc

    def retrieve_for_mode(
        self,
        query: str,
        *,
        mode: RAGMode | str,
        scene_top_k: int,
        blocking_top_k: int,
        lighting_top_k: int,
        combined_top_k: int = 11,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], list[RetrievalResult]]:
        """Retrieve evidence for one ablation mode using the shared retrievers."""
        selected_mode = RAGMode(mode)
        if selected_mode is RAGMode.NO_RAG:
            return [], [], []
        self._require_index(selected_mode.active_views)
        try:
            if selected_mode is RAGMode.SINGLE_COMBINED:
                if combined_top_k < 1:
                    raise ValueError("Combined Top-K must be positive")
                candidates = [
                    *self.dependencies.scene_retriever.retrieve(
                        query, limit=combined_top_k, query_text=query
                    ),
                    *self.dependencies.blocking_retriever.retrieve(
                        query, limit=combined_top_k, query_text=query
                    ),
                    *self.dependencies.lighting_retriever.retrieve(
                        query, limit=combined_top_k, query_text=query
                    ),
                ]
                combined = sorted(candidates, key=lambda item: (-item.score, item.rank))[
                    :combined_top_k
                ]
                reranked = [
                    item.model_copy(update={"rank": rank})
                    for rank, item in enumerate(combined, start=1)
                ]
                return (
                    [item for item in reranked if item.view_type is ViewType.SCENE],
                    [item for item in reranked if item.view_type is ViewType.BLOCKING],
                    [item for item in reranked if item.view_type is ViewType.LIGHTING],
                )

            scene = (
                self.dependencies.scene_retriever.retrieve(query, limit=scene_top_k)
                if ViewType.SCENE in selected_mode.active_views else []
            )
            blocking = (
                self.dependencies.blocking_retriever.retrieve(query, limit=blocking_top_k)
                if ViewType.BLOCKING in selected_mode.active_views else []
            )
            lighting = (
                self.dependencies.lighting_retriever.retrieve(query, limit=lighting_top_k)
                if ViewType.LIGHTING in selected_mode.active_views else []
            )
            return scene, blocking, lighting
        except ProductionServiceError:
            raise
        except Exception as exc:
            raise ProductionServiceError(
                "qdrant_unavailable",
                "The local retrieval index could not be queried. Ensure Qdrant storage is available and rebuild the index if necessary.",
            ) from exc

    def _create_project(self, data: dict[str, Any]) -> TheatreProject:
        story = str(data["story_idea"]).strip()
        complete_prompt = "\n".join(
            (story, f"Scene time: {data['scene_time']}", f"Desired emotion: {data['desired_emotion']}")
        )
        lights = data["available_lights"]
        if isinstance(lights, str):
            lights = [item.strip() for item in lights.replace("\n", ",").split(",") if item.strip()]
        return TheatreProject.objects.create(
            title=story.splitlines()[0][:255], user_prompt=complete_prompt,
            language=data["language"], genre=data["genre"], theme=data["theme"],
            actor_count=data["actor_count"], duration_minutes=data["duration_minutes"],
            stage_size=data["stage_size"], available_lights=list(lights),
        )

    def _require_index(self, active_views: frozenset[ViewType]) -> None:
        try:
            missing = [
                COLLECTION_BY_VIEW[view]
                for view in active_views
                if not self.dependencies.store.client.collection_exists(COLLECTION_BY_VIEW[view])
            ]
        except Exception as exc:
            raise ProductionServiceError("qdrant_unavailable", "The local Qdrant store is unavailable. Close other index processes and try again.") from exc
        if missing:
            raise ProductionServiceError("index_not_built", "The theatre dataset has not been indexed. Run 'python manage.py build_rag_index' first.")

    @staticmethod
    def require_results_for_mode(
        mode: RAGMode,
        scene: list[RetrievalResult],
        blocking: list[RetrievalResult],
        lighting: list[RetrievalResult],
    ) -> None:
        if mode is RAGMode.NO_RAG:
            return
        by_view = {
            ViewType.SCENE: scene,
            ViewType.BLOCKING: blocking,
            ViewType.LIGHTING: lighting,
        }
        if mode is RAGMode.SINGLE_COMBINED:
            if any(by_view.values()):
                return
            empty = ["combined"]
        else:
            empty = [view.value for view in mode.active_views if not by_view[view]]
        if empty:
            raise ProductionServiceError(
                "empty_retrieval",
                f"No {'/'.join(empty)} references were found. Rebuild the RAG index and try again.",
            )

    def _fail(self, project: TheatreProject, started: float, scene: list[RetrievalResult],
              blocking: list[RetrievalResult], lighting: list[RetrievalResult],
              code: str, message: str, exception: Exception) -> None:
        logger.warning(
            "Production pipeline error code=%s project=%s error_type=%s",
            code, project.pk, type(exception).__name__,
        )
        self._record_failure(
            project, started, scene, blocking, lighting, code=code, message=message
        )
        raise ProductionServiceError(code, message, project=project) from exception

    def _record_failure(self, project: TheatreProject, started: float,
                        scene: list[RetrievalResult], blocking: list[RetrievalResult],
                        lighting: list[RetrievalResult], *, code: str, message: str,
                        validation_errors: list[dict[str, Any]] | None = None,
                        raw_output: str = "", repair_attempts: int = 0) -> GenerationRun:
        trace = [
            {
                "source_id": item.source_id,
                "rank": item.rank,
                "score": item.score,
                "view_type": item.view_type.value,
                "metadata": {
                    key: item.metadata[key]
                    for key in ("theme", "scene_type")
                    if key in item.metadata
                },
            }
            for item in [*scene, *blocking, *lighting]
        ]
        errors = validation_errors or [{"code": code, "message": message}]
        run = GenerationRun.objects.create(
            project=project, model_name=self.model_name,
            model_settings=self.model_settings, user_input=project.user_prompt,
            scene_sources=self._source_ids(scene), blocking_sources=self._source_ids(blocking),
            lighting_sources=self._source_ids(lighting), retrieval_trace=trace,
            raw_output=raw_output, validated=False, validation_errors=errors,
            repair_attempts=repair_attempts,
            generation_time_seconds=perf_counter() - started,
            research_query=self._research_query,
            retrieval_config=self._retrieval_config,
            rag_mode=self._rag_mode.value,
        )
        log_generation_run(run)
        return run

    @staticmethod
    def _source_ids(results: list[RetrievalResult]) -> list[str]:
        return [result.source_id for result in results]

    @staticmethod
    def _trace_payload(trace: list[Any], results: list[RetrievalResult]) -> list[dict[str, Any]]:
        evidence = {(item.view_type, item.source_id): item for item in results}
        payload: list[dict[str, Any]] = []
        for item in trace:
            row = item.model_dump(mode="json")
            result = evidence.get((item.view_type, item.source_id))
            row["metadata"] = {
                key: result.metadata[key]
                for key in ("theme", "scene_type")
                if result is not None and key in result.metadata
            }
            payload.append(row)
        return payload


def build_default_service() -> ProductionService:
    embedder = EmbeddingService(settings.EMBEDDING_MODEL_NAME, device=settings.EMBEDDING_DEVICE,
                                batch_size=settings.EMBEDDING_BATCH_SIZE)
    store = QdrantStore(settings.QDRANT_PATH)
    client = OllamaClient(settings.THETRESTAGEAI_OLLAMA_URL, settings.THETRESTAGEAI_LLM_MODEL,
                          timeout_seconds=settings.THETRESTAGEAI_LLM_TIMEOUT_SECONDS)
    context_builder = ContextBuilder(max_chars=settings.RAG_CONTEXT_MAX_CHARS)
    dependencies = ProductionDependencies(
        store=store,
        scene_retriever=SceneRetriever(embedder, store),
        blocking_retriever=BlockingRetriever(embedder, store),
        lighting_retriever=LightingRetriever(embedder, store),
        generator=TheatreGenerator(client, context_builder),
    )
    return ProductionService(
        dependencies,
        model_name=settings.THETRESTAGEAI_LLM_MODEL,
        model_settings=client.reproducibility_settings(),
    )


def generate_production(request_data: dict[str, Any]) -> ProductionOutcome:
    """Public application entry point used by Django and future API adapters."""
    service = build_default_service()
    try:
        return service.generate_production(request_data)
    finally:
        service.dependencies.store.close()
