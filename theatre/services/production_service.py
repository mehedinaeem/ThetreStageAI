"""Application service coordinating retrieval, generation, and persistence."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Protocol

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import transaction

from theatre.models import GenerationRun, TheatreProject
from theatre.services.data.schemas import ViewType
from theatre.services.llm import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiBadRequestError,
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiUnavailableError,
    InvalidLLMResponseError,
    LLMConnectionError,
    LLMTimeoutError,
    ModelUnavailableError,
    TheatreGenerator,
    create_provider,
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
from theatre.services.validation import ProductionValidationError, make_json_safe
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
            "scene_top_k": 3,
            "blocking_top_k": 2,
            "lighting_top_k": 2,
            "combined_top_k": 7,
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
                    scene_top_k=self._retrieval_config.get("scene_top_k", 3),
                    blocking_top_k=self._retrieval_config.get("blocking_top_k", 2),
                    lighting_top_k=self._retrieval_config.get("lighting_top_k", 2),
                    combined_top_k=self._retrieval_config.get("combined_top_k", 7),
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
                constraints={
                    "story_idea": request_data.get("story_idea"),
                    "theme": project.theme,
                    "genre": project.genre,
                    "language": project.language,
                    "actor_count": project.actor_count,
                    "duration_minutes": project.duration_minutes,
                    "stage_size": project.stage_size,
                    "available_lights": project.available_lights,
                    "scene_time": request_data.get("scene_time"),
                    "desired_emotion": request_data.get("desired_emotion"),
                },
            )
        except ProductionServiceError as exc:
            self._record_failure(
                project, started, scene_results, blocking_results, lighting_results,
                code=exc.code, message=exc.user_message,
            )
            exc.project = project
            raise
        except (LLMTimeoutError,) as exc:
            if self.model_settings.get("provider") == "gemini":
                code = "gemini_timeout"
                message = "Gemini timed out. Please retry the request."
            else:
                code = "ollama_timeout"
                message = "The local model timed out. Try a shorter brief or increase the configured timeout."
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       code, message, exc)
        except (ModelUnavailableError,) as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "model_unavailable", f"The configured Qwen model is unavailable. Run: ollama pull {self.model_name}", exc)
        except (LLMConnectionError,) as exc:
            if isinstance(exc, GeminiNetworkError):
                code = "gemini_network_error"
                message = "Gemini could not be reached because of a network transport error."
            else:
                code = "ollama_unavailable"
                message = "Ollama is not reachable. Start it with 'ollama serve' and try again."
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       code, message, exc)
        except GeminiConfigurationError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_configuration_error", "Gemini is not configured. Add GEMINI_API_KEY to the local .env file.", exc)
        except GeminiAuthenticationError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_auth_error", "Gemini rejected the configured API key. Check the local Gemini configuration.", exc)
        except GeminiRateLimitError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_rate_limit", "The Gemini free-tier request limit was reached. Please retry later.", exc)
        except GeminiBadRequestError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_bad_request", "Gemini rejected the structured generation request. Please check the generation schema.", exc)
        except GeminiUnavailableError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_unavailable", "Gemini is temporarily unavailable. Please retry later.", exc)
        except GeminiInvalidResponseError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_invalid_response", "Gemini returned an unreadable structured response. Please retry.", exc)
        except GeminiAPIError as exc:
            self._fail(project, started, scene_results, blocking_results, lighting_results,
                       "gemini_api_error", "Gemini is temporarily unavailable. Please retry later.", exc)
        except ProductionValidationError as exc:
            errors = exc.final_errors or exc.initial_errors
            cause = exc.__cause__
            if isinstance(cause, LLMTimeoutError):
                if self.model_settings.get("provider") == "gemini":
                    code = "gemini_timeout"
                    user_message = "Gemini timed out during its single validation repair attempt."
                else:
                    code = "ollama_timeout"
                    user_message = "The local model timed out during its single validation repair attempt."
            elif isinstance(cause, GeminiAuthenticationError):
                code = "gemini_auth_error"
                user_message = "Gemini rejected the configured API key during repair."
            elif isinstance(cause, GeminiRateLimitError):
                code = "gemini_rate_limit"
                user_message = "The Gemini free-tier request limit was reached during repair. Please retry later."
            elif isinstance(cause, GeminiBadRequestError):
                code = "gemini_bad_request"
                user_message = "Gemini rejected the structured validation repair request."
            elif isinstance(cause, GeminiUnavailableError):
                code = "gemini_unavailable"
                user_message = "Gemini became temporarily unavailable during repair."
            elif isinstance(cause, GeminiNetworkError):
                code = "gemini_network_error"
                user_message = "Gemini encountered a network transport error during repair."
            elif isinstance(cause, GeminiInvalidResponseError):
                code = "gemini_invalid_response"
                user_message = "Gemini returned an unreadable structured repair response."
            elif isinstance(cause, GeminiAPIError):
                code = "gemini_api_error"
                user_message = "Gemini became unavailable during the validation repair attempt."
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
                validation_history={
                    "initial": exc.initial_errors,
                    "final": exc.final_errors,
                },
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
                validation_errors=make_json_safe(generation.validation_errors),
                validation_history=make_json_safe(generation.validation_history),
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
        raw_lights = data.get("available_lights")
        if raw_lights is None:
            light_items: list[Any] = []
        elif isinstance(raw_lights, str):
            normalized_lights = raw_lights.replace("\r\n", "\n").replace("\r", "\n")
            light_items = normalized_lights.replace("\n", ",").split(",")
        elif isinstance(raw_lights, (list, tuple, set)):
            light_items = list(raw_lights)
        else:
            light_items = [raw_lights]

        lights = [str(item).strip() for item in light_items if str(item).strip()]
        lights_display = ", ".join(lights) if lights else "None specified"
        complete_prompt = "\n".join(
            (
                f"Story idea: {story}",
                f"Theme: {str(data.get('theme', '')).strip()}",
                f"Genre: {str(data.get('genre', '')).strip()}",
                f"Language: {str(data.get('language', '')).strip()}",
                f"Number of actors: {data.get('actor_count', '')}",
                f"Target duration: {data.get('duration_minutes', '')} minutes",
                f"Stage size: {str(data.get('stage_size', '')).strip()}",
                f"Available lighting fixtures: {lights_display}",
                f"Scene time: {str(data.get('scene_time', '')).strip()}",
                f"Desired emotion: {str(data.get('desired_emotion', '')).strip()}",
            )
        )
        return TheatreProject.objects.create(
            title=story.splitlines()[0][:255], user_prompt=complete_prompt,
            language=data["language"], genre=data["genre"], theme=data["theme"],
            actor_count=data["actor_count"], duration_minutes=data["duration_minutes"],
            stage_size=data["stage_size"], available_lights=lights,
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
                        validation_history: dict[str, Any] | None = None,
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
        errors = make_json_safe(
            validation_errors or [{"code": code, "message": message}]
        )
        run = GenerationRun.objects.create(
            project=project, model_name=self.model_name,
            model_settings=self.model_settings, user_input=project.user_prompt,
            scene_sources=self._source_ids(scene), blocking_sources=self._source_ids(blocking),
            lighting_sources=self._source_ids(lighting), retrieval_trace=trace,
            raw_output=raw_output, validated=False, validation_errors=errors,
            validation_history=make_json_safe(validation_history or {}),
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


@lru_cache(maxsize=1)
def build_default_service() -> ProductionService:
    """Build one local-model/Qdrant service per process.

    Qdrant local persistent mode uses an exclusive file lock and cannot safely be
    reopened for every request within the same Django worker.
    """
    embedder = EmbeddingService(settings.EMBEDDING_MODEL_NAME, device=settings.EMBEDDING_DEVICE,
                                batch_size=settings.EMBEDDING_BATCH_SIZE)
    store = QdrantStore(settings.QDRANT_PATH)
    client = create_provider()
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
        model_name=client.model,
        model_settings=client.reproducibility_settings(),
    )


def generate_production(request_data: dict[str, Any]) -> ProductionOutcome:
    """Public application entry point used by Django and future API adapters."""
    try:
        service = build_default_service()
    except ImproperlyConfigured as exc:
        raise ProductionServiceError(
            "llm_configuration_error",
            "The configured LLM provider is unsupported. Choose 'gemini' or 'ollama'.",
        ) from exc
    try:
        return service.generate_production(request_data)
    finally:
        service.dependencies.store.close()
