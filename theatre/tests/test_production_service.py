from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from theatre.models import GenerationRun, TheatreProject
from theatre.services.data.schemas import ViewType
from theatre.services.llm import LLMConnectionError, LLMTimeoutError, ModelUnavailableError
from theatre.services.llm.generator import GenerationResult
from theatre.services.production_service import (
    ProductionDependencies,
    ProductionService,
    ProductionServiceError,
)
from theatre.services.rag.context_builder import RetrievalTrace
from theatre.services.rag.modes import RAGMode
from theatre.services.retrieval.base import RetrievalResult
from theatre.services.validation import Production, ProductionValidationError


def request_data() -> dict[str, Any]:
    return {
        "story_idea": "দুই বোনের পারিবারিক সংঘাত",
        "theme": "সম্পর্ক", "genre": "family_drama", "language": "bn",
        "actor_count": 2, "duration_minutes": 15, "stage_size": "small",
        "available_lights": ["PAR01", "PAR02"], "scene_time": "রাত",
        "desired_emotion": "রাগ",
    }


def production() -> Production:
    return Production.model_validate({
        "title": "শেষ কথা", "theme": "সম্পর্ক", "genre": "family_drama",
        "characters": [{"name": "মায়া", "description": "বড় বোন"}],
        "scenes": [{
            "id": "S01", "title": "মুখোমুখি", "location": "ঘর", "time": "রাত",
            "dialogue": [{"id": "D01", "speaker": "মায়া", "text": "আজ কথা হবে।"}],
            "stage_directions": ["মায়া সামনে আসে।"],
            "blocking": [{"actor": "মায়া", "from": "USL", "to": "CSC", "action": "walk", "trigger": "D01"}],
            "lighting": [{"cue_id": "LQ01", "trigger": "scene_start", "fixture": "PAR01", "focus_zone": "CSC", "rgb": [200, 80, 50], "intensity": 50, "fade_seconds": 2}],
        }],
    })


def retrieval(view: ViewType, number: int) -> RetrievalResult:
    return RetrievalResult(
        rank=number + 1, score=0.95 - number / 100, source_id=f"natok_{number}",
        view_type=view, search_text="বাংলা রেফারেন্স",
        metadata={"theme": "সম্পর্ক", "scene_type": "confrontation"}, payload={},
    )


class FakeClient:
    def __init__(self, exists: bool = True) -> None:
        self.exists = exists

    def collection_exists(self, _: str) -> bool:
        return self.exists


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.limits: list[int] = []
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, _: str, **kwargs: Any) -> list[RetrievalResult]:
        self.limits.append(kwargs["limit"])
        self.calls.append(kwargs)
        return self.results[: kwargs["limit"]]


class FakeGenerator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.user_requirements: list[str] = []

    def generate(
        self,
        user_requirements: str,
        scene: list[RetrievalResult],
        blocking: list[RetrievalResult],
        lighting: list[RetrievalResult],
        *,
        constraints: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.user_requirements.append(user_requirements)
        if self.error:
            raise self.error
        all_results = [*scene, *blocking, *lighting]
        return GenerationResult(
            production=production(),
            retrieval_trace=[RetrievalTrace(source_id=item.source_id, rank=item.rank, score=item.score, view_type=item.view_type) for item in all_results],
            raw_output=json.dumps(production().model_dump(mode="json", by_alias=True), ensure_ascii=False),
            accepted_output="{}", validation_errors=[], repaired=False,
            validation_history={"initial": [], "final": []},
        )


def service(*, exists: bool = True, empty_view: ViewType | None = None,
            generator_error: Exception | None = None) -> tuple[ProductionService, tuple[FakeRetriever, ...]]:
    scene = FakeRetriever([] if empty_view is ViewType.SCENE else [retrieval(ViewType.SCENE, i) for i in range(5)])
    blocking = FakeRetriever([] if empty_view is ViewType.BLOCKING else [retrieval(ViewType.BLOCKING, i) for i in range(3)])
    lighting = FakeRetriever([] if empty_view is ViewType.LIGHTING else [retrieval(ViewType.LIGHTING, i) for i in range(3)])
    store = SimpleNamespace(client=FakeClient(exists))
    dependencies = ProductionDependencies(
        store=store, scene_retriever=scene, blocking_retriever=blocking,
        lighting_retriever=lighting, generator=FakeGenerator(generator_error),
    )
    return ProductionService(dependencies, model_name="qwen3:4b"), (scene, blocking, lighting)


class ProductionServiceIntegrationTests(TestCase):
    def test_complete_form_constraints_are_saved_and_passed_to_generation(self) -> None:
        pipeline, retrievers = service()
        form_data = {
            "story_idea": "বিশ্ববিদ্যালয়ের দুই শিক্ষার্থীর মধ্যে একটি ভুয়া পোস্ট নিয়ে সংঘাত তৈরি হয়।",
            "theme": "Fake News",
            "genre": "Social Drama",
            "language": "bn",
            "actor_count": 2,
            "duration_minutes": 10,
            "stage_size": "small",
            "available_lights": "RGB_PAR_01, RGB_PAR_02, RGB_PAR_03, RGB_PAR_04",
            "scene_time": "সন্ধ্যা",
            "desired_emotion": "রাগ, উত্তেজনা, উপলব্ধি",
        }

        outcome = pipeline.generate_production(form_data)

        expected_lines = (
            "Story idea: বিশ্ববিদ্যালয়ের দুই শিক্ষার্থীর মধ্যে একটি ভুয়া পোস্ট নিয়ে সংঘাত তৈরি হয়।",
            "Theme: Fake News",
            "Genre: Social Drama",
            "Language: bn",
            "Number of actors: 2",
            "Target duration: 10 minutes",
            "Stage size: small",
            "Available lighting fixtures: RGB_PAR_01, RGB_PAR_02, RGB_PAR_03, RGB_PAR_04",
            "Scene time: সন্ধ্যা",
            "Desired emotion: রাগ, উত্তেজনা, উপলব্ধি",
        )
        self.assertEqual(outcome.project.user_prompt, "\n".join(expected_lines))
        self.assertEqual(
            outcome.project.available_lights,
            ["RGB_PAR_01", "RGB_PAR_02", "RGB_PAR_03", "RGB_PAR_04"],
        )
        self.assertEqual(
            pipeline.dependencies.generator.user_requirements,
            [outcome.project.user_prompt],
        )
        self.assertTrue(
            all(retriever.calls for retriever in retrievers),
            "The complete project prompt must also reach all active retrievers.",
        )

    def test_available_lights_normalizes_multiline_lists_and_empty_values(self) -> None:
        cases = (
            (
                "RGB_PAR_01\n RGB_PAR_02,RGB_PAR_03",
                ["RGB_PAR_01", "RGB_PAR_02", "RGB_PAR_03"],
            ),
            ([" RGB_PAR_01 ", 42, None, ""], ["RGB_PAR_01", "42", "None"]),
            (None, []),
        )
        for raw_lights, expected in cases:
            with self.subTest(raw_lights=raw_lights):
                pipeline, _ = service()
                outcome = pipeline.generate_production(
                    request_data() | {"available_lights": raw_lights}
                )
                self.assertEqual(outcome.project.available_lights, expected)
                expected_display = ", ".join(expected) if expected else "None specified"
                self.assertIn(
                    f"Available lighting fixtures: {expected_display}",
                    outcome.project.user_prompt,
                )

    def test_complete_pipeline_persists_validated_project_and_trace(self) -> None:
        pipeline, retrievers = service()
        outcome = pipeline.generate_production(request_data())

        outcome.project.refresh_from_db()
        self.assertEqual(outcome.project.generated_json["title"], "শেষ কথা")
        self.assertTrue(outcome.run.validated)
        self.assertEqual(outcome.run.generated_json["title"], "শেষ কথা")
        self.assertEqual(outcome.run.user_input, outcome.project.user_prompt)
        self.assertEqual(outcome.run.repair_attempts, 0)
        self.assertEqual(outcome.run.retrieval_config["scene_top_k"], 3)
        self.assertEqual(outcome.run.retrieval_config["combined_top_k"], 7)
        self.assertEqual(len(outcome.run.scene_sources), 3)
        self.assertEqual(len(outcome.run.blocking_sources), 2)
        self.assertEqual(len(outcome.run.lighting_sources), 2)
        self.assertEqual(len(outcome.run.retrieval_trace), 7)
        self.assertEqual(outcome.run.retrieval_trace[0]["metadata"]["theme"], "সম্পর্ক")
        self.assertEqual([item.limits for item in retrievers], [[3], [2], [2]])

    def test_research_generation_persists_exact_query_and_top_k(self) -> None:
        pipeline, _ = service()
        selected = (
            [retrieval(ViewType.SCENE, 0)],
            [retrieval(ViewType.BLOCKING, 1)],
            [retrieval(ViewType.LIGHTING, 2)],
        )
        outcome = pipeline.generate_production(
            request_data(),
            retrieved_results=selected,
            research_query="পরীক্ষামূলক সংঘাত",
            retrieval_config={"scene_top_k": 1, "blocking_top_k": 1, "lighting_top_k": 1},
        )
        self.assertEqual(outcome.run.research_query, "পরীক্ষামূলক সংঘাত")
        self.assertEqual(outcome.run.retrieval_config["scene_top_k"], 1)
        self.assertEqual(outcome.run.scene_sources, ["natok_0"])
        self.assertEqual(outcome.run.retrieval_trace[0]["score"], 0.95)

    def test_no_rag_bypasses_index_and_persists_mode(self) -> None:
        pipeline, retrievers = service(exists=False)

        outcome = pipeline.generate_production(request_data(), rag_mode=RAGMode.NO_RAG)

        self.assertEqual(outcome.run.rag_mode, RAGMode.NO_RAG.value)
        self.assertEqual(outcome.run.retrieval_trace, [])
        self.assertEqual(outcome.run.scene_sources, [])
        self.assertEqual([item.calls for item in retrievers], [[], [], []])

    def test_partial_modes_call_only_their_active_retrievers(self) -> None:
        cases = (
            (RAGMode.SCENE_ONLY, [[3], [], []]),
            (RAGMode.SCENE_BLOCKING, [[3], [2], []]),
            (RAGMode.SCENE_LIGHTING, [[3], [], [2]]),
            (RAGMode.FULL_MULTIVIEW, [[3], [2], [2]]),
        )
        for mode, expected_limits in cases:
            with self.subTest(mode=mode):
                pipeline, retrievers = service()
                outcome = pipeline.generate_production(request_data(), rag_mode=mode)
                self.assertEqual(outcome.run.rag_mode, mode.value)
                self.assertEqual([item.limits for item in retrievers], expected_limits)

    def test_single_combined_mode_uses_one_raw_query_and_global_top_k(self) -> None:
        pipeline, retrievers = service()

        outcome = pipeline.generate_production(
            request_data(),
            rag_mode=RAGMode.SINGLE_COMBINED,
            retrieval_config={"combined_top_k": 2},
        )

        self.assertEqual(outcome.run.rag_mode, RAGMode.SINGLE_COMBINED.value)
        self.assertEqual(len(outcome.run.retrieval_trace), 2)
        self.assertTrue(all(item.calls[0]["query_text"] for item in retrievers))
        self.assertEqual([item.limits for item in retrievers], [[2], [2], [2]])

    def test_missing_index_is_controlled_and_recorded(self) -> None:
        pipeline, _ = service(exists=False)
        with self.assertRaises(ProductionServiceError) as captured:
            pipeline.generate_production(request_data())
        self.assertEqual(captured.exception.code, "index_not_built")
        self.assertFalse(GenerationRun.objects.get().validated)
        self.assertEqual(GenerationRun.objects.get().validation_errors[0]["code"], "index_not_built")

    def test_empty_retrieval_is_controlled_and_recorded(self) -> None:
        pipeline, _ = service(empty_view=ViewType.LIGHTING)
        with self.assertRaises(ProductionServiceError) as captured:
            pipeline.generate_production(request_data())
        self.assertEqual(captured.exception.code, "empty_retrieval")
        self.assertEqual(len(GenerationRun.objects.get().scene_sources), 3)

    def test_local_llm_failures_have_specific_codes(self) -> None:
        cases = (
            (LLMConnectionError("offline"), "ollama_unavailable"),
            (LLMTimeoutError("slow"), "ollama_timeout"),
            (ModelUnavailableError("missing"), "model_unavailable"),
        )
        for error, code in cases:
            with self.subTest(code=code):
                pipeline, _ = service(generator_error=error)
                with self.assertRaises(ProductionServiceError) as captured:
                    pipeline.generate_production(
                        request_data(), rag_mode=RAGMode.SCENE_ONLY
                    )
                self.assertEqual(captured.exception.code, code)
                self.assertEqual(
                    GenerationRun.objects.latest("id").rag_mode,
                    RAGMode.SCENE_ONLY.value,
                )

    def test_failed_repair_never_saves_generated_json(self) -> None:
        error = ProductionValidationError(
            "invalid", initial_errors=[{"type": "intensity"}],
            final_errors=[{"type": "rgb"}],
        )
        pipeline, _ = service(generator_error=error)
        with self.assertRaises(ProductionServiceError) as captured:
            pipeline.generate_production(request_data())
        self.assertEqual(captured.exception.code, "validation_failed")
        project = TheatreProject.objects.get(pk=captured.exception.project.pk)
        self.assertEqual(project.generated_json, {})
        self.assertFalse(project.generation_runs.get().validated)
        self.assertEqual(project.generation_runs.get().repair_attempts, 1)
        self.assertEqual(
            project.generation_runs.get().validation_errors,
            [{"type": "rgb"}],
        )
        self.assertEqual(
            project.generation_runs.get().validation_history,
            {
                "initial": [{"type": "intensity"}],
                "final": [{"type": "rgb"}],
            },
        )

    def test_failure_errors_with_exception_context_are_saved_as_json(self) -> None:
        unsafe_error = {
            "type": "value_error",
            "loc": ("scenes", 0),
            "msg": "Value error, Invalid blocking trigger D03",
            "ctx": {"error": ValueError("Invalid blocking trigger D03")},
        }
        error = ProductionValidationError(
            "invalid",
            initial_errors=[unsafe_error],
            final_errors=[unsafe_error],
            initial_output='{"invalid": true}',
            corrected_output='{"still_invalid": true}',
        )
        pipeline, _ = service(generator_error=error)

        with self.assertRaises(ProductionServiceError) as captured:
            pipeline.generate_production(request_data())

        self.assertEqual(
            captured.exception.user_message,
            "The generated production could not be validated after one repair attempt. "
            "No unsafe lighting output was saved.",
        )
        run = GenerationRun.objects.get()
        self.assertFalse(run.validated)
        self.assertEqual(run.repair_attempts, 1)
        self.assertEqual(run.raw_output, '{"still_invalid": true}')
        self.assertEqual(run.validation_errors[0]["loc"], ["scenes", 0])
        self.assertEqual(
            run.validation_errors[0]["ctx"]["error"],
            "Invalid blocking trigger D03",
        )
        json.dumps(run.validation_errors)

    @patch("theatre.views.generate_production")
    def test_view_renders_safe_pipeline_error(self, mocked_generate: Any) -> None:
        mocked_generate.side_effect = ProductionServiceError(
            "ollama_unavailable", "Ollama is not reachable. Start it and try again."
        )
        response = self.client.post(reverse("theatre:new_production"), request_data() | {
            "available_lights": "PAR01, PAR02"
        })
        self.assertEqual(response.status_code, 503)
        self.assertContains(response, "Ollama is not reachable", status_code=503)
        self.assertNotContains(response, "Traceback", status_code=503)
