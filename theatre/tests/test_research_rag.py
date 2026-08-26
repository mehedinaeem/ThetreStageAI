from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from django.core import signing
from django.test import TestCase
from django.urls import reverse

from theatre.models import TheatreProject
from theatre.services.data.schemas import ViewType
from theatre.services.research_service import (
    ResearchRetrieval,
    generate_from_research_selection,
    retrieve_for_research,
)
from theatre.services.retrieval.base import RetrievalResult
from theatre.services.rag.modes import RAGMode


def result(view: ViewType, source: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=score, source_id=source, view_type=view,
        search_text="দীর্ঘ বাংলা অনুসন্ধান পাঠ্য",
        metadata={
            "title": "গবেষণা নাটক", "theme": "সংঘাত",
            "scene_type": "confrontation", "emotion": "রাগ",
        },
        payload={"reference": view.value},
    )


class ClosableStore:
    def close(self) -> None:
        return None


class RetrievalOnlyService:
    def __init__(self) -> None:
        self.dependencies = SimpleNamespace(store=ClosableStore())

    def retrieve_sources(self, query: str, **_: Any) -> tuple[list[RetrievalResult], ...]:
        return (
            [result(ViewType.SCENE, "scene_1", 0.91)],
            [result(ViewType.BLOCKING, "blocking_1", 0.82)],
            [result(ViewType.LIGHTING, "lighting_1", 0.79)],
        )

    def retrieve_for_mode(self, query: str, **kwargs: Any) -> tuple[list[RetrievalResult], ...]:
        return self.retrieve_sources(query, **kwargs)

    @staticmethod
    def require_results_for_mode(*_: object) -> None:
        return None


class GenerationCaptureService:
    def __init__(self, project: TheatreProject) -> None:
        self.dependencies = SimpleNamespace(store=ClosableStore())
        self.project = project
        self.kwargs: dict[str, Any] = {}

    def generate_production(self, _: dict[str, Any], **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(project=self.project, run=SimpleNamespace())


class ResearchServiceTests(TestCase):
    @patch("theatre.services.research_service.build_default_service")
    def test_signed_selection_round_trip_uses_exact_results(self, mocked_builder: Any) -> None:
        mocked_builder.return_value = RetrievalOnlyService()
        retrieval = retrieve_for_research(
            "পারিবারিক সংঘাত", scene_top_k=5, blocking_top_k=3, lighting_top_k=3
        )
        project = TheatreProject.objects.create(
            title="Research", user_prompt="query", actor_count=2, duration_minutes=15
        )
        capture = GenerationCaptureService(project)
        mocked_builder.return_value = capture

        outcome = generate_from_research_selection(retrieval.selection_token)

        selected = capture.kwargs["retrieved_results"]
        self.assertEqual(selected[0][0].source_id, "scene_1")
        self.assertEqual(selected[1][0].score, 0.82)
        self.assertEqual(capture.kwargs["research_query"], "পারিবারিক সংঘাত")
        self.assertEqual(capture.kwargs["retrieval_config"]["lighting_top_k"], 3)
        self.assertEqual(capture.kwargs["rag_mode"], RAGMode.FULL_MULTIVIEW)
        self.assertEqual(outcome.project, project)

    def test_tampered_selection_is_rejected(self) -> None:
        with self.assertRaises(signing.BadSignature):
            generate_from_research_selection("tampered-token")


class ResearchRAGViewTests(TestCase):
    @patch("theatre.views.retrieve_for_research")
    def test_page_displays_three_retrieval_columns(self, mocked_retrieve: Any) -> None:
        mocked_retrieve.return_value = ResearchRetrieval(
            query="সংঘাত",
            rag_mode=RAGMode.FULL_MULTIVIEW,
            scene_results=[result(ViewType.SCENE, "scene_1", 0.91)],
            blocking_results=[result(ViewType.BLOCKING, "blocking_1", 0.82)],
            lighting_results=[result(ViewType.LIGHTING, "lighting_1", 0.79)],
            top_k={"scene_top_k": 5, "blocking_top_k": 3, "lighting_top_k": 3},
            selection_token="signed-selection",
        )
        response = self.client.post(
            reverse("theatre:research_rag"),
            {
                "action": "retrieve", "query": "সংঘাত", "scene_top_k": 5,
                "blocking_top_k": 3, "lighting_top_k": 3,
                "combined_top_k": 11, "rag_mode": RAGMode.FULL_MULTIVIEW.value,
            },
        )
        self.assertContains(response, "Scene Retrieval")
        self.assertContains(response, "Blocking Retrieval")
        self.assertContains(response, "Lighting Retrieval")
        self.assertContains(response, "scene_1")
        self.assertContains(response, "গবেষণা নাটক")
        self.assertContains(response, "Generate Using These Sources")
        self.assertContains(response, "Full Multi-View RAG")
        self.assertEqual(
            mocked_retrieve.call_args.kwargs["rag_mode"],
            RAGMode.FULL_MULTIVIEW.value,
        )

    @patch("theatre.views.retrieve_for_research")
    def test_researcher_can_select_no_rag_mode(self, mocked_retrieve: Any) -> None:
        mocked_retrieve.return_value = ResearchRetrieval(
            query="সংঘাত", rag_mode=RAGMode.NO_RAG,
            scene_results=[], blocking_results=[], lighting_results=[],
            top_k={
                "scene_top_k": 5, "blocking_top_k": 3,
                "lighting_top_k": 3, "combined_top_k": 11,
            },
            selection_token="signed-no-rag",
        )
        response = self.client.post(
            reverse("theatre:research_rag"),
            {
                "action": "retrieve", "query": "সংঘাত",
                "rag_mode": RAGMode.NO_RAG.value,
                "scene_top_k": 5, "blocking_top_k": 3,
                "lighting_top_k": 3, "combined_top_k": 11,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Mode 1 — No RAG")
        self.assertEqual(
            mocked_retrieve.call_args.kwargs["rag_mode"], RAGMode.NO_RAG.value
        )

    @patch("theatre.views.generate_from_research_selection")
    def test_generate_button_redirects_to_generated_project(self, mocked_generate: Any) -> None:
        project = TheatreProject.objects.create(
            title="Result", user_prompt="query", actor_count=2, duration_minutes=15
        )
        mocked_generate.return_value = SimpleNamespace(project=project)
        response = self.client.post(
            reverse("theatre:research_rag"),
            {"action": "generate", "selection_token": "signed-selection"},
        )
        self.assertRedirects(
            response, reverse("theatre:production_detail", args=[project.pk])
        )
