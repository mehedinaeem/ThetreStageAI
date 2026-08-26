from django.test import TestCase
from django.urls import reverse
from types import SimpleNamespace
from unittest.mock import patch

from theatre.models import GenerationRun, TheatreProject


class FrontendPageTests(TestCase):
    def setUp(self) -> None:
        self.project = TheatreProject.objects.create(
            title="আলোর ওপারে", user_prompt="দুই চরিত্রের পারিবারিক সংঘাত",
            language="bn", genre="family_drama", theme="সম্পর্ক",
            actor_count=2, duration_minutes=15, stage_size="small",
            available_lights=["PAR01"],
            generated_json={
                "title": "আলোর ওপারে", "theme": "সম্পর্ক", "genre": "family_drama",
                "characters": [{"name": "মায়া", "description": "মা"}],
                "scenes": [{
                    "id": "S01", "title": "ফেরা", "location": "ঘর", "time": "রাত",
                    "stage_directions": ["মায়া ধীরে দাঁড়ায়।"],
                    "dialogue": [{"id": "D01", "speaker": "মায়া", "text": "ফিরে এসো।"}],
                    "blocking": [{"actor": "মায়া", "trigger": "D01", "from": "USL", "to": "CSC", "action": "walk"}],
                    "lighting": [{"cue_id": "LQ01", "trigger": "scene_start", "fixture": "PAR01", "focus_zone": "CSC", "rgb": [220, 120, 60], "intensity": 45, "fade_seconds": 2}],
                }],
            },
        )
        GenerationRun.objects.create(
            project=self.project, model_name="qwen3:4b", scene_sources=["natok_001"],
            retrieval_trace=[{"source_id": "natok_001", "view_type": "scene", "rank": 1, "score": 0.91, "metadata": {"theme": "সম্পর্ক", "scene_type": "confrontation"}}],
            raw_output="{}", validated=True, validation_errors=[], generation_time_seconds=4.2,
        )

    def test_all_public_pages_render(self) -> None:
        urls = (
            reverse("theatre:home"), reverse("theatre:new_production"),
            reverse("theatre:production_detail", args=[self.project.pk]),
            reverse("theatre:project_history"), reverse("theatre:rag_sources"),
            reverse("theatre:project_rag_sources", args=[self.project.pk]),
            reverse("theatre:research_about"),
        )
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_generated_page_renders_content_and_tabs(self) -> None:
        response = self.client.get(reverse("theatre:production_detail", args=[self.project.pk]))
        self.assertContains(response, "ফিরে এসো।")
        self.assertContains(response, "Blocking")
        self.assertContains(response, "Lighting")
        self.assertContains(response, "RAG Sources")
        self.assertContains(response, "--swatch-rgb: 220, 120, 60", html=False)

    @patch("theatre.views.generate_production")
    def test_valid_brief_calls_pipeline_and_redirects(self, mocked_generate: object) -> None:
        generated_project = TheatreProject.objects.create(
            title="নদীর ধারে", user_prompt="brief", actor_count=2,
            duration_minutes=12, available_lights=["PAR01", "Fresnel01"],
        )
        mocked_generate.return_value = SimpleNamespace(project=generated_project)  # type: ignore[attr-defined]
        response = self.client.post(reverse("theatre:new_production"), {
            "story_idea": "নদীর ধারে দুই বন্ধুর পুনর্মিলন", "theme": "বন্ধুত্ব",
            "genre": "social_drama", "language": "bn", "actor_count": 2,
            "duration_minutes": 12, "stage_size": "small",
            "available_lights": "PAR01, Fresnel01", "scene_time": "সন্ধ্যা",
            "desired_emotion": "আশা",
        })
        self.assertRedirects(response, reverse("theatre:production_detail", args=[generated_project.pk]))
        request_data = mocked_generate.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request_data["available_lights"], ["PAR01", "Fresnel01"])

    def test_invalid_form_has_accessible_error(self) -> None:
        response = self.client.post(reverse("theatre:new_production"), {})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="field-error"', html=False)
