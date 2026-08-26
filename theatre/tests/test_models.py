from django.test import TestCase

from theatre.models import GenerationRun, TheatreProject


class TheatrePersistenceTests(TestCase):
    def setUp(self) -> None:
        self.project = TheatreProject.objects.create(
            title="শেষ আলো",
            user_prompt="দুই চরিত্রের একটি পারিবারিক নাটক",
            language="bn",
            genre="social_drama",
            theme="পারিবারিক সংঘাত",
            actor_count=2,
            duration_minutes=20,
            stage_size="medium",
            available_lights=["PAR01", "PAR02"],
            generated_json={"title": "শেষ আলো", "scenes": []},
        )

    def test_project_preserves_json_and_has_useful_string(self) -> None:
        project = TheatreProject.objects.get(pk=self.project.pk)
        self.assertEqual(project.available_lights, ["PAR01", "PAR02"])
        self.assertEqual(project.generated_json["title"], "শেষ আলো")
        self.assertEqual(str(project), f"শেষ আলো (#{project.pk})")

    def test_generation_run_preserves_sources_scores_and_validation_evidence(self) -> None:
        run = GenerationRun.objects.create(
            project=self.project,
            model_name="qwen3:4b",
            scene_sources=["natok_001"],
            blocking_sources=["natok_010"],
            lighting_sources=["natok_020"],
            retrieval_trace=[
                {
                    "source_id": "natok_001",
                    "rank": 1,
                    "score": 0.9123,
                    "view_type": "scene",
                }
            ],
            raw_output='{"title":"শেষ আলো"}',
            validated=True,
            validation_errors=[],
            generation_time_seconds=12.5,
        )

        saved = GenerationRun.objects.get(pk=run.pk)
        self.assertEqual(saved.retrieval_trace[0]["score"], 0.9123)
        self.assertEqual(saved.scene_sources, ["natok_001"])
        self.assertIn("qwen3:4b", str(saved))
        self.assertIn("validated", str(saved))
        self.assertEqual(self.project.generation_runs.count(), 1)
