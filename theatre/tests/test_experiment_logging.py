from __future__ import annotations

import json

from django.test import TestCase

from theatre.models import GenerationRun, TheatreProject
from theatre.services.experiment_logging import log_generation_run, safe_model_settings


class ExperimentLoggingTests(TestCase):
    def test_generation_event_contains_reproducibility_fields_without_secrets(self) -> None:
        project = TheatreProject.objects.create(
            title="গবেষণা", user_prompt="brief", actor_count=2, duration_minutes=15
        )
        run = GenerationRun.objects.create(
            project=project,
            model_name="qwen3:4b",
            model_settings={
                "provider": "ollama", "temperature": 0.2,
                "timeout_seconds": 180, "api_key": "never-log-this",
                "base_url": "http://user:password@localhost",
            },
            user_input="পারিবারিক সংঘাত token=super-secret",
            rag_mode="scene_only",
            retrieval_config={
                "scene_top_k": 5, "blocking_top_k": 0,
                "lighting_top_k": 0, "combined_top_k": 0,
            },
            scene_sources=["natok_1"],
            retrieval_trace=[{
                "source_id": "natok_1", "view_type": "scene",
                "rank": 1, "score": 0.9123,
                "metadata": {"payload_that_must_not_be_logged": "private"},
            }],
            validated=False,
            validation_errors=[{
                "code": "timeout", "message": "password=hunter2",
                "input": "raw model output must not be logged",
            }],
            generation_time_seconds=3.25,
            repair_attempts=1,
        )

        with self.assertLogs("theatre.research.experiments", level="INFO") as captured:
            log_generation_run(run)

        message = captured.output[0]
        payload = json.loads(message.split("generation_record ", 1)[1])
        self.assertEqual(payload["generation_run_id"], run.pk)
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertEqual(payload["model_settings"]["temperature"], 0.2)
        self.assertEqual(payload["rag_mode"], "scene_only")
        self.assertEqual(payload["top_k"]["scene"], 5)
        self.assertEqual(payload["retrieval"]["scene"][0]["source_id"], "natok_1")
        self.assertEqual(payload["retrieval"]["scene"][0]["score"], 0.9123)
        self.assertEqual(payload["generation_duration_seconds"], 3.25)
        self.assertEqual(payload["validation_status"], "invalid")
        self.assertEqual(payload["repair_attempts"], 1)
        self.assertIn("[REDACTED]", payload["user_input"])
        self.assertNotIn("super-secret", message)
        self.assertNotIn("hunter2", message)
        self.assertNotIn("never-log-this", message)
        self.assertNotIn("base_url", message)
        self.assertNotIn("payload_that_must_not_be_logged", message)
        self.assertNotIn("raw model output", message)

    def test_model_setting_allowlist_rejects_credential_fields(self) -> None:
        safe = safe_model_settings({
            "provider": "ollama", "seed": 42, "token": "secret",
            "authorization": "Bearer secret", "url": "https://secret@example.test",
        })
        self.assertEqual(safe, {"provider": "ollama", "seed": 42})
