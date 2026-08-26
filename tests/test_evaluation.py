from __future__ import annotations

import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from evaluation.experiment_runner import (
    ExperimentRunner,
    ExperimentStore,
    RetrievedEvidence,
    SystemRunOutput,
    SystemType,
)
from evaluation.export_results import (
    RATING_COLUMNS,
    export_expert_evaluation_csv,
    export_experiment_csv,
)
from evaluation.generation_metrics import measure_generation
from evaluation.retrieval_metrics import (
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def sample_production() -> dict[str, object]:
    return {
        "title": "নতুন নাটক",
        "scenes": [
            {
                "id": "S01",
                "title": "দৃশ্য",
                "dialogue": [{"id": "D01"}, {"id": "D02"}],
                "stage_directions": ["নির্দেশনা"],
                "blocking": [{"actor": "মায়া"}],
                "lighting": [{"cue_id": "LQ01"}, {"cue_id": "LQ02"}],
            }
        ],
    }


class FakeSystem:
    def __init__(self, system_type: SystemType) -> None:
        self.system_type = system_type

    def execute(self, user_query: str) -> SystemRunOutput:
        retrieved = []
        top_k = None
        if self.system_type in {
            SystemType.SYSTEM_C_SINGLE_COMBINED_RAG,
            SystemType.SYSTEM_D_MULTIVIEW_RAG,
        }:
            retrieved = [RetrievedEvidence(source_id="natok_001", score=0.91)]
            top_k = 1
        return SystemRunOutput(
            prompt=f"Prompt: {user_query}",
            model="qwen3:4b",
            retrieved=retrieved,
            top_k=top_k,
            validation_success=True,
            repair_attempts=0,
            production=sample_production(),
        )


class RetrievalMetricTests(SimpleTestCase):
    def test_binary_ranked_metrics(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"b", "c", "d"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 3), 2 / 3)
        self.assertEqual(reciprocal_rank(retrieved, relevant), 0.5)
        expected_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / (
            1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        )
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 3), expected_ndcg)

        metrics = evaluate_retrieval(retrieved, relevant, k=3)
        self.assertEqual(metrics.retrieved_count, 3)
        self.assertEqual(metrics.relevant_count, 3)

    def test_graded_ndcg_and_empty_relevance(self) -> None:
        score = ndcg_at_k(["medium", "high"], {"high": 3, "medium": 1}, 2)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)
        self.assertEqual(ndcg_at_k([], set(), 5), 0)
        self.assertEqual(recall_at_k(["a"], set(), 1), 0)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            precision_at_k([], set(), 0)


class GenerationMetricTests(SimpleTestCase):
    def test_counts_only_observable_structure(self) -> None:
        metrics = measure_generation(
            sample_production(), validation_success=True, repair_attempts=1
        )
        self.assertEqual(metrics.scene_count, 1)
        self.assertEqual(metrics.dialogue_count, 2)
        self.assertEqual(metrics.blocking_cue_count, 1)
        self.assertEqual(metrics.lighting_cue_count, 2)
        self.assertEqual(metrics.repair_attempts, 1)


class ExperimentInfrastructureTests(SimpleTestCase):
    def test_runner_tracks_and_round_trips_bengali_experiment(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = ExperimentStore(Path(temporary_directory) / "experiments.jsonl")
            record = ExperimentRunner(store).run(
                FakeSystem(SystemType.SYSTEM_D_MULTIVIEW_RAG),
                "দুই চরিত্রের সংঘাত",
                project_id=12,
                experiment_id="exp-001",
            )
            loaded = store.load()
        self.assertEqual(record.experiment_id, "exp-001")
        self.assertEqual(record.retrieved_source_ids, ["natok_001"])
        self.assertEqual(record.similarity_scores, [0.91])
        self.assertEqual(record.generated_script[0]["scene_id"], "S01")
        self.assertEqual(record.generated_lighting[1]["cue_id"], "LQ02")
        self.assertEqual(loaded[0].user_query, "দুই চরিত্রের সংঘাত")

    def test_comparison_requires_and_runs_all_four_systems(self) -> None:
        systems = {system_type: FakeSystem(system_type) for system_type in SystemType}
        records = ExperimentRunner().run_comparison(systems, ["প্রশ্ন এক", "প্রশ্ন দুই"])
        self.assertEqual(len(records), 8)
        self.assertEqual({record.system for record in records}, set(SystemType))
        with self.assertRaisesRegex(ValueError, "missing system"):
            ExperimentRunner().run_comparison(
                {SystemType.SYSTEM_A_BASE_QWEN: FakeSystem(SystemType.SYSTEM_A_BASE_QWEN)},
                ["প্রশ্ন"],
            )

    def test_expert_export_leaves_every_human_rating_empty(self) -> None:
        record = ExperimentRunner().run(
            FakeSystem(SystemType.SYSTEM_A_BASE_QWEN),
            "মানব মূল্যায়ন",
            project_id=8,
            experiment_id="exp-human",
        )
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            expert_path = export_expert_evaluation_csv([record], directory / "expert.csv")
            objective_path = export_experiment_csv([record], directory / "objective.csv")
            with expert_path.open(encoding="utf-8-sig", newline="") as source:
                row = next(csv.DictReader(source))
            with objective_path.open(encoding="utf-8-sig") as source:
                objective_text = source.read()
        self.assertTrue(all(row[column] == "" for column in RATING_COLUMNS))
        self.assertEqual(row["comments"], "")
        self.assertEqual(row["experiment_id"], "exp-human")
        self.assertIn("system_a_base_qwen", objective_text)
