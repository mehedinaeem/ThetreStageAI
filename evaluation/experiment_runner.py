"""System-neutral execution and durable JSONL experiment tracking."""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .generation_metrics import extract_generation_components


class SystemType(StrEnum):
    SYSTEM_A_BASE_QWEN = "system_a_base_qwen"
    SYSTEM_B_PROMPT_ENGINEERED_QWEN = "system_b_prompt_engineered_qwen"
    SYSTEM_C_SINGLE_COMBINED_RAG = "system_c_single_combined_rag"
    SYSTEM_D_MULTIVIEW_RAG = "system_d_thetrestageai_multiview_rag"


class RetrievedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1)
    score: float
    view_type: str | None = None


class SystemRunOutput(BaseModel):
    """Actual output supplied by a concrete A/B/C/D system adapter."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    model: str = Field(min_length=1)
    retrieved: list[RetrievedEvidence] = Field(default_factory=list)
    top_k: int | dict[str, int] | None = None
    validation_success: bool
    repair_attempts: int = Field(ge=0)
    production: dict[str, Any] | None = None


class ExperimentSystem(Protocol):
    system_type: SystemType

    def execute(self, user_query: str) -> SystemRunOutput: ...


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str = Field(min_length=1)
    project_id: int | None = None
    user_query: str = Field(min_length=1)
    system: SystemType
    retrieved_source_ids: list[str]
    similarity_scores: list[float]
    top_k: int | dict[str, int] | None
    prompt: str
    model: str
    generation_latency_seconds: float = Field(ge=0)
    validation_success: bool
    repair_attempts: int = Field(ge=0)
    generated_script: list[dict[str, Any]]
    generated_blocking: list[dict[str, Any]]
    generated_lighting: list[dict[str, Any]]
    created_at: datetime

    @model_validator(mode="after")
    def evidence_lengths_match(self) -> ExperimentRecord:
        if len(self.retrieved_source_ids) != len(self.similarity_scores):
            raise ValueError("Retrieved source IDs and similarity scores must have equal lengths")
        return self


class ExperimentStore:
    """Append-only UTF-8 JSONL storage configured by the caller."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def append(self, record: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as destination:
            destination.write(record.model_dump_json() + "\n")

    def load(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        records: list[ExperimentRecord] = []
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    records.append(ExperimentRecord.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid experiment record at {self.path}:{line_number}"
                    ) from exc
        return records


class ExperimentRunner:
    def __init__(self, store: ExperimentStore | None = None) -> None:
        self.store = store

    def run(
        self,
        system: ExperimentSystem,
        user_query: str,
        *,
        project_id: int | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        query = user_query.strip()
        if not query:
            raise ValueError("User query cannot be empty")
        started = perf_counter()
        output = system.execute(query)
        latency = perf_counter() - started
        script, blocking, lighting = extract_generation_components(output.production)
        record = ExperimentRecord(
            experiment_id=experiment_id or str(uuid4()),
            project_id=project_id,
            user_query=query,
            system=system.system_type,
            retrieved_source_ids=[item.source_id for item in output.retrieved],
            similarity_scores=[item.score for item in output.retrieved],
            top_k=output.top_k,
            prompt=output.prompt,
            model=output.model,
            generation_latency_seconds=latency,
            validation_success=output.validation_success,
            repair_attempts=output.repair_attempts,
            generated_script=script,
            generated_blocking=blocking,
            generated_lighting=lighting,
            created_at=datetime.now(UTC),
        )
        if self.store is not None:
            self.store.append(record)
        return record

    def run_comparison(
        self,
        systems: Mapping[SystemType, ExperimentSystem],
        user_queries: Sequence[str],
    ) -> list[ExperimentRecord]:
        """Run every query against A/B/C/D, rejecting incomplete comparisons."""
        required = set(SystemType)
        missing = required - set(systems)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"Comparison is missing system adapter(s): {names}")
        records: list[ExperimentRecord] = []
        for query in user_queries:
            for system_type in SystemType:
                system = systems[system_type]
                if system.system_type is not system_type:
                    raise ValueError(f"Adapter key does not match {system.system_type.value}")
                records.append(self.run(system, query))
        return records
