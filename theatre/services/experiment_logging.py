"""Safe structured logging for reproducible generation experiments."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from theatre.models import GenerationRun

logger = logging.getLogger("theatre.research.experiments")

SAFE_MODEL_SETTING_KEYS = frozenset(
    {
        "provider",
        "temperature",
        "think",
        "timeout_seconds",
        "context_window",
        "seed",
        "top_p",
        "top_k",
        "repeat_penalty",
    }
)
SAFE_ERROR_KEYS = frozenset({"code", "type", "loc", "msg", "message"})
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def safe_model_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Allowlist experiment parameters; credential-like fields never pass through."""
    if not values:
        return {}
    return {
        key: value
        for key, value in values.items()
        if key in SAFE_MODEL_SETTING_KEYS and _is_scalar(value)
    }


def log_generation_run(run: GenerationRun) -> None:
    """Emit one UTF-8 JSON event after a GenerationRun has been persisted."""
    payload = {
        "event": "generation_record",
        "generation_run_id": run.pk,
        "project_id": run.project_id,
        "timestamp": run.created_at.isoformat(),
        "model": run.model_name,
        "model_settings": safe_model_settings(run.model_settings),
        "rag_mode": run.rag_mode,
        "user_input": _redact_text(run.user_input),
        "top_k": _effective_top_k(run),
        "retrieval": _retrieval_evidence(run.retrieval_trace),
        "generation_duration_seconds": run.generation_time_seconds,
        "validation_status": "validated" if run.validated else "invalid",
        "repair_attempts": run.repair_attempts,
        "errors": _safe_errors(run.validation_errors),
    }
    logger.info(
        "generation_record %s",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _retrieval_evidence(trace: object) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "scene": [], "blocking": [], "lighting": []
    }
    if not isinstance(trace, list):
        return grouped
    for item in trace:
        if not isinstance(item, dict):
            continue
        view = str(item.get("view_type", ""))
        if view not in grouped:
            continue
        grouped[view].append(
            {"source_id": str(item.get("source_id", "")), "score": item.get("score")}
        )
    return grouped


def _effective_top_k(run: GenerationRun) -> dict[str, int | None]:
    config = run.retrieval_config
    active_by_mode = {
        "no_rag": frozenset(),
        "scene_only": frozenset({"scene"}),
        "scene_blocking": frozenset({"scene", "blocking"}),
        "scene_lighting": frozenset({"scene", "lighting"}),
        "single_combined": frozenset({"combined"}),
        "full_multiview": frozenset({"scene", "blocking", "lighting"}),
    }
    active = active_by_mode.get(run.rag_mode, frozenset())
    return {
        view: config.get(f"{view}_top_k") if view in active else 0
        for view in ("scene", "blocking", "lighting", "combined")
    }


def _safe_errors(errors: object) -> list[dict[str, Any]]:
    if not isinstance(errors, list):
        return []
    safe: list[dict[str, Any]] = []
    for error in errors:
        if isinstance(error, dict):
            safe.append(
                {
                    key: _redact_value(error[key])
                    for key in SAFE_ERROR_KEYS
                    if key in error
                }
            )
    return safe


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _redact_value(value: object) -> object:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def _redact_text(value: str) -> str:
    redacted = _CREDENTIAL_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    return _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
