"""Server-rendered presentation views; RAG and LLM work remains in services."""
from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from typing import Any

from django.core import signing
from django.db.models import Prefetch
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from pydantic import ValidationError
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import GenerationComparisonForm, ProductionBriefForm, ResearchRAGForm
from .models import GenerationRun, TheatreProject
from .services.export_service import (
    export_blocking_csv,
    export_json,
    export_lighting_csv,
    export_script_txt,
)
from .services.production_service import ProductionServiceError, generate_production
from .services.research_service import generate_from_research_selection, retrieve_for_research
from .services.validation import Production

logger = logging.getLogger(__name__)


def home(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "theatre/home.html",
        {
            "project_count": TheatreProject.objects.count(),
            "validated_run_count": GenerationRun.objects.filter(validated=True).count(),
        },
    )


def new_production(request: HttpRequest) -> HttpResponse:
    form = ProductionBriefForm(request.POST if request.method == "POST" else None)
    if request.method == "POST" and form.is_valid():
        request_data = dict(form.cleaned_data)
        request_data["available_lights"] = form.fixtures()
        try:
            outcome = generate_production(request_data)
        except ProductionServiceError as exc:
            return render(
                request,
                "theatre/new_production.html",
                {"form": form, "pipeline_error": exc.user_message, "error_code": exc.code},
                status=503 if exc.code != "validation_failed" else 422,
            )
        return redirect("theatre:production_detail", pk=outcome.project.pk)
    return render(request, "theatre/new_production.html", {"form": form})


def production_detail(request: HttpRequest, pk: int) -> HttpResponse:
    project = get_object_or_404(TheatreProject, pk=pk)
    run = project.generation_runs.first()
    production = _safe_production_snapshot(project.generated_json)
    for scene in production.get("scenes", []):
        for cue in scene.get("lighting", []):
            rgb = cue.get("rgb", [])
            if isinstance(rgb, list) and len(rgb) == 3:
                cue["rgb_css"] = ", ".join(str(value) for value in rgb)
    sources = _group_sources(run.retrieval_trace if run else [])
    return render(
        request,
        "theatre/production_detail.html",
        {
            "project": project,
            "production": production,
            "run": run,
            "sources": sources,
            "raw_json": json.dumps(project.generated_json, ensure_ascii=False, indent=2),
        },
    )


def export_production(request: HttpRequest, pk: int, export_format: str) -> HttpResponse:
    project = get_object_or_404(TheatreProject, pk=pk)
    if not project.generated_json:
        raise Http404("No validated production is available for export.")
    exporters = {
        "json": (export_json, "application/json; charset=utf-8", "production.json"),
        "txt": (export_script_txt, "text/plain; charset=utf-8", "script.txt"),
        "blocking.csv": (
            export_blocking_csv, "text/csv; charset=utf-8", "blocking.csv"
        ),
        "lighting.csv": (
            export_lighting_csv, "text/csv; charset=utf-8", "lighting.csv"
        ),
    }
    selected = exporters.get(export_format)
    if selected is None:
        raise Http404("Unsupported export format.")
    exporter, content_type, suffix = selected
    try:
        content = exporter(project.generated_json)
    except ValidationError:
        return HttpResponse(
            "The stored production is not valid and cannot be exported.",
            status=422,
            content_type="text/plain; charset=utf-8",
        )
    response = HttpResponse(content, content_type=content_type)
    response["Content-Disposition"] = (
        f'attachment; filename="thetrestageai-project-{project.pk}-{suffix}"'
    )
    response["X-Content-Type-Options"] = "nosniff"
    return response


def project_history(request: HttpRequest) -> HttpResponse:
    latest_runs = GenerationRun.objects.order_by("-created_at")
    projects = TheatreProject.objects.prefetch_related(
        Prefetch("generation_runs", queryset=latest_runs, to_attr="history_runs")
    ).all()
    return render(request, "theatre/project_history.html", {"projects": projects})


def compare_generations(request: HttpRequest) -> HttpResponse:
    form = GenerationComparisonForm(request.GET or None)
    comparisons: list[dict[str, Any]] = []
    if form.is_valid():
        comparisons = [
            _comparison_payload(form.cleaned_data["run_a"]),
            _comparison_payload(form.cleaned_data["run_b"]),
        ]
    return render(
        request,
        "theatre/generation_comparison.html",
        {"form": form, "comparisons": comparisons},
    )


def rag_sources(request: HttpRequest, pk: int | None = None) -> HttpResponse:
    if pk is not None:
        project = get_object_or_404(TheatreProject, pk=pk)
        run = project.generation_runs.first()
    else:
        run = GenerationRun.objects.select_related("project").first()
        project = run.project if run else None
    return render(
        request,
        "theatre/rag_sources.html",
        {
            "project": project,
            "run": run,
            "sources": _group_sources(run.retrieval_trace if run else []),
        },
    )


def research_about(request: HttpRequest) -> HttpResponse:
    return render(request, "theatre/research_about.html")


def research_rag(request: HttpRequest) -> HttpResponse:
    form = ResearchRAGForm(request.POST if request.method == "POST" else None)
    context: dict[str, Any] = {"form": form}
    if request.method == "POST" and request.POST.get("action") == "generate":
        try:
            outcome = generate_from_research_selection(request.POST.get("selection_token", ""))
        except signing.BadSignature:
            context["pipeline_error"] = (
                "The signed retrieval selection is invalid or expired. Run the retrieval again."
            )
            return render(request, "theatre/research_rag.html", context, status=400)
        except ProductionServiceError as exc:
            context.update(pipeline_error=exc.user_message, error_code=exc.code)
            return render(request, "theatre/research_rag.html", context, status=503)
        return redirect("theatre:production_detail", pk=outcome.project.pk)

    if request.method == "POST" and form.is_valid():
        try:
            retrieval = retrieve_for_research(
                form.cleaned_data["query"],
                scene_top_k=form.cleaned_data["scene_top_k"],
                blocking_top_k=form.cleaned_data["blocking_top_k"],
                lighting_top_k=form.cleaned_data["lighting_top_k"],
                combined_top_k=form.cleaned_data["combined_top_k"],
                rag_mode=form.cleaned_data["rag_mode"],
            )
        except ProductionServiceError as exc:
            context.update(pipeline_error=exc.user_message, error_code=exc.code)
            return render(request, "theatre/research_rag.html", context, status=503)
        context["retrieval"] = retrieval
    return render(request, "theatre/research_rag.html", context)


def _group_sources(trace: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trace:
        if not isinstance(item, dict):
            continue
        view_type = str(item.get("view_type", "")).lower()
        if view_type not in {"scene", "blocking", "lighting"}:
            continue
        try:
            rank = int(item.get("rank"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        source_id = str(item.get("source_id", "")).strip()
        if rank < 1 or not math.isfinite(score) or not source_id:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        grouped[view_type].append({
            "view_type": view_type,
            "source_id": source_id[:255],
            "rank": rank,
            "score": score,
            "metadata": {
                key: str(metadata[key])[:500]
                for key in ("theme", "scene_type")
                if key in metadata
            },
        })
    for results in grouped.values():
        results.sort(key=lambda item: (item.get("rank", 10_000), -item.get("score", 0)))
        del results[50:]
    return dict(grouped)


def _comparison_payload(run: GenerationRun) -> dict[str, Any]:
    production = _safe_production_snapshot(run.generated_json)
    for scene in production.get("scenes", []):
        for cue in scene.get("lighting", []):
            rgb = cue.get("rgb", [])
            if isinstance(rgb, list) and len(rgb) == 3:
                cue["rgb_css"] = ", ".join(str(value) for value in rgb)
    return {
        "run": run,
        "production": production,
        "sources": _group_sources(run.retrieval_trace),
    }


def _safe_production_snapshot(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        production = Production.model_validate(value)
    except ValidationError:
        logger.warning("Refusing to render malformed stored production JSON")
        return {}
    return copy.deepcopy(production.model_dump(mode="json", by_alias=True))


@api_view(["GET"])
def health(_: HttpRequest) -> Response:
    return Response({"status": "ok", "service": "ThetreStageAI"})
