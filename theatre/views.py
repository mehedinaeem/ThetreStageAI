"""Server-rendered presentation views; RAG and LLM work remains in services."""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .forms import ProductionBriefForm
from .models import GenerationRun, TheatreProject
from .services.production_service import ProductionServiceError, generate_production


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
    production = copy.deepcopy(project.generated_json) if project.generated_json else {}
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


def project_history(request: HttpRequest) -> HttpResponse:
    projects = TheatreProject.objects.prefetch_related("generation_runs").all()
    return render(request, "theatre/project_history.html", {"projects": projects})


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


def _group_sources(trace: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trace:
        if not isinstance(item, dict):
            continue
        view_type = str(item.get("view_type", "")).lower()
        if view_type in {"scene", "blocking", "lighting"}:
            grouped[view_type].append(item)
    for results in grouped.values():
        results.sort(key=lambda item: (item.get("rank", 10_000), -item.get("score", 0)))
    return dict(grouped)


@api_view(["GET"])
def health(_: HttpRequest) -> Response:
    return Response({"status": "ok", "service": "ThetreStageAI"})
