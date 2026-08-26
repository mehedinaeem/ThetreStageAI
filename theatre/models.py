"""Application persistence for theatre projects and reproducible generations."""
from __future__ import annotations

from django.core.validators import MinValueValidator
from django.db import models


class TheatreProject(models.Model):
    """A user's theatre brief and its latest validated generated artifact."""

    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=255)
    user_prompt = models.TextField()
    language = models.CharField(max_length=16, default="bn")
    genre = models.CharField(max_length=100, blank=True)
    theme = models.CharField(max_length=255, blank=True)
    actor_count = models.PositiveSmallIntegerField(validators=[MinValueValidator(1)])
    duration_minutes = models.PositiveSmallIntegerField(validators=[MinValueValidator(1)])
    stage_size = models.CharField(max_length=100, blank=True)
    available_lights = models.JSONField(
        default=list,
        blank=True,
        help_text="JSON list of fixture names or fixture specifications.",
    )
    generated_json = models.JSONField(
        default=dict,
        blank=True,
        help_text="Latest fully validated production JSON; never store unsafe output here.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=("created_at",), name="theatre_proj_created_idx"),
            models.Index(fields=("language", "genre"), name="theatre_proj_lang_gen_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(actor_count__gte=1),
                name="theatre_project_actor_count_gte_1",
            ),
            models.CheckConstraint(
                condition=models.Q(duration_minutes__gte=1),
                name="theatre_project_duration_gte_1",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.title} (#{self.pk})" if self.pk else self.title


class GenerationRun(models.Model):
    """Immutable-style evidence for one retrieval and generation experiment."""

    id = models.BigAutoField(primary_key=True)
    project = models.ForeignKey(
        TheatreProject,
        on_delete=models.CASCADE,
        related_name="generation_runs",
    )
    model_name = models.CharField(max_length=255)
    scene_sources = models.JSONField(
        default=list,
        blank=True,
        help_text="Ordered scene source IDs used for this run.",
    )
    blocking_sources = models.JSONField(
        default=list,
        blank=True,
        help_text="Ordered blocking source IDs used for this run.",
    )
    lighting_sources = models.JSONField(
        default=list,
        blank=True,
        help_text="Ordered lighting source IDs used for this run.",
    )
    retrieval_trace = models.JSONField(
        default=list,
        blank=True,
        help_text="Ranks, similarity scores, source IDs, and view types for reproducibility.",
    )
    research_query = models.TextField(
        blank=True,
        help_text="Exact researcher query used for this run, when applicable.",
    )
    retrieval_config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Selected per-view Top-K configuration for reproducibility.",
    )
    rag_mode = models.CharField(
        max_length=32,
        default="full_multiview",
        db_index=True,
        choices=(
            ("no_rag", "Mode 1 — No RAG"),
            ("scene_only", "Mode 2 — Scene-only RAG"),
            ("scene_blocking", "Mode 3 — Scene + Blocking RAG"),
            ("scene_lighting", "Mode 4 — Scene + Lighting RAG"),
            ("single_combined", "Mode 5 — Single combined retrieval RAG"),
            ("full_multiview", "Mode 6 — Full Multi-View RAG"),
        ),
        help_text="Retrieval ablation mode used for this generation run.",
    )
    raw_output = models.TextField(blank=True)
    validated = models.BooleanField(default=False, db_index=True)
    validation_errors = models.JSONField(default=list, blank=True)
    generation_time_seconds = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="End-to-end generation duration in seconds.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=("project", "created_at"), name="theatre_run_proj_time_idx"),
            models.Index(fields=("model_name",), name="theatre_run_model_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(generation_time_seconds__gte=0),
                name="theatre_run_generation_time_gte_0",
            )
        ]

    def __str__(self) -> str:
        status = "validated" if self.validated else "invalid"
        identifier = f"#{self.pk}" if self.pk else "unsaved"
        return f"{self.project.title} — {self.model_name} ({identifier}, {status})"
