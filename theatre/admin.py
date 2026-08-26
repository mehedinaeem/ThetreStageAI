"""Research-friendly administration for projects and generation runs."""
from django.contrib import admin

from .models import GenerationRun, TheatreProject


class GenerationRunInline(admin.TabularInline):
    model = GenerationRun
    fields = ("model_name", "validated", "generation_time_seconds", "created_at")
    readonly_fields = fields
    extra = 0
    show_change_link = True
    can_delete = False


@admin.register(TheatreProject)
class TheatreProjectAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "language",
        "genre",
        "actor_count",
        "duration_minutes",
        "created_at",
        "updated_at",
    )
    list_filter = ("language", "genre", "created_at")
    search_fields = ("title", "theme", "user_prompt")
    readonly_fields = ("created_at", "updated_at")
    inlines = (GenerationRunInline,)


@admin.register(GenerationRun)
class GenerationRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "project",
        "model_name",
        "validated",
        "generation_time_seconds",
        "created_at",
    )
    list_filter = ("validated", "model_name", "created_at")
    search_fields = ("project__title", "model_name", "raw_output")
    readonly_fields = ("created_at",)
    list_select_related = ("project",)
