"""User-facing forms for theatre-production briefs."""
from __future__ import annotations

from django import forms


class ProductionBriefForm(forms.Form):
    story_idea = forms.CharField(
        label="Story idea",
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": "Describe the central story, conflict, and dramatic situation…",
                "autofocus": True,
            }
        ),
    )
    theme = forms.CharField(label="Theme", max_length=255)
    genre = forms.ChoiceField(
        label="Genre",
        choices=(
            ("social_drama", "Social drama"),
            ("family_drama", "Family drama"),
            ("tragedy", "Tragedy"),
            ("comedy", "Comedy"),
            ("experimental", "Experimental"),
            ("historical", "Historical"),
        ),
    )
    language = forms.ChoiceField(
        label="Language",
        choices=(("bn", "বাংলা (Bengali)"), ("bn-en", "Bengali with English terminology")),
        initial="bn",
    )
    actor_count = forms.IntegerField(label="Number of actors", min_value=1, max_value=30)
    duration_minutes = forms.IntegerField(
        label="Duration in minutes", min_value=1, max_value=240
    )
    stage_size = forms.ChoiceField(
        label="Stage size",
        choices=(("small", "Small"), ("medium", "Medium"), ("large", "Large")),
    )
    available_lights = forms.CharField(
        label="Available lighting fixtures",
        help_text="Separate fixtures with commas, for example: PAR01, PAR02, Fresnel01",
        widget=forms.TextInput(attrs={"placeholder": "PAR01, PAR02, Fresnel01"}),
    )
    scene_time = forms.CharField(
        label="Scene time", max_length=100, widget=forms.TextInput(attrs={"placeholder": "সন্ধ্যা"})
    )
    desired_emotion = forms.CharField(
        label="Desired emotion",
        max_length=100,
        widget=forms.TextInput(attrs={"placeholder": "রাগ, দ্বিধা, আশা"}),
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            css_class = "form-select" if isinstance(field.widget, forms.Select) else "form-control"
            field.widget.attrs["class"] = css_class

    def fixtures(self) -> list[str]:
        value = self.cleaned_data["available_lights"]
        return [fixture.strip() for fixture in value.replace("\n", ",").split(",") if fixture.strip()]

    def complete_prompt(self) -> str:
        data = self.cleaned_data
        return "\n".join(
            (
                data["story_idea"].strip(),
                f"Scene time: {data['scene_time']}",
                f"Desired emotion: {data['desired_emotion']}",
            )
        )
