"""Build semantically distinct queries for each theatre retrieval view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from theatre.services.data.schemas import ViewType

MetadataValue: TypeAlias = str | int | bool


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    view_type: ViewType
    text: str
    metadata_filters: dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class ViewQueries:
    scene: RetrievalQuery
    blocking: RetrievalQuery
    lighting: RetrievalQuery


class MultiViewQueryBuilder:
    """Expand one request into three domain-focused Bengali search queries."""

    _headings: dict[ViewType, tuple[str, ...]] = {
        ViewType.SCENE: (
            "গল্প বা বিষয়",
            "থিম",
            "ধরন",
            "সংলাপের ভঙ্গি",
            "নাটকীয় পরিস্থিতি",
            "চরিত্র সংখ্যা",
            "স্থান",
            "আবেগ",
        ),
        ViewType.BLOCKING: (
            "অভিনেতার সংখ্যা",
            "মঞ্চে চলাচল",
            "প্রবেশ",
            "প্রস্থান",
            "সংঘাতের ধরন",
            "দৃশ্যের ধরন",
            "নাটকীয় উত্তেজনা",
            "আবেগ",
        ),
        ViewType.LIGHTING: (
            "দৃশ্যের ধরন",
            "আবেগ",
            "আলোর তীব্রতা",
            "মঞ্চের ফোকাস",
            "দিনের সময়",
            "অভিনেতার ব্লকিং",
            "RGB আলো",
            "মুড",
            "উপলভ্য লাইট ফিক্সচার",
        ),
    }

    def build_for_view(
        self,
        user_request: str,
        view_type: ViewType,
        *,
        metadata_filters: dict[str, MetadataValue] | None = None,
    ) -> RetrievalQuery:
        request = user_request.strip()
        if not request:
            raise ValueError("User request cannot be empty")
        focus = ", ".join(self._headings[view_type])
        text = (
            f"বাংলা থিয়েটারের {view_type.value} উদাহরণ অনুসন্ধান।\n"
            f"অনুরোধ: {request}\n"
            f"বিশেষভাবে বিবেচ্য: {focus}।"
        )
        return RetrievalQuery(view_type, text, dict(metadata_filters or {}))

    def build(
        self,
        user_request: str,
        *,
        metadata_filters: dict[str, MetadataValue] | None = None,
    ) -> ViewQueries:
        return ViewQueries(
            scene=self.build_for_view(
                user_request, ViewType.SCENE, metadata_filters=metadata_filters
            ),
            blocking=self.build_for_view(
                user_request, ViewType.BLOCKING, metadata_filters=metadata_filters
            ),
            lighting=self.build_for_view(
                user_request, ViewType.LIGHTING, metadata_filters=metadata_filters
            ),
        )
