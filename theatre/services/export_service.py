"""Deterministic exports for validated theatre production snapshots."""
from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping
from typing import Any

from theatre.services.validation import Production

BLOCKING_COLUMNS = ("scene_id", "actor", "trigger", "from", "to", "action")
LIGHTING_COLUMNS = (
    "scene_id",
    "cue_id",
    "trigger",
    "fixture",
    "focus_zone",
    "red",
    "green",
    "blue",
    "intensity",
    "fade_seconds",
)


def export_json(production_data: Mapping[str, Any]) -> bytes:
    production = _validated_dict(production_data)
    text = json.dumps(production, ensure_ascii=False, indent=2)
    return (text + "\n").encode("utf-8")


def export_script_txt(production_data: Mapping[str, Any]) -> bytes:
    production = _validated_dict(production_data)
    lines = [
        production["title"],
        "=" * len(production["title"]),
        f"থিম: {production['theme']}",
        f"ধরন: {production['genre']}",
        "",
        "চরিত্রসমূহ",
        "------------",
    ]
    for character in production["characters"]:
        lines.append(f"{character['name']} — {character['description']}")

    for scene in production["scenes"]:
        lines.extend(
            (
                "",
                f"দৃশ্য {scene['id']}: {scene['title']}",
                f"স্থান: {scene['location']} | সময়: {scene['time']}",
                "",
                "মঞ্চ নির্দেশনা:",
            )
        )
        lines.extend(f"[{direction}]" for direction in scene["stage_directions"])
        lines.append("")
        for dialogue in scene["dialogue"]:
            lines.append(
                f"{dialogue['speaker']} ({dialogue['id']}): {dialogue['text']}"
            )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def export_blocking_csv(production_data: Mapping[str, Any]) -> bytes:
    production = _validated_dict(production_data)
    rows = (
        {
            "scene_id": _spreadsheet_safe(scene["id"]),
            "actor": _spreadsheet_safe(cue["actor"]),
            "trigger": _spreadsheet_safe(cue["trigger"]),
            "from": _spreadsheet_safe(cue["from"]),
            "to": _spreadsheet_safe(cue["to"]),
            "action": _spreadsheet_safe(cue["action"]),
        }
        for scene in production["scenes"]
        for cue in scene["blocking"]
    )
    return _csv_bytes(BLOCKING_COLUMNS, rows)


def export_lighting_csv(production_data: Mapping[str, Any]) -> bytes:
    production = _validated_dict(production_data)
    rows = (
        {
            "scene_id": _spreadsheet_safe(scene["id"]),
            "cue_id": _spreadsheet_safe(cue["cue_id"]),
            "trigger": _spreadsheet_safe(cue["trigger"]),
            "fixture": _spreadsheet_safe(cue["fixture"]),
            "focus_zone": _spreadsheet_safe(cue["focus_zone"]),
            "red": cue["rgb"][0],
            "green": cue["rgb"][1],
            "blue": cue["rgb"][2],
            "intensity": cue["intensity"],
            "fade_seconds": cue["fade_seconds"],
        }
        for scene in production["scenes"]
        for cue in scene["lighting"]
    )
    return _csv_bytes(LIGHTING_COLUMNS, rows)


def _validated_dict(production_data: Mapping[str, Any]) -> dict[str, Any]:
    return Production.model_validate(dict(production_data)).model_dump(
        mode="json", by_alias=True
    )


def _csv_bytes(
    columns: tuple[str, ...], rows: Iterable[Mapping[str, object]],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _spreadsheet_safe(value: object) -> object:
    """Prevent untrusted text cells from being interpreted as spreadsheet formulas."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
