# Bangla Natok 500 Multi-View RAG Dataset

Bangla Natok 500 is a synthetic Bengali theatre dataset prepared for research on
retrieval-augmented theatre generation. Each of the 500 source records includes
dramatic content together with actor-blocking and lighting annotations. Three
specialized retrieval views expose the same source records from scene, blocking,
and lighting perspectives.

The dataset is used by **ThetreStageAI**, a Bengali theatre-production intelligence
prototype that retrieves view-specific references before generating new scripts,
blocking, and structured lighting cues.

## Dataset summary

| Content | Records |
|---|---:|
| Original theatre records | 500 |
| Scene retrieval documents | 500 |
| Blocking retrieval documents | 500 |
| Lighting retrieval documents | 500 |
| Total retrieval documents | 1,500 |
| Human-readable script files | 500 |

The counts above were verified directly from the JSONL and script files. They are
also recorded in `dataset_summary.json`.

## Directory structure

```text
bangla_natok_500/
├── README.md
├── bangla_natok_500.jsonl
├── dataset_summary.json
├── preview_500.csv
├── qdrant_ingest_example.py
├── retrieval_views/
│   ├── scene_view.jsonl
│   ├── blocking_view.jsonl
│   ├── lighting_view.jsonl
│   └── all_retrieval_views.jsonl
└── scripts/
    ├── natok_001.txt
    ├── ...
    └── natok_500.txt
```

## Original record format

Each line of `bangla_natok_500.jsonl` is one UTF-8 JSON object. Records contain:

| Field | Purpose |
|---|---|
| `id` | Stable source identifier such as `natok_001` |
| `document_type` | Dataset document category |
| `language` | Content language |
| `title` | Bengali production title |
| `theme` | Central theme |
| `genre` | Dramatic genre |
| `scene_type` | Type of dramatic scene |
| `location` | Scene location |
| `time` | Scene time |
| `emotion` | Dominant emotion information |
| `characters` | Character definitions |
| `dialogue` | Structured character dialogue |
| `stage_directions` | Performance and staging directions |
| `blocking` | Actor movement and stage-zone information |
| `lighting` | Structured lighting directions |
| `sound` | Sound directions or cues |
| `summary` | Concise dramatic summary |
| `license_note` | Per-record synthetic-origin note |

Fields can contain nested objects or arrays. Consumers should validate their types
instead of assuming every optional value is populated.

## Retrieval-view format

Every retrieval-view line contains the following structure:

```json
{
  "id": "natok_001__scene",
  "view_type": "scene",
  "source_id": "natok_001",
  "search_text": "Text embedded for semantic retrieval",
  "metadata": {},
  "payload": {}
}
```

The fields have these roles:

- `id`: unique retrieval-document identifier.
- `view_type`: `scene`, `blocking`, or `lighting`.
- `source_id`: corresponding ID in `bangla_natok_500.jsonl`.
- `search_text`: Bengali text intended for embedding and similarity search.
- `metadata`: compact attributes for filtering and research inspection.
- `payload`: structured reference content returned after retrieval.

All three views provide metadata such as title, theme, genre, scene type, location,
time, emotion, character names, actor count, language, document type, and source ID.
Their payloads differ intentionally:

| View | Payload content | Intended retrieval question |
|---|---|---|
| Scene | `dialogue`, `stage_directions`, `summary` | What dramatic situation or dialogue pattern is relevant? |
| Blocking | `blocking`, `stage_directions` | What actor movement or staging pattern is relevant? |
| Lighting | `lighting`, `blocking` | What lighting treatment fits the mood, focus, and movement? |

## Multi-view retrieval design

ThetreStageAI uses independent queries and collections for each view:

```text
User requirements
       │
       ├── Scene query ──────> Scene Retriever ──────> Top 5
       ├── Blocking query ───> Blocking Retriever ───> Top 3
       └── Lighting query ───> Lighting Retriever ───> Top 3
                                      │
                                      ▼
                              RAG Context Builder
                                      │
                                      ▼
                       New script + blocking + lighting
```

Retrieved documents are untrusted reference material. They must not override system
instructions, and generated dialogue should not copy retrieved dialogue verbatim.

## Combined-view warning

`all_retrieval_views.jsonl` contains the same 1,500 documents found across the three
individual view files. Use either:

- the three individual files for multi-view indexing, or
- the combined file for a single-retrieval baseline.

Do **not** index the combined file together with the three individual files unless
duplicate entries are explicitly desired.

## Using the dataset with ThetreStageAI

Configure the dataset directory in the project `.env` file:

```env
THEATRE_DATASET_PATH=Bangla_Natok_500_MultiView_RAG_Dataset/bangla_natok_500
```

Inspect and validate the files without modifying them:

```bash
python manage.py inspect_theatre_dataset
```

Build the three persistent Qdrant collections:

```bash
python manage.py build_rag_index
```

To deliberately recreate only the project collections:

```bash
python manage.py build_rag_index --rebuild
```

Test independent Bengali retrieval:

```bash
python manage.py test_retrieval "দুইজন চরিত্রের রাগপূর্ণ পারিবারিক সংঘাত"
```

The application embeds only `search_text`. It preserves `id`, `source_id`,
`view_type`, `search_text`, `metadata`, and `payload` in each Qdrant point.

## Safe JSONL loading

Read the files as UTF-8 and parse one line at a time:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            yield record
```

The project loader additionally applies record-size limits, typed Pydantic schemas,
malformed-record logging, view detection, and aggregate statistics.

## Data integrity expectations

Before indexing, verify that:

- every JSONL line is valid UTF-8 JSON;
- every retrieval document has a non-empty `id`, `view_type`, and `source_id`;
- `view_type` agrees with its source file;
- `search_text` is present and non-empty;
- `metadata` and `payload` are JSON objects;
- every `source_id` refers to an original record;
- retrieval IDs are unique within the dataset;
- the expected counts remain 500 per view.

The dataset must be treated as immutable source material. Validation and indexing
should never rewrite these files.

## Research limitations

- The records are synthetic originals rather than a representative archive of all
  Bengali theatre traditions.
- Blocking and lighting annotations are useful for controlled prototyping but still
  require evaluation by theatre practitioners.
- Dataset balance across themes, genres, emotions, locations, and staging conditions
  should be measured before making comparative claims.
- Retrieval similarity does not establish artistic quality, cultural validity, or
  production feasibility.
- Human expert evaluation and real theatre material are required for stronger
  external-validity claims.

## Provenance and licensing

Each source record states that it is an original synthetic script generated for
research/RAG use and was not copied from an existing play. This provenance note is
not, by itself, a complete software or dataset license. Confirm distribution,
attribution, and reuse terms with the dataset owner before publishing or
redistributing the dataset.

When citing experimental results, record the dataset version or Git commit, the
retrieval view, embedding model, Qdrant collection, Top-K values, metadata filters,
LLM provider/model, prompt, validation outcome, and retrieval source IDs.
