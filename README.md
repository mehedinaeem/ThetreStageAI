# ThetreStageAI

ThetreStageAI is a research prototype for Bengali theatre-production intelligence. The planned system will use retrieval-augmented generation (RAG) to produce Bengali scenes, dialogue, actor blocking, stage directions, lighting directions, and structured lighting cues.

This repository currently contains only the project architecture and local-development configuration. Dataset ingestion, indexing, retrieval, generation, and evaluation behavior are intentionally not implemented in this phase.

## Prerequisites

- Python 3.11 or newer
- A local Ollama installation (needed in a later generation phase)

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies and configure the environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Then validate and start the project:

```bash
python manage.py check
python manage.py migrate
python manage.py runserver
```

## Architecture

- `thetrestageai/` contains project-wide Django configuration, routing, and ASGI/WSGI entry points.
- `theatre/` is the application boundary for theatre workflows, persistence, forms, HTTP views, and API routing.
- `theatre/services/` keeps domain operations out of Django views. Its subpackages isolate data preparation, RAG orchestration, retrieval, local LLM access, and output validation.
- `theatre/templates/` contains server-rendered Django templates; `static/` holds project-wide CSS, JavaScript, and images.
- `data/raw/` is reserved for immutable source material; `data/processed/` for normalized artifacts; and `data/retrieval_views/` for retrieval-oriented representations. Generated contents are ignored by Git.
- `storage/qdrant/` is the environment-configured persistence directory for Qdrant local mode. Its generated contents are ignored by Git.
- `evaluation/` will hold offline RAG and generation evaluation code and fixtures.
- `scripts/` will hold explicit research and maintenance entry points, separate from web requests.
- `tests/` is reserved for cross-application and integration tests; application-specific tests live under `theatre/tests/`.

## Configuration

Settings are read from `.env` and exposed through Django settings. Relative data and storage paths are resolved from `BASE_DIR`; callers should not hard-code filesystem locations. Never commit `.env`, local databases, model caches, source datasets, or Qdrant storage.

Ollama and embedding clients will be initialized lazily in later phases. Starting Django does not download an embedding model or contact Ollama.

## Inspecting the dataset

Set `THEATRE_DATASET_PATH` in `.env` to the directory containing
`bangla_natok_500.jsonl` and `retrieval_views/`, then run:

```bash
python manage.py inspect_theatre_dataset
```

An explicit read-only location can also be supplied without changing configuration:

```bash
python manage.py inspect_theatre_dataset --dataset-path /path/to/dataset
```

The command validates the three canonical retrieval-view files individually. It does
not load `all_retrieval_views.jsonl`, which would duplicate those records, and it
never writes to the dataset.

## Building the multi-view RAG index

With the full requirements installed, build the three persistent local Qdrant
collections with:

```bash
python manage.py build_rag_index
```

To explicitly delete and recreate only `thetrestageai_scene`,
`thetrestageai_blocking`, and `thetrestageai_lighting` before indexing:

```bash
python manage.py build_rag_index --rebuild
```

The command embeds only each document's `search_text` using normalized BAAI/bge-m3
vectors and cosine similarity. It preserves the complete retrieval document in the
Qdrant point payload. The embedding and upsert batch sizes can be tuned through
`EMBEDDING_BATCH_SIZE` and `QDRANT_UPSERT_BATCH_SIZE`.

## Testing multi-view retrieval

After building the index, run one Bengali request through three independently
constructed semantic queries:

```bash
python manage.py test_retrieval "দুইজন চরিত্রের রাগপূর্ণ পারিবারিক সংঘাত"
```

Scene retrieval returns five results, while blocking and lighting retrieval each
return three. Application code may also pass exact-match metadata filters such as
`theme`, `genre`, `scene_type`, `location`, `time`, `actors_count`, or `emotion` to
an individual retriever.

The `ContextBuilder` combines those result sets into size-bounded, clearly separated
reference sections and returns an IEEE-evaluation-friendly retrieval trace. Duplicate
sources are removed within each view, irrelevant metadata is excluded, and explicit
production rules prohibit copying retrieved dialogue verbatim. Context construction
does not call Ollama or any other language model.

## Local Ollama and Qwen setup

No paid API is required. On Linux, install Ollama using its official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the local server if it is not already running as a service:

```bash
ollama serve
```

In another terminal, download the configured Qwen model:

```bash
ollama pull qwen3:4b
ollama list
```

Configure `.env`:

```env
THETRESTAGEAI_OLLAMA_URL=http://localhost:11434
THETRESTAGEAI_LLM_MODEL=qwen3:4b
THETRESTAGEAI_LLM_TIMEOUT_SECONDS=180
```

The local client uses Ollama's non-streaming structured-output endpoint. It passes
the Pydantic JSON schema to Ollama and validates the returned JSON again locally.
Connection failures, timeouts, unavailable models, malformed API envelopes, Markdown
wrappers, invalid stage zones, and invalid cue values are rejected explicitly.

Generated JSON passes through a second strict validation boundary before it can be
used by any application or future hardware-control layer. Cross-record checks enforce
unique dialogue and lighting IDs, known speakers and actors, and cue triggers that
reference `scene_start`, `scene_end`, or a dialogue ID in the same scene. If initial
validation fails, exactly one schema-constrained correction is requested from the
configured local model. A second failure raises a controlled `ProductionValidationError`
containing both sets of validation issues; no partially valid lighting data is returned.

For macOS and Windows installers, use the official Ollama download page:
<https://ollama.com/download>.
# ThetreStageAI
