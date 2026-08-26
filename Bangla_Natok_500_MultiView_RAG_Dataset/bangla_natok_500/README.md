# Bangla Natok 500 - Updated Multi-View RAG Dataset

এই dataset-এ একই 500 synthetic Bangla natok record থেকে 3টি retrieval view তৈরি করা হয়েছে।

## Files

- `bangla_natok_500.jsonl` — original 500 complete theatre records
- `retrieval_views/scene_view.jsonl` — 500 scene retrieval documents
- `retrieval_views/blocking_view.jsonl` — 500 blocking retrieval documents
- `retrieval_views/lighting_view.jsonl` — 500 lighting retrieval documents
- `retrieval_views/all_retrieval_views.jsonl` — all 1500 retrieval documents together
- `scripts/` — 500 readable text scripts
- `qdrant_ingest_example.py` — BGE-M3 + Qdrant indexing example

## Recommended RAG design

User query
→ Scene Retriever (Top 5)
→ Blocking Retriever (Top 3)
→ Lighting Retriever (Top 3)
→ Context Builder
→ LLM
→ New Script + Blocking + Lighting JSON

## View document structure

Each retrieval document contains:

- `id` — unique view document ID
- `view_type` — scene / blocking / lighting
- `source_id` — original natok ID
- `search_text` — text to embed
- `metadata` — filters such as genre, theme, scene_type, emotion, actor count
- `payload` — structured source data to send to the LLM after retrieval

## Important research note

These 500 scripts and their blocking/lighting annotations are synthetic originals. They are useful for RAG pipeline development and controlled experiments, but final research evaluation should also include real Bangla theatre data and expert-validated blocking/lighting examples.
