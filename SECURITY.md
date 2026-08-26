# ThetreStageAI security and robustness review

Review date: 2026-08-27

## Scope and trust boundaries

The theatre dataset, Qdrant payloads, user prompts, signed browser form values,
Ollama responses, and previously stored JSON are treated as untrusted. Environment
configuration and management-command access remain trusted local-administrator
boundaries. This prototype has no hardware-control integration; only fully validated
lighting data may cross such a boundary in the future.

## Implemented controls

- Django forms bound story length, research-query length, numeric ranges, fixture
  count, and fixture-name length. Django request parsing is capped at 1 MiB and 100
  fields by default; declared oversized bodies receive a controlled HTTP 413 before
  form parsing.
- CSRF middleware remains enabled. Templates use Django autoescaping and contain no
  `safe`, `mark_safe`, or disabled-autoescape rendering. Stored production JSON is
  schema-validated again before rendering, including RGB values used in inline CSS.
- Retrieved content is placed inside explicit `UNTRUSTED_RETRIEVED_REFERENCE`
  delimiters. Both the system prompt and final production rules state that commands,
  system-message claims, schema changes, and prompt overrides inside user or
  retrieved content must be ignored. Nested metadata and payload content is depth,
  item, string, and total-context bounded before prompt construction.
- Malformed Qdrant points, non-object payloads, empty IDs/text, non-finite scores,
  and oversized fields are skipped with safe logging instead of failing the whole
  retrieval. Displayed retrieval traces are normalized and capped.
- JSONL records have a per-line byte limit and Pydantic field/list bounds. Source
  files are opened read-only. Writable Qdrant/data paths cannot be the filesystem or
  project root, and Qdrant storage cannot be placed inside the source dataset.
- Ollama HTTP response reads are bounded. Generated JSON rejects duplicate keys,
  non-standard numeric constants, oversized responses, extra fields, excessive
  strings/lists, unknown actors/speakers, invalid triggers/zones, and unsafe RGB,
  intensity, or fade values. One bounded correction is allowed, followed by a hard
  failure.
- Exports revalidate the complete production. Download filenames are server-created,
  response sniffing is disabled, and textual CSV cells beginning with spreadsheet
  formula characters are escaped. Bengali remains UTF-8; CSV uses a UTF-8 BOM.
- Production mode cannot use the known development secret. Clickjacking, content
  sniffing, referrer, SameSite, and HttpOnly settings are explicit; HTTPS redirect
  and secure cookies are environment-switchable for deployment.

## Important residual risks

### Prompt injection remains probabilistic — high impact

Prompt delimiters and instruction hierarchy reduce risk but cannot prove that a
local LLM will never follow hostile retrieved prose. Keep model output behind the
Pydantic boundary, never expose tools to the generation model, and never send cues
directly to lighting hardware. Dataset provenance and manual review remain important.

### No authentication or rate limiting — high if network-exposed

The application is intentionally a local research prototype. Anyone who can reach it
can inspect projects and trigger expensive embedding/model work. Bind the development
server and Ollama to trusted interfaces only. Add authentication, authorization,
rate limits, and a production WSGI/ASGI server before multi-user or public deployment;
they are not added prematurely here.

### Local data and model services — medium

SQLite, Qdrant local storage, dataset files, and Ollama have no application-level
encryption or integrity signature. A user with filesystem access can alter records or
indices. Use operating-system permissions, backups, dataset checksums, and isolated
service accounts when experiments require stronger provenance.

### Trusted configuration can redirect local access — medium

An administrator controls dataset paths and the Ollama URL. A malicious environment
configuration can point the process at unintended local resources. Do not accept
these settings from web requests, and protect deployment environment files.

### External frontend assets — medium for deployment

Bootstrap is currently loaded from a CDN. A production deployment should pin and
self-host reviewed assets or add suitable Subresource Integrity and Content Security
Policy controls. A strict CSP was not added because the current UI contains inline
RGB style variables and would require a coordinated frontend refactor.

### Resource exhaustion and concurrency — medium

Request, JSONL-line, context, selection-token, and model-response limits are present,
but repeated valid generation requests can still consume substantial CPU/RAM. SQLite
and local Qdrant also have limited concurrent-write behavior. Use queueing, process
limits, timeouts, rate limits, and monitoring when moving beyond a single researcher.

## Deployment checklist

Set a random `DJANGO_SECRET_KEY`, disable `DJANGO_DEBUG`, configure exact
`DJANGO_ALLOWED_HOSTS`, enable `DJANGO_SECURE_SSL_REDIRECT` and
`DJANGO_SECURE_COOKIES` behind HTTPS, restrict Ollama/Qdrant/filesystem access, and
run `python manage.py check --deploy`. Do not use Django's development server for an
internet-facing deployment.
