# Plan: pandoc DOCX (track-changes + comments) for `/text`, and `.eml` / `.msg` extraction

## Goals

1. **DOCX via pandoc on `/text`** — when the `/text` endpoint receives a `.docx`,
   extract text through **pandoc** so we can surface **tracked changes** and
   **comments**, instead of the current `Docx2txtLoader` which silently drops both.
2. **Email extraction** — support `.eml` and `.msg` so both `/text` and the
   embedding endpoints can read email bodies (and key headers).

---

## Current behavior (for context)

- `get_loader()` (`app/utils/document_loader.py`) routes by file extension /
  content-type and returns `(loader, known_type, file_ext)`. Routes consume it
  with `list(loader.lazy_load())`, so **every loader must implement `lazy_load()`**.
- `.docx`/`.doc` → `Docx2txtLoader` (plain text only; no tracked changes, no comments).
- The `/text` endpoint (`document_routes.py:1113`) calls `load_file_content(..., raw_text=True)`.
  `raw_text=True` already switches Markdown to a verbatim `TextLoader`; this is the
  established hook for "give me the raw content" and is where the DOCX change belongs.
- Unknown extensions fall through to `TextLoader` with `known_type=False` — so today
  `.eml` dumps raw headers+body as text and `.msg` produces binary garbage.
- `extract_text_from_documents()` simply concatenates `page_content` per document
  (special-casing only PDF cleanup).

### What's already available (no install needed for the common path)
- `pypandoc==1.15` is a dependency **and pandoc is installed in both `Dockerfile`
  and `Dockerfile.lite`**. The `/text` route already handles a "No pandoc" error
  (`ERROR_MESSAGES.PANDOC_NOT_INSTALLED`).
- `python-oxmsg` is already present (transitive via `unstructured`) → `.msg` works
  through `unstructured.partition.msg.partition_msg`.
- `langchain_community` exposes `UnstructuredEmailLoader` (`.eml`, has `lazy_load`).
- ⚠️ pandoc is **not** on the local Windows dev venv, so DOCX-via-pandoc and any
  pandoc-backed email rendering can only be exercised in Docker / Linux CI.

---

## Feature 1 — DOCX via pandoc on `/text`

### Approach
Add a small custom loader (e.g. `PandocDocxLoader`) in `document_loader.py` that
wraps `pypandoc.convert_file()` and implements `lazy_load()`/`load()` (yielding a
single `Document`). Route to it **only when `raw_text=True` and `file_ext == "docx"`**,
mirroring the existing Markdown `raw_text` branch:

```python
elif file_ext in ["doc", "docx"] or file_content_type in [...]:
    if raw_text and file_ext == "docx":
        loader = PandocDocxLoader(filepath)          # tracked changes + comments
    else:
        loader = Docx2txtLoader(filepath)            # embedding path unchanged
```

Conversion call (final flags TBD — see decisions):
```python
pypandoc.convert_file(path, to="markdown", format="docx",
                      extra_args=["--track-changes=all", "--wrap=none"])
```

### Key design decisions (need your input — see Open Questions)
- **`.docx` only.** pandoc reads OOXML `.docx`, **not** legacy binary `.doc`.
  `.doc` stays on `Docx2txtLoader`.
- **`--track-changes=all`** keeps insertions *and* deletions and is the only mode
  that emits comments. (`accept` = final text, `reject` = original; both drop comments.)
- **Output format.** `markdown` preserves change/comment annotations as spans
  (`[text]{.insertion}`, `[...]{.comment-start}`); `plain` is cleaner but **drops**
  the change/comment markup — which would defeat the feature.
- **Scope to `/text` only.** Embedding path keeps `Docx2txtLoader` so vector quality
  and behavior are unchanged.

### Pros
- ✅ Reuses an existing, already-shipped dependency (pypandoc + pandoc-in-Docker).
- ✅ Only pandoc reliably extracts Word **comments** and **tracked changes**.
- ✅ Change is surgical and isolated behind the existing `raw_text` flag — zero risk
  to embeddings.
- ✅ The route already degrades gracefully when pandoc is missing.

### Cons / risks
- ⚠️ **Not testable on local Windows dev** (no pandoc); relies on Docker/CI.
- ⚠️ Markdown output is **noisier** than `docx2txt` plain text — downstream consumers
  of `/text` must tolerate span/annotation syntax. (Mitigation: make track-changes
  mode / output format configurable via env, default to the richest.)
- ⚠️ Comment **anchoring** is approximate — pandoc inlines comment text near its
  range but exact author/timestamp fidelity varies by pandoc version.
- ⚠️ pandoc spawns a subprocess per file (latency + temp I/O) — fine for `/text`,
  another reason not to put it on the bulk embedding path.

---

## Feature 2 — `.eml` and `.msg` extraction

### `.eml` (RFC-822) — `UnstructuredEmailLoader`
Stdlib-based partitioning, no new dependency, already has `lazy_load()`. Verified
working locally.

- **Header note:** by default it returns the **body only** (no From/To/Subject/Date).
  We almost certainly want a small wrapper that prepends key headers to
  `page_content` so the `/text` output is useful for parsing.

### `.msg` (Outlook) — `unstructured.partition.msg` (python-oxmsg backend)
Already installed. Two implementation choices:

| Option | Backend | New dep? | Notes |
|---|---|---|---|
| **A. `partition_msg` via a tiny custom loader** (recommended) | `python-oxmsg` | **No** (already present) | Consistent with our `unstructured` stack; we control header handling + `lazy_load`. |
| B. `langchain_community.OutlookMessageLoader` | `extract_msg` | **Yes** (`extract_msg` not installed) | Off-the-shelf, but adds a dependency and gives less control. |

### Routing additions in `get_loader()`
```python
elif file_ext == "eml" or file_content_type == "message/rfc822":
    loader = EmailLoader(filepath)        # .eml wrapper (headers + body)
elif file_ext == "msg" or file_content_type == "application/vnd.ms-outlook":
    loader = OutlookMsgLoader(filepath)   # .msg via partition_msg
```
- Add `"msg"` to `_BINARY_FILE_EXTENSIONS` (binary; guard against text-mime hijack).
  `.eml` is text, so it does not go in that set.

### Pros
- ✅ `.eml` needs **no new dependency**; `.msg` reuses the already-installed
  `python-oxmsg` (Option A).
- ✅ Works for **both** `/text` and embedding endpoints once routed.
- ✅ Unstructured loaders already provide `lazy_load()`.

### Cons / risks
- ⚠️ Header inclusion is a **product decision** — strip vs. keep From/To/Subject/Date
  (and whether embeddings should include headers, which add noise to vectors).
- ⚠️ **Attachments** are out of scope by default (`process_attachments=False`).
  Extracting attachment text is a larger, separate effort.
- ⚠️ `.msg` rendering fidelity (HTML bodies, encodings) depends on `python-oxmsg`;
  edge cases (RTF-only bodies, winmail.dat) may extract poorly.
- ⚠️ Option B (`OutlookMessageLoader`) would add `extract_msg` and a second email
  code path — avoid unless we hit an oxmsg limitation.

---

## Dependencies & Docker impact
- **Feature 1:** none — pypandoc + pandoc already shipped. (Confirm pandoc present
  in any non-Docker deployment.)
- **Feature 2 (Option A):** none — `python-oxmsg` already transitively installed.
  Recommend pinning it **explicitly** in `requirements*.txt` so we don't depend on
  it remaining a transitive of `unstructured`.
- **Feature 2 (Option B):** adds `extract_msg` to `requirements*.txt` (not recommended).

## Files likely to change
- `app/utils/document_loader.py` — new `PandocDocxLoader`, `EmailLoader`,
  `OutlookMsgLoader`; routing branches; `_BINARY_FILE_EXTENSIONS += {"msg"}`.
- `app/routes/document_routes.py` — possibly map `.eml`/`.msg` content-types;
  error handling parity (the pandoc-missing branch already exists).
- `requirements.txt` / `requirements.lite.txt` — explicit `python-oxmsg` pin
  (and `extract_msg` only if Option B).
- `app/config.py` — optional env toggles (track-changes mode, include-headers).
- `tests/utils/` — new tests (see below).
- `README.md` — document newly supported types + any env vars.

## Testing plan
- **`.eml`:** unit test with a fixture email → asserts body (and headers if included).
  Runs locally (no pandoc/network).
- **`.msg`:** unit test with a small `.msg` fixture via `partition_msg`. Runs locally.
- **DOCX/pandoc:** build a `.docx` containing an insertion, a deletion, and a comment;
  assert all three appear. **Gate with `skipif(pandoc not available)`** so local
  Windows runs skip and Docker/Linux CI exercises it (mirrors existing pandoc skips
  in `tests/utils/test_lazy_load.py`).
- Add `.eml`/`.msg`/`.docx` cases to the `lazy_load()` parametrized suite.

---

## Decisions (locked) ✅
1. **DOCX track-changes mode** → `all` (insertions + deletions + comments).
   Configurable via `DOCX_TEXT_TRACK_CHANGES` (default `all`).
2. **DOCX output format** → `markdown` (preserves change/comment markup).
3. **Email headers** → **prepend** From/To/Cc/Subject/Date. Configurable via
   `EMAIL_INCLUDE_HEADERS` (default `True`); applies to both `/text` and embeddings.
4. **`.msg` backend** → **`python-oxmsg`** (no new dep, MIT, consistent with stack).
5. **Scope** → DOCX-via-pandoc on **`/text` only** (gated by `raw_text=True` and
   `DOCX_TEXT_USE_PANDOC`, default `True`); embedding path keeps `Docx2txtLoader`.

## Follow-up additions (this round)
- **DOCX edit author/date + comments** — already delivered by the original choice
  (Markdown output + `--track-changes=all`): pandoc records each insertion/
  deletion/comment span's `author`/`date`. No code change; documented above.
- **DOCX headers/footers** — pandoc drops them, so they're extracted separately via
  **python-docx** and prepended to the `/text` output (matter numbers,
  "PRIVILEGED & CONFIDENTIAL", "DRAFT"). Toggle: `DOCX_TEXT_INCLUDE_HEADERS_FOOTERS`
  (default `True`). Fails soft (returns "" on any read error).
- **`.rtf` support** — routed to `UnstructuredRTFLoader` (pandoc-backed; already in
  Docker). Added to `_BINARY_FILE_EXTENSIONS` so a stray `text/markdown` MIME can't
  hijack it. New dependency: `python-docx==1.1.2` (MIT).
- **`.wpd` (WordPerfect)** — explicitly **out of scope** (no Python lib; would need
  the `libwpd` system binary in Docker; niche format).
- **Standalone image OCR** — image uploads (`.png/.jpg/.jpeg/.gif/.bmp/.tif/.tiff/.webp`)
  now run through Mistral OCR via a new `ImageOCRLoader`, normalized to PNG with Pillow
  so any format works; **multi-page TIFF** (discovery productions) yields one Document
  per frame. The PDF OCR path is unchanged (no embedded-image extraction). Factored a
  shared `_run_mistral_ocr()` helper used by both the PDF and image loaders. New
  dependency: `Pillow>=10.0.0`.
  - Cleanup: removed the dead/no-op `PDF_EXTRACT_IMAGES` config + env var + README entry,
    and the vestigial `extract_images` param on `SafePyPDFLoader` (it was hardcoded off
    and unused). The PDF OCR behavior is unchanged.

## Code-review fixes (applied)
Correctness:
- **`.eml` routing**: added `eml` to `_BINARY_FILE_EXTENSIONS` so a stray
  `text/markdown` Content-Type can't hijack it away from `EmailLoader`.
- **`.msg` Bcc leak / Cc loss**: `OutlookMsgLoader` now builds headers from the
  message's transport headers (real To/Cc split, Bcc absent by nature) instead of
  `msg.recipients` (which oxmsg exposes without a recipient type, so it both
  collapsed To/Cc and could leak Bcc from Sent-Items files). From/Subject/Date
  fall back to the structured attributes.
- **Image OCR frame cap**: `ImageOCRLoader` stops after `IMAGE_OCR_MAX_PAGES`
  (default 100) frames, preventing unbounded OCR fan-out from an animated GIF or
  huge multi-page TIFF.
- **PDF/image page numbering**: OCR pages are now numbered 1-based sequentially
  (was passing Mistral's 0-based index straight through, which `process_documents`
  treated as "no page", dropping the first page's marker).

Efficiency / robustness:
- Split the OCR helper into `_mistral_ocr_client()` + `_ocr_document()`: the
  client is built **once** per file (was once per image frame), and the API-key
  check runs **before** any file decode/encode (fail-fast, fixes the image path's
  contract violation).
- `ImageOCRLoader` passes single-frame **PNG/JPEG through unmodified** (no Pillow
  decode/re-encode); only multi-frame/odd-mode images are normalized to PNG.
- Single-pass page numbering (removed the build-then-renumber second loop); zero-
  frame images still return a placeholder Document (≥1-doc invariant preserved).
- `EmailLoader` header parse uses `headersonly=True` (no second body parse).
- Pinned `python-oxmsg>=0.0.2,<0.1` (was unbounded on a 0.0.x package).

Deferred (larger refactors, noted not done): a single extension→content-type→loader
registry (the routing lists are still maintained per-branch), generalizing the
`raw_text` per-format special-casing, and a shared base loader for the
`lazy_load`/`_temp_filepath` boilerplate.

## Implementation status — DONE
- `app/config.py`: `DOCX_TEXT_USE_PANDOC`, `DOCX_TEXT_TRACK_CHANGES`, `EMAIL_INCLUDE_HEADERS`.
- `app/utils/document_loader.py`: new `PandocDocxLoader`, `EmailLoader`,
  `OutlookMsgLoader` (+ header helpers); routing branches for `.docx` raw_text /
  `.eml` / `.msg`; `"msg"` added to `_BINARY_FILE_EXTENSIONS`.
- `requirements.txt` / `requirements.lite.txt`: explicit `python-oxmsg` pin.
- Tests in `tests/utils/test_document_loader.py` (loaders + routing) and an `.eml`
  case in `tests/utils/test_lazy_load.py`. Full suite green (208 passed) except the
  pre-existing Windows-only symlink tests.
