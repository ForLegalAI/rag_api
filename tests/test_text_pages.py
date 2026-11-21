"""/text: per-page output (``pages=true``) and status codes for OCR service failures.

The Mistral client is faked, but the loaders, the route and the SDK's own
exception types are real.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from mistralai.client.errors import NoResponseError, SDKError

from app.utils import document_loader
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def thread_pool():
    # The app creates its pool on startup, which TestClient without a context manager skips.
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)


@pytest.fixture
def auth_headers():
    os.environ["JWT_SECRET"] = "testsecret"
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, 'testsecret', algorithm='HS256')}"}


def _ocr_response(*markdowns):
    return SimpleNamespace(
        pages=[SimpleNamespace(markdown=md, images=[]) for md in markdowns]
    )


@pytest.fixture
def ocr(monkeypatch):
    """A fake Mistral client; set ``ocr.ocr.process.side_effect`` / ``return_value``."""
    fake = MagicMock()
    monkeypatch.setattr(document_loader, "_mistral_ocr_client", lambda: fake)
    monkeypatch.setattr(document_loader, "MISTRAL_OCR_EMBEDDED_IMAGES", False)
    return fake


def _post_text(path, filename, content_type, headers, **form):
    with open(path, "rb") as f:
        return client.post(
            "/text",
            data={"file_id": "f1", **form},
            files={"file": (filename, f, content_type)},
            headers=headers,
        )


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "contract.pdf"
    path.write_bytes(b"%PDF-1.4 not parsed locally; OCR is faked")
    return path


def _upstream_error(status_code, headers=None):
    response = httpx.Response(
        status_code, headers=headers, request=httpx.Request("POST", "https://api.mistral.ai/v1/ocr")
    )
    return SDKError("API error occurred", response, "upstream said no")


# --- pages=true -------------------------------------------------------------


def test_pdf_pages_numbered_and_text_unchanged(ocr, pdf_file, auth_headers):
    ocr.ocr.process.return_value = _ocr_response("first\x00 page", "second page", "third page")

    plain = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)
    paged = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers, pages="true")

    assert plain.status_code == 200 and paged.status_code == 200, paged.text
    assert "pages" not in plain.json()
    body = paged.json()
    assert body["pages"] == [
        {"page": 1, "text": "first page"},  # cleaned like the joined text (NUL removed)
        {"page": 2, "text": "second page"},
        {"page": 3, "text": "third page"},
    ]
    assert {k: v for k, v in body.items() if k != "pages"} == plain.json()
    assert body["text"] == "first page\nsecond page\nthird page"


def test_blank_pages_are_kept_with_empty_text(ocr, pdf_file, auth_headers):
    ocr.ocr.process.return_value = _ocr_response("cover", "", "body")

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers, pages="true")

    assert [p["page"] for p in response.json()["pages"]] == [1, 2, 3]
    assert response.json()["pages"][1]["text"] == ""


def test_multipage_tiff_returns_one_entry_per_frame(ocr, tmp_path, auth_headers):
    from PIL import Image

    path = tmp_path / "scan.tiff"
    frames = [Image.new("RGB", (8, 8), color) for color in ("white", "black", "gray")]
    frames[0].save(path, format="TIFF", save_all=True, append_images=frames[1:])
    ocr.ocr.process.side_effect = [_ocr_response(f"frame {n}") for n in (1, 2, 3)]

    response = _post_text(path, "scan.tiff", "image/tiff", auth_headers, pages="true")

    assert response.status_code == 200, response.text
    assert response.json()["pages"] == [
        {"page": 1, "text": "frame 1"},
        {"page": 2, "text": "frame 2"},
        {"page": 3, "text": "frame 3"},
    ]


def test_format_without_pages_returns_empty_list(tmp_path, auth_headers):
    pytest.importorskip("pypandoc")
    from docx import Document as DocxDocument

    path = tmp_path / "memo.docx"
    doc = DocxDocument()
    doc.add_paragraph("Smlouva o dílo")
    doc.save(path)
    docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    plain = _post_text(path, "memo.docx", docx_type, auth_headers)
    paged = _post_text(path, "memo.docx", docx_type, auth_headers, pages="true")

    assert paged.status_code == 200, paged.text
    assert paged.json()["pages"] == []
    assert paged.json()["text"] == plain.json()["text"]
    assert "Smlouva o dílo" in paged.json()["text"]


def test_pages_false_keeps_current_shape(tmp_path, auth_headers):
    path = tmp_path / "note.txt"
    path.write_text("hello")

    response = _post_text(path, "note.txt", "text/plain", auth_headers, pages="false")

    assert set(response.json()) == {"text", "file_id", "filename", "known_type"}


# --- OCR service failures -----------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 408, 500, 502, 503])
def test_transient_upstream_failure_answers_503(ocr, pdf_file, auth_headers, status_code):
    ocr.ocr.process.side_effect = _upstream_error(status_code)

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 503
    # Same detail as before, so callers matching on it keep working.
    assert response.json()["detail"].startswith("Error during text extraction: ")


def test_rate_limit_forwards_retry_after(ocr, pdf_file, auth_headers):
    ocr.ocr.process.side_effect = _upstream_error(429, headers={"Retry-After": "7"})

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection refused"),
        NoResponseError(),
    ],
)
def test_timeout_or_unreachable_answers_503(ocr, pdf_file, auth_headers, error):
    ocr.ocr.process.side_effect = error

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 503
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credentials_answer_502(ocr, pdf_file, auth_headers, status_code):
    ocr.ocr.process.side_effect = _upstream_error(status_code)

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 502


def test_ocr_not_configured_answers_500(monkeypatch, pdf_file, auth_headers):
    monkeypatch.setattr(document_loader, "MISTRAL_API_KEY", "")

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 500
    assert "MISTRAL_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("status_code", [400, 422])
def test_mistral_rejecting_the_document_stays_400(ocr, pdf_file, auth_headers, status_code):
    ocr.ocr.process.side_effect = _upstream_error(status_code)

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 400


def test_non_ocr_failure_stays_400(ocr, pdf_file, auth_headers):
    ocr.ocr.process.side_effect = ValueError("file is corrupt")

    response = _post_text(pdf_file, "contract.pdf", "application/pdf", auth_headers)

    assert response.status_code == 400
    assert response.json()["detail"] == "Error during text extraction: file is corrupt"


def test_failure_wrapped_by_another_exception_is_still_classified():
    try:
        try:
            raise _upstream_error(429)
        except SDKError as inner:
            raise RuntimeError("load failed") from inner
    except RuntimeError as outer:
        assert document_loader.classify_ocr_failure(outer) == (503, None)
