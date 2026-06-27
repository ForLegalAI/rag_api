import os
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from app.utils.document_loader import get_loader, clean_text, process_documents
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


def test_clean_text():
    text = "Hello\x00World"
    cleaned = clean_text(text)
    assert "\x00" not in cleaned
    assert cleaned == "HelloWorld"


def test_get_loader_text(tmp_path):
    # Create a temporary text file.
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = loader.load()
    # Check that data is loaded.
    assert data is not None


def test_process_documents():
    docs = [
        Document(
            page_content="Page 1 content", metadata={"source": "dummy.txt", "page": 1}
        ),
        Document(
            page_content="Page 2 content", metadata={"source": "dummy.txt", "page": 2}
        ),
    ]
    processed = process_documents(docs)
    assert "dummy.txt" in processed
    assert "# PAGE 1" in processed
    assert "# PAGE 2" in processed


def test_safe_pdf_loader_class():
    """Test that SafePyPDFLoader class can be instantiated"""
    from app.utils.document_loader import SafePyPDFLoader

    # Test instantiation. Mistral OCR does not extract images, so the
    # extract_images flag is accepted for backward compatibility but forced off.
    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)
    assert loader.filepath == "dummy.pdf"
    assert loader.extract_images is False
    assert loader._temp_filepath is None


def test_get_loader_text_lazy_load(tmp_path):
    """Test that lazy_load returns an iterator yielding documents."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = list(loader.lazy_load())
    assert len(data) > 0
    assert hasattr(data[0], "page_content")


def test_get_loader_pdf(tmp_path):
    """Test get_loader returns SafePyPDFLoader for PDF files"""
    # Create a dummy PDF file path (doesn't need to be real for this test)
    file_path = tmp_path / "test.pdf"
    file_path.write_text("dummy content")  # Not a real PDF, but that's OK for this test

    loader, known_type, file_ext = get_loader(
        "test.pdf", "application/pdf", str(file_path)
    )

    # Check that we get our SafePyPDFLoader
    from app.utils.document_loader import SafePyPDFLoader

    assert isinstance(loader, SafePyPDFLoader)
    assert known_type is True
    assert file_ext == "pdf"


def test_safe_pdf_loader_lazy_load():
    """Test that SafePyPDFLoader.lazy_load() returns an Iterator."""
    from app.utils.document_loader import SafePyPDFLoader

    loader = SafePyPDFLoader("dummy.pdf", extract_images=False)
    assert hasattr(loader, "lazy_load")
    result = loader.lazy_load()
    assert isinstance(result, Iterator)


def _make_mistral_module(pages=None, raise_exc=None):
    """Build a fake ``mistralai`` module whose ``Mistral().ocr.process`` is stubbed.

    Returns ``(fake_module, client)`` so callers can both inject the module via
    ``sys.modules`` and make assertions on the OCR client.
    """
    fake_module = MagicMock()
    client = MagicMock()
    if raise_exc is not None:
        client.ocr.process.side_effect = raise_exc
    else:
        response = MagicMock()
        response.pages = pages
        client.ocr.process.return_value = response
    fake_module.Mistral.return_value = client
    return fake_module, client


def test_safe_pdf_loader_ocr_maps_pages():
    """load() maps Mistral OCR pages to Documents with their page index in metadata."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    page0 = MagicMock(index=0, markdown="page zero text")
    page1 = MagicMock(index=1, markdown="page one text")
    fake_module, client = _make_mistral_module(pages=[page0, page1])

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict("sys.modules", {"mistralai": fake_module}), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"):
        result = loader.load()

    assert [d.page_content for d in result] == ["page zero text", "page one text"]
    assert result[0].metadata == {"source": "dummy.pdf", "page": 0}
    assert result[1].metadata == {"source": "dummy.pdf", "page": 1}
    client.ocr.process.assert_called_once()


def test_safe_pdf_loader_lazy_load_yields_ocr_pages():
    """lazy_load() yields the same Documents that load() produces."""
    from app.utils.document_loader import SafePyPDFLoader

    docs = [Document(page_content="p1"), Document(page_content="p2")]
    loader = SafePyPDFLoader("dummy.pdf")
    with patch.object(SafePyPDFLoader, "load", return_value=docs):
        result = list(loader.lazy_load())

    assert result == docs


def test_safe_pdf_loader_empty_pages_returns_placeholder():
    """An OCR response with no pages yields a single empty placeholder Document."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    fake_module, _ = _make_mistral_module(pages=[])

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict("sys.modules", {"mistralai": fake_module}), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"):
        result = loader.load()

    assert len(result) == 1
    assert result[0].page_content == ""
    assert result[0].metadata == {"source": "dummy.pdf", "page": 1}


def test_safe_pdf_loader_ocr_error_propagates():
    """An error raised by the Mistral OCR API call propagates to the caller."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    fake_module, _ = _make_mistral_module(raise_exc=RuntimeError("OCR boom"))

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict("sys.modules", {"mistralai": fake_module}), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"):
        with pytest.raises(RuntimeError, match="OCR boom"):
            loader.load()


def test_safe_pdf_loader_requires_api_key():
    """load() raises a clear error when MISTRAL_API_KEY is not configured."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    fake_module, _ = _make_mistral_module(pages=[])

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict("sys.modules", {"mistralai": fake_module}), patch.object(
        document_loader, "MISTRAL_API_KEY", ""
    ):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            loader.load()


MARKDOWN_SAMPLE = (
    "# Heading\n\n"
    "**bold** and *italic* text with a [link](https://example.com).\n\n"
    "- item 1\n"
    "- item 2\n\n"
    "> a blockquote\n"
)


def test_get_loader_markdown_embed_uses_unstructured(tmp_path):
    """Default (embedding) path must keep UnstructuredMarkdownLoader for .md."""
    file_path = tmp_path / "notes.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "notes.md", "text/markdown", str(file_path)
    )

    assert isinstance(loader, UnstructuredMarkdownLoader)
    assert known_type is True
    assert file_ext == "md"


@pytest.mark.parametrize(
    "content_type",
    [
        "text/markdown",
        "text/x-markdown",
        "application/markdown",
        "application/x-markdown",
    ],
)
def test_get_loader_markdown_raw_text_uses_text_loader(tmp_path, content_type):
    """/text path (raw_text=True) must load .md verbatim so formatting survives."""
    file_path = tmp_path / "notes.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "notes.md", content_type, str(file_path), raw_text=True
    )

    assert isinstance(loader, TextLoader)
    assert known_type is True
    assert file_ext == "md"

    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == MARKDOWN_SAMPLE


def test_get_loader_markdown_raw_text_by_extension_only(tmp_path):
    """Extension-based detection must still kick in when content type is generic."""
    file_path = tmp_path / "README.md"
    file_path.write_text(MARKDOWN_SAMPLE, encoding="utf-8")

    loader, _, _ = get_loader(
        "README.md", "application/octet-stream", str(file_path), raw_text=True
    )

    assert isinstance(loader, TextLoader)


def test_get_loader_raw_text_leaves_pdf_alone(tmp_path):
    """raw_text must not disturb binary formats — PDF still uses the PDF loader."""
    from app.utils.document_loader import SafePyPDFLoader

    file_path = tmp_path / "doc.pdf"
    file_path.write_text("not a real pdf")

    loader, _, file_ext = get_loader(
        "doc.pdf", "application/pdf", str(file_path), raw_text=True
    )

    assert isinstance(loader, SafePyPDFLoader)
    assert file_ext == "pdf"


@pytest.mark.parametrize(
    "filename, expected_loader_name",
    [
        ("doc.pdf", "SafePyPDFLoader"),
        ("report.docx", "Docx2txtLoader"),
        ("book.epub", "UnstructuredEPubLoader"),
        ("data.xlsx", "UnstructuredExcelLoader"),
        ("slides.pptx", "UnstructuredPowerPointLoader"),
    ],
)
def test_get_loader_raw_text_respects_binary_extensions_over_markdown_mime(
    tmp_path, filename, expected_loader_name
):
    """A markdown Content-Type must not override a known binary extension.

    Some clients send conflicting multipart content types. For an upload named
    `doc.pdf` with Content-Type `text/markdown`, the PDF loader still has to
    win — otherwise a binary file is read as UTF-8 text.
    """
    file_path = tmp_path / filename
    file_path.write_text("placeholder binary content")

    loader, _, _ = get_loader(
        filename, "text/markdown", str(file_path), raw_text=True
    )

    assert type(loader).__name__ == expected_loader_name
