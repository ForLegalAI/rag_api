import os
from collections.abc import Iterator
from contextlib import ExitStack
from types import SimpleNamespace
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

    loader = SafePyPDFLoader("dummy.pdf")
    assert loader.filepath == "dummy.pdf"
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

    loader = SafePyPDFLoader("dummy.pdf")
    assert hasattr(loader, "lazy_load")
    result = loader.lazy_load()
    assert isinstance(result, Iterator)


def _make_mistral_module(pages=None, raise_exc=None):
    """Build a fake ``mistralai.client`` module whose ``Mistral().ocr.process`` is stubbed.

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
    """load() maps Mistral OCR pages to Documents with 1-based sequential page metadata."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    # Mistral indexes pages 0-based; we renumber to 1-based in response order.
    page0 = MagicMock(index=0, markdown="page zero text")
    page1 = MagicMock(index=1, markdown="page one text")
    fake_module, client = _make_mistral_module(pages=[page0, page1])

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"):
        result = loader.load()

    assert [d.page_content for d in result] == ["page zero text", "page one text"]
    assert result[0].metadata == {"source": "dummy.pdf", "page": 1}
    assert result[1].metadata == {"source": "dummy.pdf", "page": 2}
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
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
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
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
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
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", ""
    ):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            loader.load()


def test_mistral_client_uses_sdk_default_server_without_override():
    """Without MISTRAL_SERVER_URL the client is built with the SDK's default server."""
    from app.utils import document_loader

    fake_module, _ = _make_mistral_module(pages=[])

    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(document_loader, "MISTRAL_SERVER_URL", ""):
        document_loader._mistral_ocr_client()

    fake_module.Mistral.assert_called_once_with(api_key="test-key")


def test_mistral_client_uses_configured_server_url():
    """MISTRAL_SERVER_URL (e.g. the EU regional endpoint) is passed to the client."""
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    fake_module, client = _make_mistral_module(pages=[])

    loader = SafePyPDFLoader("dummy.pdf")
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(
        document_loader, "MISTRAL_SERVER_URL", "https://api.eu.mistral.ai"
    ), patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"):
        loader.load()

    fake_module.Mistral.assert_called_once_with(
        api_key="test-key", server_url="https://api.eu.mistral.ai"
    )
    client.ocr.process.assert_called_once()


# --- Embedded-image OCR inside PDFs -----------------------------------------


def _ocr_page(markdown, images=()):
    return SimpleNamespace(markdown=markdown, images=list(images))


def _ocr_image(image_id, image_base64):
    return SimpleNamespace(id=image_id, image_base64=image_base64)


def _load_pdf_with_images(pages, image_results=None, **settings):
    """Load a PDF through a fake Mistral client that serves per-image OCR results.

    ``image_results`` maps an image's base64 payload to the text its OCR call
    returns (a str, a list of page texts, or an Exception to raise). ``settings``
    override ``document_loader`` module settings (defaults are pinned so tests
    don't depend on the environment). Returns ``(documents, client)``.
    """
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    settings = {
        "MISTRAL_OCR_EMBEDDED_IMAGES": True,
        "MISTRAL_OCR_IMAGE_MIN_SIZE": 100,
        "MISTRAL_OCR_MAX_EMBEDDED_IMAGES": 50,
        "MISTRAL_OCR_IMAGE_CONCURRENCY": 4,
        **settings,
    }
    results = image_results or {}

    def process(**kwargs):
        document = kwargs["document"]
        if document["type"] == "document_url":
            return pages if isinstance(pages, dict) else SimpleNamespace(pages=pages)
        outcome = results[document["image_url"].split(",", 1)[1]]
        if isinstance(outcome, Exception):
            raise outcome
        texts = outcome if isinstance(outcome, list) else [outcome]
        return SimpleNamespace(pages=[SimpleNamespace(markdown=t) for t in texts])

    fake_module, client = _make_mistral_module(pages=[])
    client.ocr.process.side_effect = process

    with ExitStack() as stack:
        stack.enter_context(
            patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module})
        )
        stack.enter_context(patch.object(document_loader, "MISTRAL_API_KEY", "test-key"))
        stack.enter_context(
            patch.object(SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data")
        )
        for name, value in settings.items():
            stack.enter_context(patch.object(document_loader, name, value))
        documents = SafePyPDFLoader("scan.pdf").load()
    return documents, client


def _image_calls(client):
    return [
        c.kwargs
        for c in client.ocr.process.call_args_list
        if c.kwargs["document"]["type"] == "image_url"
    ]


def test_pdf_embedded_image_text_replaces_placeholder():
    """The image's OCR text takes the placeholder's place; page text and metadata stay."""
    pages = [
        _ocr_page(
            "Before\n\n![img-0.jpeg](img-0.jpeg)\n\nAfter",
            [_ocr_image("img-0.jpeg", "AAA")],
        )
    ]

    docs, client = _load_pdf_with_images(pages, {"AAA": "  Text from the scan  "})

    assert len(docs) == 1
    assert docs[0].page_content == "Before\n\nText from the scan\n\nAfter"
    assert docs[0].metadata == {"source": "scan.pdf", "page": 1}
    assert "![img-0.jpeg]" not in docs[0].page_content
    assert client.ocr.process.call_count == 2


def test_pdf_embedded_images_request_kwargs_when_enabled():
    """The main request asks for image bytes with the configured size and limit."""
    docs, client = _load_pdf_with_images(
        [_ocr_page("text")],
        MISTRAL_OCR_EMBEDDED_IMAGES=True,
        MISTRAL_OCR_IMAGE_MIN_SIZE=123,
        MISTRAL_OCR_MAX_EMBEDDED_IMAGES=7,
    )

    client.ocr.process.assert_called_once()
    kwargs = client.ocr.process.call_args.kwargs
    assert kwargs["document"]["type"] == "document_url"
    assert kwargs["include_image_base64"] is True
    assert kwargs["image_min_size"] == 123
    assert kwargs["image_limit"] == 7


def test_pdf_embedded_images_no_image_limit_when_uncapped():
    _, client = _load_pdf_with_images(
        [_ocr_page("text")],
        MISTRAL_OCR_EMBEDDED_IMAGES=True,
        MISTRAL_OCR_MAX_EMBEDDED_IMAGES=0,
    )

    kwargs = client.ocr.process.call_args.kwargs
    assert kwargs["include_image_base64"] is True
    assert "image_limit" not in kwargs


def test_pdf_embedded_images_disabled_keeps_original_request():
    """With the feature off the request is as before and no image is OCR'd."""
    pages = [_ocr_page("![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "AAA")])]

    docs, client = _load_pdf_with_images(
        pages, {"AAA": "never read"}, MISTRAL_OCR_EMBEDDED_IMAGES=False
    )

    client.ocr.process.assert_called_once()
    kwargs = client.ocr.process.call_args.kwargs
    assert kwargs["include_image_base64"] is False
    assert "image_min_size" not in kwargs
    assert "image_limit" not in kwargs
    assert docs[0].page_content == "![img-0.jpeg](img-0.jpeg)"


@pytest.mark.parametrize("image_text", ["![img-0.jpeg](img-0.jpeg)", "   ", ""])
def test_pdf_embedded_image_without_text_keeps_placeholder(image_text):
    """An image that yields only placeholders/whitespace leaves the original placeholder."""
    pages = [_ocr_page("A ![img-0.jpeg](img-0.jpeg) B", [_ocr_image("img-0.jpeg", "AAA")])]

    docs, _ = _load_pdf_with_images(pages, {"AAA": image_text})

    assert docs[0].page_content == "A ![img-0.jpeg](img-0.jpeg) B"


def test_pdf_embedded_image_nested_placeholders_are_stripped():
    pages = [_ocr_page("![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "AAA")])]

    docs, _ = _load_pdf_with_images(pages, {"AAA": "Real text ![img-0.jpeg](img-0.jpeg)"})

    assert docs[0].page_content.strip() == "Real text"


def test_pdf_embedded_image_multiple_ocr_pages_joined_with_blank_line():
    pages = [_ocr_page("![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "AAA")])]

    docs, _ = _load_pdf_with_images(pages, {"AAA": ["first", "second"]})

    assert docs[0].page_content.strip() == "first\n\nsecond"


def test_pdf_embedded_image_failure_keeps_placeholder_and_other_images(caplog):
    """One failed image OCR call doesn't fail the document or the other images."""
    pages = [
        _ocr_page(
            "![img-0.jpeg](img-0.jpeg) | ![img-1.jpeg](img-1.jpeg)",
            [_ocr_image("img-0.jpeg", "SECRETB64"), _ocr_image("img-1.jpeg", "BBB")],
        )
    ]

    with caplog.at_level("WARNING"):
        docs, _ = _load_pdf_with_images(
            pages, {"SECRETB64": RuntimeError("boom"), "BBB": "second image"}
        )

    # A non-fatal image failure is a warning, never an error.
    assert [r.levelname for r in caplog.records if "img-0.jpeg" in r.getMessage()] == [
        "WARNING"
    ]
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    content = docs[0].page_content
    assert "![img-0.jpeg](img-0.jpeg)" in content
    assert "second image" in content
    assert "img-0.jpeg" in caplog.text and "scan.pdf" in caplog.text
    assert "SECRETB64" not in caplog.text


def test_pdf_main_ocr_failure_still_raises_and_logs_error(caplog):
    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    fake_module, _ = _make_mistral_module(raise_exc=RuntimeError("OCR boom"))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(document_loader, "MISTRAL_API_KEY", "test-key"), patch.object(
        document_loader, "MISTRAL_OCR_EMBEDDED_IMAGES", True
    ), patch.object(
        SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"
    ), caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="OCR boom"):
        SafePyPDFLoader("scan.pdf").load()

    assert [r.levelname for r in caplog.records if "OCR boom" in r.getMessage()] == ["ERROR"]


def test_pdf_embedded_image_base64_without_prefix_gets_data_uri():
    pages = [
        _ocr_page(
            "![a.png](a.png) ![b.jpg](b.jpg) ![c.avif](c.avif) ![d.jpeg](d.jpeg)",
            [
                _ocr_image("a.png", "AAA"),
                _ocr_image("b.jpg", "BBB"),
                _ocr_image("c.avif", "CCC"),
                _ocr_image("d.jpeg", "data:image/webp;base64,DDD"),
            ],
        )
    ]

    _, client = _load_pdf_with_images(
        pages, {"AAA": "a", "BBB": "b", "CCC": "c", "DDD": "d"}
    )

    sent = sorted(c["document"]["image_url"] for c in _image_calls(client))
    assert sent == [
        "data:image/jpeg;base64,BBB",
        "data:image/jpeg;base64,CCC",
        "data:image/png;base64,AAA",
        "data:image/webp;base64,DDD",
    ]


def test_pdf_embedded_image_cap_limits_image_ocr_calls(caplog):
    pages = [
        _ocr_page(
            "![img-0.jpeg](img-0.jpeg) ![img-1.jpeg](img-1.jpeg)",
            [_ocr_image("img-0.jpeg", "AAA"), _ocr_image("img-1.jpeg", "BBB")],
        )
    ]

    with caplog.at_level("WARNING"):
        docs, client = _load_pdf_with_images(
            pages,
            {"AAA": "first", "BBB": "second"},
            MISTRAL_OCR_MAX_EMBEDDED_IMAGES=1,
        )

    assert len(_image_calls(client)) == 1
    assert "first" in docs[0].page_content
    assert "![img-1.jpeg](img-1.jpeg)" in docs[0].page_content
    assert "MISTRAL_OCR_MAX_EMBEDDED_IMAGES" in caplog.text


def test_pdf_embedded_images_reaching_cap_warns_that_more_may_exist(caplog):
    pages = [_ocr_page("![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "AAA")])]

    with caplog.at_level("WARNING"):
        docs, client = _load_pdf_with_images(
            pages, {"AAA": "text"}, MISTRAL_OCR_MAX_EMBEDDED_IMAGES=1
        )

    assert len(_image_calls(client)) == 1
    assert "may not have been extracted" in caplog.text


def test_pdf_embedded_image_only_normalizes_newlines_around_placeholder():
    pages = [
        _ocr_page(
            "![img-0.jpeg](img-0.jpeg)\n\nkept\n\n\n\nspacing",
            [_ocr_image("img-0.jpeg", "AAA")],
        )
    ]

    docs, _ = _load_pdf_with_images(pages, {"AAA": "text"})

    assert docs[0].page_content == "text\n\nkept\n\n\n\nspacing"


def test_pdf_embedded_image_ids_are_scoped_per_page():
    """Every page may have its own ``img-0.jpeg``; each gets its own text."""
    pages = [
        _ocr_page("P1 ![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "AAA")]),
        _ocr_page("P2 ![img-0.jpeg](img-0.jpeg)", [_ocr_image("img-0.jpeg", "BBB")]),
    ]

    docs, _ = _load_pdf_with_images(
        pages, {"AAA": "text one", "BBB": "text two"}, MISTRAL_OCR_IMAGE_CONCURRENCY=2
    )

    assert "text one" in docs[0].page_content and "text two" not in docs[0].page_content
    assert "text two" in docs[1].page_content and "text one" not in docs[1].page_content
    assert [d.metadata["page"] for d in docs] == [1, 2]


def test_pdf_embedded_image_text_lands_at_own_placeholder_when_calls_finish_out_of_order():
    """The first image's OCR call only finishes after the second one has."""
    import threading

    from app.utils import document_loader
    from app.utils.document_loader import SafePyPDFLoader

    second_done = threading.Event()

    def process(**kwargs):
        document = kwargs["document"]
        if document["type"] == "document_url":
            return SimpleNamespace(
                pages=[
                    _ocr_page(
                        "![img-0.jpeg](img-0.jpeg)\n\n![img-1.jpeg](img-1.jpeg)",
                        [_ocr_image("img-0.jpeg", "AAA"), _ocr_image("img-1.jpeg", "BBB")],
                    )
                ]
            )
        if document["image_url"].endswith("AAA"):
            assert second_done.wait(timeout=10), "second image OCR never completed"
            text = "first text"
        else:
            text = "second text"
            second_done.set()
        return SimpleNamespace(pages=[SimpleNamespace(markdown=text)])

    fake_module, client = _make_mistral_module(pages=[])
    client.ocr.process.side_effect = process

    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(document_loader, "MISTRAL_API_KEY", "test-key"), patch.object(
        document_loader, "MISTRAL_OCR_EMBEDDED_IMAGES", True
    ), patch.object(
        document_loader, "MISTRAL_OCR_IMAGE_CONCURRENCY", 2
    ), patch.object(
        SafePyPDFLoader, "_encode_pdf_b64", return_value="b64data"
    ):
        docs = SafePyPDFLoader("scan.pdf").load()

    assert docs[0].page_content == "first text\n\nsecond text"


def test_pdf_embedded_image_without_placeholder_is_appended():
    pages = [_ocr_page("Page body", [_ocr_image("img-0.jpeg", "AAA")])]

    docs, _ = _load_pdf_with_images(pages, {"AAA": "hidden text"})

    assert docs[0].page_content == "Page body\n\nhidden text"


def test_pdf_embedded_images_support_dict_shaped_response():
    response = {
        "pages": [
            {
                "index": 0,
                "markdown": "Top ![img-0.jpeg](img-0.jpeg)",
                "images": [{"id": "img-0.jpeg", "image_base64": "AAA"}],
            }
        ]
    }

    docs, _ = _load_pdf_with_images(response, {"AAA": "from dict"})

    assert docs[0].page_content.split() == ["Top", "from", "dict"]


def test_pdf_embedded_images_skip_images_without_bytes():
    pages = [
        _ocr_page(
            "![img-0.jpeg](img-0.jpeg)",
            [_ocr_image("img-0.jpeg", None), _ocr_image("img-1.jpeg", "")],
        )
    ]

    docs, client = _load_pdf_with_images(pages)

    client.ocr.process.assert_called_once()
    assert docs[0].page_content == "![img-0.jpeg](img-0.jpeg)"


def test_image_ocr_loader_does_not_request_embedded_images(tmp_path):
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))
    fake_module, client = _make_mistral_module(pages=[MagicMock(index=0, markdown="t")])

    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(document_loader, "MISTRAL_API_KEY", "test-key"):
        ImageOCRLoader(str(p)).load()

    kwargs = client.ocr.process.call_args.kwargs
    assert kwargs["include_image_base64"] is False
    assert "image_min_size" not in kwargs


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
        # .docx on the /text path (raw_text=True) routes to pandoc for tracked
        # changes/comments; the markdown Content-Type must not override that.
        ("report.docx", "PandocDocxLoader"),
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


# ---------------------------------------------------------------------------
# DOCX via pandoc (/text endpoint)
# ---------------------------------------------------------------------------


def test_get_loader_docx_embed_uses_docx2txt(tmp_path):
    """Embedding path (raw_text=False) keeps Docx2txtLoader for .docx."""
    from langchain_community.document_loaders import Docx2txtLoader

    file_path = tmp_path / "report.docx"
    file_path.write_bytes(b"placeholder")

    loader, known_type, file_ext = get_loader(
        "report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        str(file_path),
    )
    assert isinstance(loader, Docx2txtLoader)
    assert known_type is True
    assert file_ext == "docx"


def test_get_loader_docx_raw_text_uses_pandoc(tmp_path):
    """/text path (raw_text=True) routes .docx to PandocDocxLoader."""
    from app.utils.document_loader import PandocDocxLoader

    file_path = tmp_path / "report.docx"
    file_path.write_bytes(b"placeholder")

    loader, _, file_ext = get_loader(
        "report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        str(file_path),
        raw_text=True,
    )
    assert isinstance(loader, PandocDocxLoader)
    assert file_ext == "docx"


def test_get_loader_docx_with_msword_mime_still_loads(tmp_path):
    """A valid .docx sent with the generic application/msword mime is NOT rejected."""
    from langchain_community.document_loaders import Docx2txtLoader

    file_path = tmp_path / "report.docx"
    file_path.write_bytes(b"placeholder")

    loader, known_type, file_ext = get_loader(
        "report.docx", "application/msword", str(file_path)
    )
    assert isinstance(loader, Docx2txtLoader)
    assert file_ext == "docx"


def test_get_loader_legacy_doc_is_rejected(tmp_path):
    """Legacy binary .doc can't be loaded by any backend, so it's rejected clearly."""
    file_path = tmp_path / "legacy.doc"
    file_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Legacy .doc files are not supported"):
        get_loader("legacy.doc", "application/msword", str(file_path))


def test_get_loader_legacy_doc_rejected_even_with_markdown_mime(tmp_path):
    """A markdown Content-Type must not route .doc into the markdown loader either."""
    file_path = tmp_path / "legacy.doc"
    file_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Legacy .doc files are not supported"):
        get_loader("legacy.doc", "text/markdown", str(file_path), raw_text=True)


def test_pandoc_docx_loader_passes_track_changes_and_builds_document(tmp_path):
    """PandocDocxLoader converts via pypandoc with --track-changes and wraps output."""
    from app.utils.document_loader import PandocDocxLoader

    file_path = tmp_path / "doc.docx"
    file_path.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "converted [text]{.insertion} markdown"

    loader = PandocDocxLoader(str(file_path), track_changes="all")
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "converted [text]{.insertion} markdown"
    assert docs[0].metadata["source"] == str(file_path)

    _, kwargs = fake_pypandoc.convert_file.call_args
    assert kwargs["to"].startswith("markdown")
    assert kwargs["format"] == "docx"
    assert "--track-changes=all" in kwargs["extra_args"]


def test_pandoc_docx_loader_lazy_load_yields_from_load(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    loader = PandocDocxLoader(str(tmp_path / "doc.docx"))
    docs = [Document(page_content="x")]
    with patch.object(PandocDocxLoader, "load", return_value=docs):
        result = list(loader.lazy_load())
    assert result == docs


def _make_docx_with_header_footer(path, header, footer, body="Body paragraph."):
    from docx import Document as Docx

    doc = Docx()
    section = doc.sections[0]
    section.header.paragraphs[0].text = header
    section.footer.paragraphs[0].text = footer
    doc.add_paragraph(body)
    doc.save(path)


def test_extract_docx_headers_footers(tmp_path):
    from app.utils.document_loader import _extract_docx_headers_footers

    p = tmp_path / "doc.docx"
    _make_docx_with_header_footer(
        str(p), "PRIVILEGED & CONFIDENTIAL", "Matter No. 12345"
    )

    block = _extract_docx_headers_footers(str(p))
    assert "[Header] PRIVILEGED & CONFIDENTIAL" in block
    assert "[Footer] Matter No. 12345" in block


def test_extract_docx_headers_footers_handles_bad_file(tmp_path):
    """Non-docx input must not raise; returns an empty string so body load survives."""
    from app.utils.document_loader import _extract_docx_headers_footers

    p = tmp_path / "bad.docx"
    p.write_bytes(b"not a real docx")
    assert _extract_docx_headers_footers(str(p)) == ""


def test_pandoc_docx_loader_prepends_headers_footers(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    _make_docx_with_header_footer(
        str(p), "PRIVILEGED & CONFIDENTIAL", "Matter No. 12345"
    )

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "Body markdown from pandoc."

    loader = PandocDocxLoader(str(p), include_headers_footers=True)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        docs = loader.load()

    text = docs[0].page_content
    assert "PRIVILEGED & CONFIDENTIAL" in text
    assert "Matter No. 12345" in text
    assert "Body markdown from pandoc." in text
    # Header/footer block precedes the pandoc body
    assert text.index("PRIVILEGED") < text.index("Body markdown")


def test_pandoc_docx_loader_headers_footers_disabled(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    _make_docx_with_header_footer(str(p), "CONFIDENTIAL", "Matter No. 999")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "Body markdown only."

    loader = PandocDocxLoader(str(p), include_headers_footers=False)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        docs = loader.load()

    assert docs[0].page_content == "Body markdown only."
    assert "CONFIDENTIAL" not in docs[0].page_content


# ---------------------------------------------------------------------------
# DOCX table compaction and Markdown cleanup
# ---------------------------------------------------------------------------


def test_pandoc_markdown_format_compact_disables_padded_tables():
    """Compact style turns off the table syntaxes that pad every cell."""
    from app.utils.document_loader import _pandoc_markdown_format

    fmt = _pandoc_markdown_format(table_style="compact", strip_heading_anchors=True)
    assert fmt.startswith("markdown")
    for extension in ("grid_tables", "multiline_tables", "simple_tables"):
        assert f"-{extension}" in fmt
    assert "-header_attributes" in fmt


def test_pandoc_markdown_format_grid_is_pandoc_default():
    """Grid style restores pandoc's own writer, as an escape hatch."""
    from app.utils.document_loader import _pandoc_markdown_format

    assert (
        _pandoc_markdown_format(table_style="grid", strip_heading_anchors=False)
        == "markdown"
    )


def test_pandoc_docx_loader_requests_compact_tables(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    p.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "body"

    loader = PandocDocxLoader(str(p), include_headers_footers=False)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        loader.load()

    _, kwargs = fake_pypandoc.convert_file.call_args
    assert "-grid_tables" in kwargs["to"]
    assert "--wrap=none" in kwargs["extra_args"]


def test_pandoc_docx_loader_grid_style_opt_out(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    p.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "body"

    loader = PandocDocxLoader(
        str(p),
        include_headers_footers=False,
        table_style="grid",
        strip_heading_anchors=False,
    )
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        loader.load()

    _, kwargs = fake_pypandoc.convert_file.call_args
    assert kwargs["to"] == "markdown"


def test_drop_empty_table_columns_removes_unused_columns():
    """A bilingual table with spare gutter columns keeps only the filled ones."""
    from app.utils.document_loader import _drop_empty_table_columns

    table = (
        "| výpověď | Notice of Termination |  |  |\n"
        "|---------|-----------------------|--|--|\n"
        "| česky   | in English            |  |  |\n"
    )
    out = _drop_empty_table_columns(table)

    assert "výpověď" in out and "Notice of Termination" in out
    assert "česky" in out and "in English" in out
    for line in out.strip().split("\n"):
        assert line.count("|") == 3, line


def test_drop_empty_table_columns_keeps_partially_filled_column():
    from app.utils.document_loader import _drop_empty_table_columns

    table = "| a | b |\n|---|---|\n| x |   |\n|   | y |\n"
    assert _drop_empty_table_columns(table) == table


def test_drop_empty_table_columns_keeps_spacer_rows_intact():
    """A row of empty cells is a spacer, and must not be rewritten as a rule."""
    from app.utils.document_loader import _drop_empty_table_columns

    table = "| a | b |  |\n|---|---|--|\n|   |   |  |\n| c | d |  |\n"
    out = _drop_empty_table_columns(table).strip().split("\n")

    assert out[1] == "| --- | --- |"  # the one rule the table had
    assert out[2] == "|  |  |"  # spacer row stays a spacer row
    assert out[3] == "| c | d |"


def test_drop_empty_table_columns_leaves_non_tables_untouched():
    from app.utils.document_loader import _drop_empty_table_columns

    text = "Plain paragraph.\n\n<table>\n<tr><td>a</td><td></td></tr>\n</table>\n"
    assert _drop_empty_table_columns(text) == text


def test_drop_empty_table_columns_respects_escaped_pipes():
    """A literal pipe in cell text must not be read as a column boundary."""
    from app.utils.document_loader import _drop_empty_table_columns

    table = "| a \\| b | c |  |\n|--------|---|--|\n| d      | e |  |\n"
    out = _drop_empty_table_columns(table)

    assert "a \\| b" in out
    assert out.strip().split("\n")[0].count("|") == 4  # 2 columns + escaped pipe


def test_clean_pandoc_markdown_strips_empty_change_spans():
    """A deleted space costs ~70 bytes of metadata and says nothing; drop it."""
    from app.utils.document_loader import _clean_pandoc_markdown

    text = (
        '§ 52 písm. g) []{.deletion author="Jaroslav Skubal" '
        'date="2026-09-14T17:08:00Z"}a f) zákoníku práce'
    )
    out = _clean_pandoc_markdown(text)

    assert ".deletion" not in out
    assert out == "§ 52 písm. g) a f) zákoníku práce"


def test_clean_pandoc_markdown_preserves_tracked_changes_with_text():
    """Insertions/deletions that carry text keep their author and date verbatim."""
    from app.utils.document_loader import _clean_pandoc_markdown

    text = (
        '[inserted]{.insertion author="Jaroslav Skubal" date="2026-09-14T17:08:00Z"} '
        '[deleted]{.deletion author="Jaroslav Skubal" date="2026-09-14T17:08:00Z"}'
    )
    assert _clean_pandoc_markdown(text) == text


def test_clean_pandoc_markdown_preserves_comment_anchors():
    """Comment ranges need their (empty) end anchor to stay delimited."""
    from app.utils.document_loader import _clean_pandoc_markdown

    text = (
        '[clause]{.comment-start id="1" author="Jaroslav Skubal"} text '
        '[]{.comment-end id="1"}'
    )
    assert _clean_pandoc_markdown(text) == text


def test_clean_pandoc_markdown_collapses_blank_lines_and_invisibles():
    """Word spacer paragraphs and invisible characters carry no content."""
    from app.utils.document_loader import _clean_pandoc_markdown

    nbsp, soft_hyphen, zero_width = "\u00a0", "\u00ad", "\u200b"
    text = f"First\n\n\n\n\nSecond{nbsp}para{soft_hyphen}graph{zero_width}."

    # NBSP becomes a real space so "§ 52" stays searchable; the soft hyphen and
    # zero-width space are dropped, leaving the single word Word meant to render.
    assert _clean_pandoc_markdown(text) == "First\n\nSecond paragraph."


def test_clean_pandoc_markdown_keeps_toc_by_default():
    from app.utils.document_loader import _clean_pandoc_markdown

    text = "[Article 1 Definitions](#_Toc123456789)\n\nBody."
    assert "#_Toc123456789" in _clean_pandoc_markdown(text)
    assert "Article 1" not in _clean_pandoc_markdown(text, drop_toc=True)


def test_pandoc_docx_loader_cleanup_can_be_disabled(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    p.write_bytes(b"placeholder")
    raw = 'text []{.deletion author="A" date="D"}here\n\n\n\nmore'

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = raw

    loader = PandocDocxLoader(str(p), include_headers_footers=False, cleanup=False)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        docs = loader.load()

    assert docs[0].page_content == raw


def test_pandoc_docx_loader_cleans_output_by_default(tmp_path):
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    p.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = (
        'text []{.deletion author="A" date="D"}here'
    )

    loader = PandocDocxLoader(str(p), include_headers_footers=False)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}):
        docs = loader.load()

    assert docs[0].page_content == "text here"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("4", True),
        ("- 12 -", True),
        ("Page 4 of 10", True),
        ("Strana 4/10", True),
        ("Matter No. 12345", False),
        ("12345", False),  # a numeric matter number is not a page number
        ("§ 3", False),  # an article running header, not a page number
        ("(5)", False),
        ("PRIVILEGED & CONFIDENTIAL", False),
        ("Article 5", False),
    ],
)
def test_is_page_number_only(text, expected):
    from app.utils.document_loader import _is_page_number_only

    assert _is_page_number_only(text) is expected


def test_extract_docx_headers_footers_skips_page_numbers(tmp_path):
    """A footer holding only a page number adds no content, so it is dropped."""
    from app.utils.document_loader import _extract_docx_headers_footers

    p = tmp_path / "doc.docx"
    _make_docx_with_header_footer(str(p), "PRIVILEGED & CONFIDENTIAL", "4")

    block = _extract_docx_headers_footers(str(p))
    assert "[Header] PRIVILEGED & CONFIDENTIAL" in block
    assert "[Footer]" not in block


def test_drop_toc_keeps_in_body_cross_references():
    """Word reuses _Toc bookmarks for cross-references; a clause is not a TOC entry."""
    from app.utils.document_loader import _clean_pandoc_markdown

    clause = (
        "As set out in [Section 3.2 (Termination)](#_Toc123456789) below, the "
        "Employer may terminate this Agreement immediately."
    )
    assert _clean_pandoc_markdown(clause, drop_toc=True) == clause


def test_drop_toc_removes_generated_toc_lines():
    from app.utils.document_loader import _clean_pandoc_markdown

    text = (
        "[Article 1 Definitions](#_Toc1) 3\n"
        "[Article 2 Term](#_Toc2) 7\n"
        "\n"
        "Body text."
    )
    assert _clean_pandoc_markdown(text, drop_toc=True) == "Body text."


def test_cleanup_drops_column_holding_only_empty_change_spans():
    """The noise metadata is stripped before columns are judged empty."""
    from app.utils.document_loader import _clean_pandoc_markdown

    table = (
        '| Term | Meaning | []{.deletion author="A" date="D"} |\n'
        "|---|---|---|\n"
        '| Employer | means the company | []{.deletion author="A" date="D"} |\n'
    )
    out = _clean_pandoc_markdown(table)

    assert ".deletion" not in out
    for line in out.strip().split("\n"):
        assert line.count("|") == 3, line


def test_drop_empty_table_columns_keeps_dash_only_data_row():
    """Only the rule under the header is a rule; later dashes are cell content."""
    from app.utils.document_loader import _drop_empty_table_columns

    table = "| A | B |  |\n|---|---|--|\n| x | y |  |\n| --- | ---- |  |\n"
    out = _drop_empty_table_columns(table).strip().split("\n")

    assert out[1] == "| --- | --- |"  # the header rule, compacted
    assert out[3] == "| --- | ---- |"  # data row, verbatim


def test_drop_empty_table_columns_skips_fenced_code_blocks():
    """Pipe-looking lines inside a code fence are content, not a table."""
    from app.utils.document_loader import _drop_empty_table_columns

    text = "```\n| id | name |  |\n| 1  | x    |  |\n```\n"
    assert _drop_empty_table_columns(text) == text


def test_clean_pandoc_markdown_strips_merged_empty_change_spans():
    """Two adjacent empty deletions are merged by pandoc into "[ ]{...}"."""
    from app.utils.document_loader import _clean_pandoc_markdown

    text = 'g) [ ]{.deletion author="Jaroslav Skubal" date="2026-09-14T17:08:00Z"}a f)'
    assert _clean_pandoc_markdown(text) == "g) a f)"


def test_cleanup_failure_falls_back_to_uncleaned_text(tmp_path):
    """A bug in cleanup must not cost the caller an extraction pandoc completed."""
    from app.utils import document_loader
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "doc.docx"
    p.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "| a | b |\n|---|---|\n"

    loader = PandocDocxLoader(str(p), include_headers_footers=False)
    with patch.dict("sys.modules", {"pypandoc": fake_pypandoc}), patch.object(
        document_loader, "_clean_pandoc_markdown", side_effect=RuntimeError("boom")
    ):
        docs = loader.load()

    assert docs[0].page_content == "| a | b |\n|---|---|\n"


def test_extraction_log_line_omits_the_file_path(tmp_path, caplog):
    """Upload paths carry the client's filename, which names matters and parties."""
    import logging

    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "Smith_v_Jones_Settlement.docx"
    p.write_bytes(b"placeholder")

    fake_pypandoc = MagicMock()
    fake_pypandoc.convert_file.return_value = "Body."

    loader = PandocDocxLoader(str(p), include_headers_footers=False)
    with caplog.at_level(logging.INFO), patch.dict(
        "sys.modules", {"pypandoc": fake_pypandoc}
    ):
        loader.load()

    assert any("Chars:" in record.message for record in caplog.records)
    assert not any("Smith_v_Jones" in record.getMessage() for record in caplog.records)


def test_clean_pandoc_markdown_leaves_double_spaces_elsewhere_on_the_line():
    """Removing a noise span must not reflow whitespace that is document content."""
    from app.utils.document_loader import _clean_pandoc_markdown

    text = 'Use `a  b` here []{.deletion author="A" date="D"}and "quoted  text".'
    assert _clean_pandoc_markdown(text) == 'Use `a  b` here and "quoted  text".'


def test_cleanup_drops_column_holding_only_invisible_characters():
    """A cell of nothing but a zero-width space is an empty cell."""
    from app.utils.document_loader import _clean_pandoc_markdown

    table = "| Term | Meaning | \u200b |\n|---|---|---|\n| Employer | the company | \u200b |\n"
    out = _clean_pandoc_markdown(table)

    for line in out.strip().split("\n"):
        assert line.count("|") == 3, line


def test_grid_style_alone_still_strips_heading_anchors():
    """The two knobs are independent: restoring the old output verbatim needs both."""
    from app.utils.document_loader import _pandoc_markdown_format

    assert (
        _pandoc_markdown_format(table_style="grid", strip_heading_anchors=True)
        == "markdown-header_attributes"
    )


# --- end-to-end, with a real pandoc binary ---------------------------------


def _pandoc_available() -> bool:
    try:
        import pypandoc

        pypandoc.get_pandoc_version()
        return True
    except Exception:
        return False


def _make_bilingual_contract(path):
    """A bilingual contract laid out as one long table, as legal offices write them."""
    from docx import Document as Docx

    long_cs = (
        "Zaměstnavatel tímto dává zaměstnanci výpověď z pracovního poměru z důvodu "
        "uvedeného v ustanovení § 52 písm. g) zákoníku práce, tedy pro závažné porušení "
        "povinnosti vyplývající z právních předpisů vztahujících se k vykonávané práci."
    )
    long_en = (
        "The Employer hereby gives the Employee notice of termination of employment on "
        "the grounds set out in Section 52(g) of the Labour Code, i.e. for a serious "
        "breach of an obligation arising from legal regulations relating to the work."
    )

    doc = Docx()
    table = doc.add_table(rows=0, cols=4)
    for i in range(1, 31):
        row = table.add_row()
        if i % 8 == 0:
            row.cells[0].text = f"{i}. {long_cs}"
            row.cells[1].text = f"{i}. {long_en}"
        else:
            row.cells[0].text = f"{i}. Smluvní strany se dohodly."
            row.cells[1].text = f"{i}. The Parties have agreed."
    doc.save(path)


@pytest.mark.skipif(not _pandoc_available(), reason="pandoc binary not available")
def test_compact_tables_shrink_bilingual_contract(tmp_path):
    """Compact extraction is a fraction of the grid form and loses no content."""
    from app.utils.document_loader import PandocDocxLoader

    p = tmp_path / "contract.docx"
    _make_bilingual_contract(str(p))

    grid = (
        PandocDocxLoader(
            str(p),
            include_headers_footers=False,
            table_style="grid",
            cleanup=False,
            strip_heading_anchors=False,
        )
        .load()[0]
        .page_content
    )
    compact = (
        PandocDocxLoader(str(p), include_headers_footers=False).load()[0].page_content
    )

    assert len(compact) < len(grid) / 2
    # No cell is padded out to the width of the longest line in its column.
    assert max(len(line) for line in compact.split("\n")) < 700
    for probe in (
        "Smluvní strany se dohodly",
        "The Parties have agreed",
        "§ 52 písm. g) zákoníku práce",
        "30\\. Smluvní strany",
    ):
        assert probe in compact


@pytest.mark.skipif(not _pandoc_available(), reason="pandoc binary not available")
def test_compact_tables_keep_tracked_changes(tmp_path):
    """Redlines survive compaction, in pipe tables and in the HTML fallback alike."""
    from app.utils.document_loader import PandocDocxLoader
    from docx import Document as Docx

    p = tmp_path / "redline.docx"
    doc = Docx()
    doc.add_paragraph("Clause one.")
    doc.save(str(p))

    _inject_tracked_change(str(p))

    text = (
        PandocDocxLoader(str(p), include_headers_footers=False).load()[0].page_content
    )
    assert "VLOZENO" in text and "SMAZANO" in text
    assert "insertion" in text and "deletion" in text
    assert "Jaroslav Skubal" in text


def _inject_tracked_change(path):
    """Add one w:ins and one w:del run to a .docx (python-docx can't author them)."""
    import shutil
    import zipfile

    source = f"{path}.orig"
    shutil.move(path, source)
    revision = (
        '<w:p><w:ins w:id="777" w:author="Jaroslav Skubal" w:date="2026-09-14T17:08:00Z">'
        "<w:r><w:t>VLOZENO</w:t></w:r></w:ins>"
        '<w:del w:id="778" w:author="Jaroslav Skubal" w:date="2026-09-14T17:08:00Z">'
        "<w:r><w:delText>SMAZANO</w:delText></w:r></w:del></w:p>"
    )
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(path, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                marker = "<w:sectPr" if "<w:sectPr" in xml else "</w:body>"
                data = xml.replace(marker, revision + marker, 1).encode("utf-8")
            zout.writestr(item, data)


# ---------------------------------------------------------------------------
# RTF (.rtf)
# ---------------------------------------------------------------------------


def test_get_loader_rtf(tmp_path):
    from langchain_community.document_loaders import UnstructuredRTFLoader

    file_path = tmp_path / "brief.rtf"
    file_path.write_text(r"{\rtf1 hello}", encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "brief.rtf", "application/rtf", str(file_path)
    )
    assert isinstance(loader, UnstructuredRTFLoader)
    assert known_type is True
    assert file_ext == "rtf"


def test_get_loader_rtf_not_hijacked_by_markdown_mime(tmp_path):
    """A markdown Content-Type must not route .rtf into the markdown loader."""
    from langchain_community.document_loaders import UnstructuredRTFLoader

    file_path = tmp_path / "brief.rtf"
    file_path.write_text(r"{\rtf1 hello}", encoding="utf-8")

    loader, _, _ = get_loader("brief.rtf", "text/markdown", str(file_path))
    assert isinstance(loader, UnstructuredRTFLoader)


# ---------------------------------------------------------------------------
# Email (.eml / .msg)
# ---------------------------------------------------------------------------

EML_SAMPLE = (
    "From: Alice <alice@example.com>\n"
    "To: Bob <bob@example.com>\n"
    "Subject: Quarterly numbers\n"
    "Date: Tue, 1 Apr 2025 10:00:00 -0000\n"
    'Content-Type: text/plain; charset="utf-8"\n'
    "\n"
    "Hi Bob,\n"
    "Here are the Q1 results. Revenue up 12%.\n"
    "Thanks,\n"
    "Alice\n"
)


def test_get_loader_eml(tmp_path):
    from app.utils.document_loader import EmailLoader

    file_path = tmp_path / "mail.eml"
    file_path.write_text(EML_SAMPLE, encoding="utf-8")

    loader, known_type, file_ext = get_loader(
        "mail.eml", "message/rfc822", str(file_path)
    )
    assert isinstance(loader, EmailLoader)
    assert known_type is True
    assert file_ext == "eml"


def test_get_loader_eml_not_hijacked_by_markdown_mime(tmp_path):
    """A markdown Content-Type must not route .eml into the markdown loader."""
    from app.utils.document_loader import EmailLoader

    file_path = tmp_path / "mail.eml"
    file_path.write_text(EML_SAMPLE, encoding="utf-8")

    loader, _, _ = get_loader("mail.eml", "text/markdown", str(file_path))
    assert isinstance(loader, EmailLoader)


def test_email_loader_prepends_headers(tmp_path):
    from app.utils.document_loader import EmailLoader

    file_path = tmp_path / "mail.eml"
    file_path.write_text(EML_SAMPLE, encoding="utf-8")

    docs = EmailLoader(str(file_path), include_headers=True).load()
    assert len(docs) == 1
    text = docs[0].page_content
    assert "From: Alice <alice@example.com>" in text
    assert "To: Bob <bob@example.com>" in text
    assert "Subject: Quarterly numbers" in text
    assert "Q1 results" in text
    # Header block precedes the body
    assert text.index("Subject:") < text.index("Q1 results")


def test_email_loader_body_only_when_headers_disabled(tmp_path):
    from app.utils.document_loader import EmailLoader

    file_path = tmp_path / "mail.eml"
    file_path.write_text(EML_SAMPLE, encoding="utf-8")

    docs = EmailLoader(str(file_path), include_headers=False).load()
    text = docs[0].page_content
    assert "From:" not in text
    assert "Subject:" not in text
    assert "Q1 results" in text


def test_get_loader_msg(tmp_path):
    from app.utils.document_loader import OutlookMsgLoader

    file_path = tmp_path / "mail.msg"
    file_path.write_bytes(b"placeholder")

    loader, known_type, file_ext = get_loader(
        "mail.msg", "application/vnd.ms-outlook", str(file_path)
    )
    assert isinstance(loader, OutlookMsgLoader)
    assert file_ext == "msg"


def test_outlook_msg_loader_builds_headers_from_transport_headers(tmp_path):
    """Headers come from transport headers (To/Cc preserved, Bcc absent by nature)."""
    from app.utils.document_loader import OutlookMsgLoader

    file_path = tmp_path / "mail.msg"
    file_path.write_bytes(b"placeholder")

    fake_msg = MagicMock()
    fake_msg.body = "Hi Bob,\nHere are the Q1 results."
    fake_msg.message_headers = {
        "From": "Alice <alice@example.com>",
        "To": "Bob <bob@example.com>",
        "Cc": "Carol <carol@example.com>",
        "Subject": "Quarterly numbers",
        "Date": "Tue, 1 Apr 2025 10:00:00 -0000",
    }

    fake_oxmsg = MagicMock()
    fake_oxmsg.Message.load.return_value = fake_msg

    loader = OutlookMsgLoader(str(file_path), include_headers=True)
    with patch.dict("sys.modules", {"oxmsg": fake_oxmsg}):
        docs = loader.load()

    assert len(docs) == 1
    text = docs[0].page_content
    assert "From: Alice <alice@example.com>" in text
    assert "To: Bob <bob@example.com>" in text
    assert "Cc: Carol <carol@example.com>" in text  # Cc preserved, not collapsed into To
    assert "Subject: Quarterly numbers" in text
    assert "Q1 results" in text
    fake_oxmsg.Message.load.assert_called_once_with(str(file_path))


def test_outlook_msg_loader_does_not_leak_bcc_recipients(tmp_path):
    """msg.recipients (which can include Bcc) must not be surfaced in the output."""
    from app.utils.document_loader import OutlookMsgLoader

    file_path = tmp_path / "mail.msg"
    file_path.write_bytes(b"placeholder")

    bcc = MagicMock(email_address="secret-bcc@example.com")
    bcc.name = "Secret"
    fake_msg = MagicMock()
    fake_msg.body = "Body."
    fake_msg.message_headers = {"To": "Bob <bob@example.com>"}
    fake_msg.recipients = [bcc]  # present, but must be ignored

    fake_oxmsg = MagicMock()
    fake_oxmsg.Message.load.return_value = fake_msg

    loader = OutlookMsgLoader(str(file_path), include_headers=True)
    with patch.dict("sys.modules", {"oxmsg": fake_oxmsg}):
        docs = loader.load()

    assert "secret-bcc@example.com" not in docs[0].page_content
    assert "To: Bob <bob@example.com>" in docs[0].page_content


def test_outlook_msg_loader_falls_back_to_attributes_without_headers(tmp_path):
    """When transport headers are absent, From/Subject/Date fall back to attributes."""
    from app.utils.document_loader import OutlookMsgLoader

    file_path = tmp_path / "mail.msg"
    file_path.write_bytes(b"placeholder")

    fake_msg = MagicMock()
    fake_msg.body = "Body text."
    fake_msg.message_headers = {}  # no transport headers (e.g. a sent draft)
    fake_msg.sender = "Alice <alice@example.com>"
    fake_msg.subject = "Quarterly numbers"
    fake_msg.sent_date = "Tue, 1 Apr 2025 10:00:00"

    fake_oxmsg = MagicMock()
    fake_oxmsg.Message.load.return_value = fake_msg

    loader = OutlookMsgLoader(str(file_path), include_headers=True)
    with patch.dict("sys.modules", {"oxmsg": fake_oxmsg}):
        docs = loader.load()

    text = docs[0].page_content
    assert "From: Alice <alice@example.com>" in text
    assert "Subject: Quarterly numbers" in text
    assert "Body text." in text


def test_outlook_msg_loader_body_only_when_headers_disabled(tmp_path):
    from app.utils.document_loader import OutlookMsgLoader

    file_path = tmp_path / "mail.msg"
    file_path.write_bytes(b"placeholder")

    fake_msg = MagicMock()
    fake_msg.body = "Just the body."

    fake_oxmsg = MagicMock()
    fake_oxmsg.Message.load.return_value = fake_msg

    loader = OutlookMsgLoader(str(file_path), include_headers=False)
    with patch.dict("sys.modules", {"oxmsg": fake_oxmsg}):
        docs = loader.load()

    assert docs[0].page_content == "Just the body."


# ---------------------------------------------------------------------------
# Standalone image OCR (.png/.jpg/.tiff/...)
# ---------------------------------------------------------------------------


def _make_png(path, color=(255, 0, 0)):
    from PIL import Image

    Image.new("RGB", (8, 8), color).save(path, format="PNG")


def _make_multiframe_tiff(path, frames=2):
    from PIL import Image

    imgs = [Image.new("RGB", (8, 8), (i * 10, 0, 0)) for i in range(frames)]
    imgs[0].save(path, format="TIFF", save_all=True, append_images=imgs[1:])


def test_get_loader_image_png(tmp_path):
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))

    loader, known_type, file_ext = get_loader("scan.png", "image/png", str(p))
    assert isinstance(loader, ImageOCRLoader)
    assert known_type is True
    assert file_ext == "png"


def test_get_loader_image_by_content_type(tmp_path):
    """Routing also works off an image/* Content-Type when the name lacks an ext."""
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan"
    _make_png(str(p))

    loader, _, _ = get_loader("scan", "image/tiff", str(p))
    assert isinstance(loader, ImageOCRLoader)


def test_get_loader_image_not_hijacked_by_markdown_mime(tmp_path):
    """A markdown Content-Type must not route an image into the markdown loader."""
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))

    loader, _, _ = get_loader("scan.png", "text/markdown", str(p))
    assert isinstance(loader, ImageOCRLoader)


def test_image_ocr_loader_single_image(tmp_path):
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))

    fake_module, client = _make_mistral_module(
        pages=[MagicMock(index=0, markdown="scanned text")]
    )

    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ):
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "scanned text"
    assert docs[0].metadata == {"source": str(p), "page": 1}
    # Sent to Mistral OCR as an image_url data URL
    _, kwargs = client.ocr.process.call_args
    assert kwargs["document"]["type"] == "image_url"
    assert kwargs["document"]["image_url"].startswith("data:image/png;base64,")


def test_image_ocr_loader_multipage_tiff_yields_one_doc_per_frame(tmp_path):
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "production.tiff"
    _make_multiframe_tiff(str(p), frames=2)

    fake_module, client = _make_mistral_module(
        pages=[MagicMock(index=0, markdown="page text")]
    )

    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ):
        docs = loader.load()

    # One OCR call and one Document per TIFF frame, with sequential page numbers.
    assert client.ocr.process.call_count == 2
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert all(d.page_content == "page text" for d in docs)


def test_image_ocr_loader_caps_frames(tmp_path):
    """Frames beyond IMAGE_OCR_MAX_PAGES are skipped (no unbounded OCR fan-out)."""
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "huge.tiff"
    _make_multiframe_tiff(str(p), frames=5)

    fake_module, client = _make_mistral_module(
        pages=[MagicMock(index=0, markdown="page text")]
    )

    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(document_loader, "IMAGE_OCR_MAX_PAGES", 2):
        docs = loader.load()

    assert client.ocr.process.call_count == 2
    assert len(docs) == 2


def test_image_ocr_loader_normalizes_cmyk_jpeg(tmp_path):
    """A CMYK JPEG must be normalized to PNG (not passed through), so OCR reads it."""
    from PIL import Image
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.jpg"
    Image.new("CMYK", (8, 8)).save(str(p), format="JPEG")

    fake_module, client = _make_mistral_module(
        pages=[MagicMock(index=0, markdown="text")]
    )
    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ):
        loader.load()

    _, kwargs = client.ocr.process.call_args
    # CMYK can't pass through as image/jpeg; it must be re-encoded to PNG.
    assert kwargs["document"]["image_url"].startswith("data:image/png;base64,")


def test_image_ocr_loader_zero_cap_means_unlimited(tmp_path):
    """IMAGE_OCR_MAX_PAGES<=0 disables the cap rather than returning blank output."""
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))

    fake_module, client = _make_mistral_module(
        pages=[MagicMock(index=0, markdown="scanned text")]
    )
    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", "test-key"
    ), patch.object(document_loader, "IMAGE_OCR_MAX_PAGES", 0):
        docs = loader.load()

    assert client.ocr.process.call_count == 1
    assert docs[0].page_content == "scanned text"


def test_image_ocr_loader_requires_api_key(tmp_path):
    from app.utils import document_loader
    from app.utils.document_loader import ImageOCRLoader

    p = tmp_path / "scan.png"
    _make_png(str(p))

    fake_module, _ = _make_mistral_module(pages=[])
    loader = ImageOCRLoader(str(p))
    with patch.dict(
        "sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}
    ), patch.object(
        document_loader, "MISTRAL_API_KEY", ""
    ):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            loader.load()
