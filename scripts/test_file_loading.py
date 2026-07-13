#!/usr/bin/env python3
"""Practical local tester for the RAG API's document-loading capabilities.

It generates a small sample file for (almost) every supported format, uploads
each to the running API and reports whether the loader extracted the expected
content. Each sample embeds unique marker tokens; a check passes when every
expected marker shows up in the API's response.

Coverage — the recently added features are exercised explicitly:
  * .docx via pandoc on /text  → tracked-changes insertions AND deletions,
                                 comments, and headers/footers are all checked
  * .xlsx  (UnstructuredExcelLoader)
  * .eml   email body + prepended From/Subject headers
  * .rtf   (UnstructuredRTFLoader)
  * .rst / .xml / .md / .html / .csv / .json / .yaml / .py  (text/unstructured)
  * legacy .doc  → must be *rejected* with a clear error (negative test)
  * .pdf and images via Mistral OCR       (opt-in: --ocr, needs MISTRAL_API_KEY)

By default it hits POST /text, which parses files WITHOUT creating embeddings —
so no real OpenAI/Mistral key is needed. The full loader dispatch in
app/utils/document_loader.py is still exercised end to end.

Requires these server-side env vars (already set in the repo's .env) for the
docx feature checks to pass:
    DOCX_TEXT_USE_PANDOC=True
    DOCX_TEXT_TRACK_CHANGES=all
    DOCX_TEXT_INCLUDE_HEADERS_FOOTERS=True
    EMAIL_INCLUDE_HEADERS=True

Usage:
    # 1. start the lite stack:
    #    docker compose -f docker-compose.yaml -f docker-compose.lite.yaml up -d --build
    # 2. run the tests:
    python scripts/test_file_loading.py
    python scripts/test_file_loading.py --embed          # also POST /embed + /query (needs real key)
    python scripts/test_file_loading.py --ocr            # also PDF via /text (needs MISTRAL_API_KEY)
    python scripts/test_file_loading.py --msg path.msg   # also test a real Outlook .msg file
    python scripts/test_file_loading.py --keep           # keep generated sample files

Uses only the Python standard library — no pip install required.
"""

import argparse
import io
import json
import mimetypes
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile

DOCX_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_CT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# OOXML namespaces
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# --------------------------------------------------------------------------- #
# Simple text/markup sample generators (each returns raw bytes).
# --------------------------------------------------------------------------- #
def make_txt():
    return b"Plain text sample ZQTXT. Second line.\n"


def make_md():
    return b"# Heading ZQMD\n\nSome **bold** markdown body.\n"


def make_csv():
    return b"col_a,col_b,col_c\n1,ZQCSV,three\n4,five,six\n"


def make_json():
    return json.dumps({"id": 1, "note": "ZQJSON", "items": [1, 2, 3]}).encode()


def make_xml():
    return b'<?xml version="1.0"?>\n<root><note>ZQXML</note><item>v</item></root>\n'


def make_html():
    return b"<html><body><p>Hello ZQHTML paragraph.</p></body></html>\n"


def make_rst():
    return b"Title ZQRST\n===========\n\nParagraph body here.\n"


def make_rtf():
    return (
        br"{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}"
        br"\f0\fs24 Rich text sample ZQRTF. End.\par }"
    )


def make_py():
    return b'# source-code sample\nTOKEN = "ZQPY"\nprint(TOKEN)\n'


def make_yaml():
    return b"key: value\nnote: ZQYAML\nlist:\n  - a\n  - b\n"


def make_eml():
    return (
        b"From: Alice Sender <alice@example.com>\r\n"
        b"To: bob@example.com\r\n"
        b"Subject: Meeting notes ZQEMLSUBJ\r\n"
        b"Date: Mon, 13 Jul 2026 10:00:00 +0000\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Email body containing the ZQEMLBODY token.\r\n"
    )


def make_doc():
    """Bytes for a fake legacy .doc — content is irrelevant; the API must reject
    it purely on the .doc extension / application/msword type before loading."""
    return b"\xd0\xcf\x11\xe0 legacy doc placeholder ZQDOC"


# --------------------------------------------------------------------------- #
# .docx builder — a full OOXML package exercising every docx feature:
# tracked-change insertion + deletion, a comment, and header/footer text.
# --------------------------------------------------------------------------- #
def make_docx():
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS_W}" xmlns:r="{NS_R}">
  <w:body>
    <w:p><w:r><w:t>Body text ZQBODY.</w:t></w:r></w:p>
    <w:p>
      <w:ins w:id="1" w:author="Alice" w:date="2026-01-01T00:00:00Z">
        <w:r><w:t>Inserted ZQINS text.</w:t></w:r>
      </w:ins>
    </w:p>
    <w:p>
      <w:del w:id="2" w:author="Bob" w:date="2026-01-01T00:00:00Z">
        <w:r><w:delText>Deleted ZQDEL text.</w:delText></w:r>
      </w:del>
    </w:p>
    <w:p>
      <w:commentRangeStart w:id="0"/>
      <w:r><w:t>Some reviewed sentence.</w:t></w:r>
      <w:commentRangeEnd w:id="0"/>
      <w:r><w:commentReference w:id="0"/></w:r>
    </w:p>
    <w:sectPr>
      <w:headerReference w:type="default" r:id="rId2"/>
      <w:footerReference w:type="default" r:id="rId3"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    comments_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="{NS_W}">
  <w:comment w:id="0" w:author="Carol" w:date="2026-01-01T00:00:00Z" w:initials="C">
    <w:p><w:r><w:t>Reviewer comment ZQCMT.</w:t></w:r></w:p>
  </w:comment>
</w:comments>"""

    header_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:hdr xmlns:w="{NS_W}"><w:p><w:r><w:t>Header ZQHDR CONFIDENTIAL</w:t>'
        f"</w:r></w:p></w:hdr>"
    )
    footer_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:ftr xmlns:w="{NS_W}"><w:p><w:r><w:t>Footer ZQFTR Matter-42</w:t>'
        f"</w:r></w:p></w:ftr>"
    )
    styles_xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{NS_W}"><w:docDefaults/></w:styles>'
    )

    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{NS_CT}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="{DOCX_CT}.main+xml"/>
  <Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    root_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_REL}">
  <Relationship Id="rId1" Type="{NS_R}/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_REL}">
  <Relationship Id="rId1" Type="{NS_R}/comments" Target="comments.xml"/>
  <Relationship Id="rId2" Type="{NS_R}/header" Target="header1.xml"/>
  <Relationship Id="rId3" Type="{NS_R}/footer" Target="footer1.xml"/>
  <Relationship Id="rId4" Type="{NS_R}/styles" Target="styles.xml"/>
</Relationships>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/comments.xml", comments_xml)
        z.writestr("word/header1.xml", header_xml)
        z.writestr("word/footer1.xml", footer_xml)
        z.writestr("word/styles.xml", styles_xml)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# .xlsx builder — minimal spreadsheet with inline strings (no sharedStrings).
# --------------------------------------------------------------------------- #
def make_xlsx():
    sheet = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS_S}">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>Header</t></is></c>
      <c r="B1" t="inlineStr"><is><t>ZQXLSX</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>alpha</t></is></c>
      <c r="B2" t="inlineStr"><is><t>bravo</t></is></c>
    </row>
  </sheetData>
</worksheet>"""

    workbook = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{NS_S}" xmlns:r="{NS_R}">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

    wb_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_REL}">
  <Relationship Id="rId1" Type="{NS_R}/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""

    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{NS_CT}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="{XLSX_CT}.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

    root_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_REL}">
  <Relationship Id="rId1" Type="{NS_R}/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Test registry. Each entry: filename, generator, mime, expected-markers,
# optional feature note, and `negative` (expect the upload to be rejected).
# --------------------------------------------------------------------------- #
class Case:
    def __init__(self, name, gen, mime, expect, note="", negative=False):
        self.name = name
        self.gen = gen
        self.mime = mime
        self.expect = expect          # list of substrings expected in response
        self.note = note
        self.negative = negative      # True => expect an error response


CASES = [
    Case("sample.txt", make_txt, "text/plain", ["ZQTXT"]),
    Case("sample.md", make_md, "text/markdown", ["ZQMD"]),
    Case("sample.csv", make_csv, "text/csv", ["ZQCSV"]),
    Case("sample.json", make_json, "application/json", ["ZQJSON"]),
    Case("sample.xml", make_xml, "application/xml", ["ZQXML"]),
    Case("sample.html", make_html, "text/html", ["ZQHTML"]),
    Case("sample.rst", make_rst, "text/x-rst", ["ZQRST"]),
    Case("sample.rtf", make_rtf, "application/rtf", ["ZQRTF"]),
    Case("sample.py", make_py, "text/x-python", ["ZQPY"]),
    Case("sample.yaml", make_yaml, "text/yaml", ["ZQYAML"]),
    Case("sample.eml", make_eml, "message/rfc822",
         ["ZQEMLBODY", "ZQEMLSUBJ", "alice@example.com"],
         note="body + prepended From/Subject headers"),
    Case("sample.xlsx", make_xlsx, XLSX_CT, ["ZQXLSX"],
         note="UnstructuredExcelLoader"),
    Case("sample.docx", make_docx, DOCX_CT,
         ["ZQBODY", "ZQINS", "ZQDEL", "ZQCMT", "ZQHDR", "ZQFTR"],
         note="pandoc: tracked insertion+deletion, comment, header/footer"),
    Case("sample.doc", make_doc, "application/msword", ["not supported"],
         note="legacy .doc must be rejected", negative=True),
]


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only).
# --------------------------------------------------------------------------- #
def post_multipart(url, fields, file_field, filename, file_bytes, content_type,
                   timeout=180.0):
    boundary = f"----ragtest{uuid.uuid4().hex}"
    body = io.BytesIO()

    def w(s):
        body.write(s.encode("utf-8"))

    for name, value in fields.items():
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        w(f"{value}\r\n")
    w(f"--{boundary}\r\n")
    w(f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n')
    w(f"Content-Type: {content_type}\r\n\r\n")
    body.write(file_bytes)
    w(f"\r\n--{boundary}--\r\n")

    req = urllib.request.Request(
        url, data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def post_json(url, payload, timeout=60.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def wait_for_health(base_url, attempts=45, delay=2.0):
    url = f"{base_url}/health"
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("status") == "UP":
                    return True
        except Exception:
            pass
        print(f"  waiting for API to be healthy... ({i + 1}/{attempts})")
        time.sleep(delay)
    return False


GREEN, RED, YELLOW, DIM, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"
)


def _c(color, text):
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


# --------------------------------------------------------------------------- #
def run_text_tests(base_url, out_dir):
    print(f"\n=== POST /text  (parse only, no embeddings) — {len(CASES)} cases ===")
    url = f"{base_url}/text"
    failures = 0
    for case in CASES:
        data = case.gen()
        with open(os.path.join(out_dir, case.name), "wb") as f:
            f.write(data)

        status, resp = post_multipart(
            url, fields={"file_id": str(uuid.uuid4())}, file_field="file",
            filename=case.name, file_bytes=data, content_type=case.mime,
        )

        if case.negative:
            body = resp if isinstance(resp, str) else json.dumps(resp)
            ok = status is not None and status >= 400 and case.expect[0].lower() in body.lower()
            label = _c(GREEN, "PASS") if ok else _c(RED, "FAIL")
            detail = f"correctly rejected (HTTP {status})" if ok else \
                     f"expected rejection containing {case.expect[0]!r}; got HTTP {status}: {str(resp)[:100]}"
            print(f"  {label}  {case.name:<13} {detail}")
            if not ok:
                failures += 1
            continue

        text = resp.get("text", "") if isinstance(resp, dict) else ""
        if status != 200 or not isinstance(resp, dict):
            failures += 1
            print(f"  {_c(RED, 'FAIL')}  {case.name:<13} HTTP {status}: {str(resp)[:140]}")
            continue

        missing = [m for m in case.expect if m.lower() not in text.lower()]
        if not missing:
            note = f"  {_c(DIM, '(' + case.note + ')')}" if case.note else ""
            print(f"  {_c(GREEN, 'PASS')}  {case.name:<13} all {len(case.expect)} marker(s) found{note}")
        else:
            failures += 1
            found = [m for m in case.expect if m not in missing]
            snippet = repr(" ".join(text.split())[:120])
            print(f"  {_c(RED, 'FAIL')}  {case.name:<13} missing {missing}; found {found}")
            print(f"        {_c(DIM, 'got: ' + snippet)}")
    return failures


def run_embed_tests(base_url, out_dir):
    print("\n=== POST /embed + /query  (needs a real RAG_OPENAI_API_KEY) ===")
    embed_url, query_url = f"{base_url}/embed", f"{base_url}/query"
    failures = 0
    for case in CASES:
        if case.negative:
            continue
        data = case.gen()
        file_id = str(uuid.uuid4())
        status, resp = post_multipart(
            embed_url, fields={"file_id": file_id}, file_field="file",
            filename=case.name, file_bytes=data, content_type=case.mime,
        )
        if status != 200 or not isinstance(resp, dict) or not resp.get("status"):
            failures += 1
            print(f"  {_c(RED, 'FAIL')}  {case.name:<13} embed -> HTTP {status}: {str(resp)[:100]}")
            continue
        qstatus, qresp = post_json(query_url, {"query": case.expect[0], "file_id": file_id, "k": 1})
        hit = qstatus == 200 and isinstance(qresp, list) and len(qresp) > 0
        mark = _c(GREEN, "PASS") if hit else _c(YELLOW, "WARN")
        note = f"{len(qresp)} chunk(s) via /query" if hit else f"/query -> {str(qresp)[:70]}"
        print(f"  {mark}  {case.name:<13} embedded ok; {note}")
    return failures


def run_ocr_test(base_url, out_dir):
    print("\n=== POST /text with a PDF  (Mistral OCR — needs MISTRAL_API_KEY) ===")
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 120]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 44>>stream\n"
        b"BT /F1 24 Tf 40 60 Td (OCR sample ZQPDF) Tj ET\n"
        b"endstream endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF"
    )
    with open(os.path.join(out_dir, "sample.pdf"), "wb") as f:
        f.write(pdf)
    status, resp = post_multipart(
        f"{base_url}/text", fields={"file_id": str(uuid.uuid4())}, file_field="file",
        filename="sample.pdf", file_bytes=pdf, content_type="application/pdf",
        timeout=240.0,
    )
    if status == 200 and isinstance(resp, dict):
        text = resp.get("text") or ""
        ok = "ZQPDF" in text.upper()
        mark = _c(GREEN, "PASS") if ok else _c(YELLOW, "WARN")
        print(f"  {mark}  sample.pdf    OCR returned {len(text)} chars: {text[:80]!r}")
        return 0 if ok else 1
    print(f"  {_c(RED, 'FAIL')}  sample.pdf    HTTP {status}: {str(resp)[:150]}")
    return 1


def run_msg_test(base_url, msg_path):
    print("\n=== POST /text with a real .msg (Outlook) ===")
    if not os.path.isfile(msg_path):
        print(f"  {_c(RED, 'FAIL')}  {msg_path} not found")
        return 1
    with open(msg_path, "rb") as f:
        data = f.read()
    status, resp = post_multipart(
        f"{base_url}/text", fields={"file_id": str(uuid.uuid4())}, file_field="file",
        filename=os.path.basename(msg_path), file_bytes=data,
        content_type="application/vnd.ms-outlook",
    )
    name = os.path.basename(msg_path)
    if status != 200 or not isinstance(resp, dict):
        print(f"  {_c(RED, 'FAIL')}  {name:<13} HTTP {status}: {str(resp)[:150]}")
        return 1

    text = resp.get("text") or ""
    # Headers must be decoded: no RFC 2047 encoded-words should survive.
    if "=?" in text and "?=" in text:
        subj = next((ln for ln in text.splitlines() if ln.startswith("Subject:")), "")
        print(f"  {_c(RED, 'FAIL')}  {name:<13} headers still MIME-encoded -> {subj[:90]!r}")
        return 1

    subj = next((ln for ln in text.splitlines() if ln.startswith("Subject:")), "(no subject)")
    print(f"  {_c(GREEN, 'PASS')}  {name:<13} extracted {len(text)} chars; "
          f"headers decoded -> {subj[:80]!r}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--embed", action="store_true",
                        help="also test POST /embed + /query (needs real embeddings key)")
    parser.add_argument("--ocr", action="store_true",
                        help="also test a PDF via /text (needs real MISTRAL_API_KEY)")
    parser.add_argument("--msg", metavar="PATH",
                        help="also test a real Outlook .msg file at PATH")
    parser.add_argument("--keep", action="store_true",
                        help="keep generated sample files")
    parser.add_argument("--no-wait", action="store_true",
                        help="skip the /health readiness wait")
    args = parser.parse_args()

    # Extracted text can contain non-ASCII (e.g. decoded Czech .msg subjects);
    # avoid crashing on a legacy Windows console codepage when printing it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    mimetypes.init()
    print(f"Target API: {args.base_url}")
    if not args.no_wait:
        if not wait_for_health(args.base_url):
            print(_c(RED, "API did not become healthy — is the stack up?"))
            return 2
        print(_c(GREEN, "API is healthy."))

    out_dir = tempfile.mkdtemp(prefix="rag_loader_samples_")
    print(f"Sample files: {out_dir}")

    failures = 0
    try:
        failures += run_text_tests(args.base_url, out_dir)
        if args.embed:
            failures += run_embed_tests(args.base_url, out_dir)
        if args.ocr:
            failures += run_ocr_test(args.base_url, out_dir)
        if args.msg:
            failures += run_msg_test(args.base_url, args.msg)
    finally:
        if args.keep:
            print(f"\nKept sample files in {out_dir}")
        else:
            for name in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, name))
            os.rmdir(out_dir)

    print("\n" + "=" * 60)
    if failures == 0:
        print(_c(GREEN, "ALL TESTS PASSED"))
    else:
        print(_c(RED, f"{failures} test(s) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
