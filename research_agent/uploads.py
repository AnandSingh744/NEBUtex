"""Safely accept an uploaded PDF: validate it, store it, ingest it.

Shared by every front door (API, web UI) so the safety checks exist in exactly one place.
Never trust an upload: check its name, size, and actual contents before using it.
"""

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO

from research_agent import config
from research_agent.ingest import IngestResult, ingest_pdf
from research_agent.pdf_loader import PDFLoadError, load_pdf
from research_agent.vector_store import VectorStore

PDF_MAGIC = b"%PDF-"  # every real PDF file starts with these bytes
UPLOAD_TMP_DIR = config.DATA_DIR / "uploads_tmp"  # same disk as PAPERS_DIR, so moving a file is instant and atomic

# Only one ingestion at a time: two at once would both rewrite manifest.json and one would lose its entry.
_ingest_lock = threading.Lock()


class UploadError(Exception):
    """A rejected upload. `reason` says why, so each front door can report it in its own way."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason  # invalid_name | not_pdf | too_large | empty | unreadable


def safe_filename(filename: str | None) -> str:
    """'../../My Paper (v2).PDF' -> 'My_Paper_v2.pdf': no folder parts, only safe characters."""
    if not (filename or "").lower().endswith(".pdf"):
        raise UploadError("invalid_name", "Only .pdf files are accepted.")
    stem = Path(Path(filename).name).stem  # .name drops any folders, like "../../"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not stem:
        raise UploadError("invalid_name", "Invalid file name.")
    return f"{stem}.pdf"


def add_uploaded_pdf(filename: str | None, data: BinaryIO, store: VectorStore) -> IngestResult:
    """Validate an uploaded PDF, save it to PAPERS_DIR, and ingest it.

    A file with the same name as an existing document replaces it (only once it's known to be valid).
    """
    name = safe_filename(filename)
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024

    # 1. Save the upload to a temporary file, checking size and file type as we go.
    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_TMP_DIR, suffix=".pdf", delete=False)
    try:
        with tmp:
            size = 0
            while block := data.read(1 << 20):  # 1 MB at a time
                if size == 0 and not block.startswith(PDF_MAGIC):
                    raise UploadError("not_pdf", "This file is not a PDF (its contents don't match).")
                size += len(block)
                if size > max_bytes:
                    raise UploadError("too_large", f"File is larger than {config.MAX_UPLOAD_MB} MB.")
                tmp.write(block)
        if size == 0:
            raise UploadError("empty", "The uploaded file is empty.")

        # 2. Make sure it's a readable PDF BEFORE it can replace an existing document with the same name.
        try:
            load_pdf(tmp.name)
        except PDFLoadError as e:
            raise UploadError("unreadable", str(e).replace(Path(tmp.name).name, name)) from e

        # 3. Move it into place and ingest it (one at a time).
        with _ingest_lock:
            final_path = config.PAPERS_DIR / name
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp.name, final_path)
            return ingest_pdf(final_path, store=store)
    finally:
        if os.path.exists(tmp.name):  # still there only if something failed
            os.unlink(tmp.name)
