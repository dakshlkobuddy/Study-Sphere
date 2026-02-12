import os
import re
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def _safe_name(file_name: str) -> str:
    base = os.path.basename(file_name)
    return re.sub(r"[^a-zA-Z0-9._ -]", "_", base)


def _file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower().replace(".", "")


def _pdf_page_count(file_path: Path) -> int:
    reader = PdfReader(str(file_path))
    return len(reader.pages)


def _load_docs_by_extension(file_path: Path):
    ext = _file_extension(file_path.name)
    if ext == "pdf":
        return PyPDFLoader(str(file_path)).load()
    if ext == "txt":
        return TextLoader(str(file_path), encoding="utf-8").load()
    if ext == "docx":
        return Docx2txtLoader(str(file_path)).load()
    raise ValueError(f"Unsupported extension for parsing: .{ext}")


def _normalize_page(metadata):
    page = metadata.get("page")
    if isinstance(page, int):
        return page
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return page_number
    return None


def cleanup_old_session_data(parent_dir, max_age_hours=24):
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    removed = {"uploads": 0, "uploads_vector_db": 0}

    targets = [
        (Path(parent_dir) / "uploads", "uploads"),
        (Path(parent_dir) / "uploads_vector_db", "uploads_vector_db"),
    ]
    for root, label in targets:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            age_seconds = now - child.stat().st_mtime
            if age_seconds > max_age_seconds:
                shutil.rmtree(child, ignore_errors=True)
                removed[label] += 1
    return removed


def validate_uploaded_files(
    uploaded_files,
    allowed_extensions=None,
    max_files=10,
    max_file_size_mb=20,
    max_pdf_pages=250,
):
    allowed_extensions = allowed_extensions or {"pdf", "docx", "txt"}
    valid_files = []
    errors = []

    if len(uploaded_files) > max_files:
        errors.append(
            f"Too many files uploaded ({len(uploaded_files)}). Maximum allowed is {max_files}."
        )
        return [], errors

    max_bytes = max_file_size_mb * 1024 * 1024
    for file in uploaded_files:
        ext = _file_extension(file.name)
        if ext not in allowed_extensions:
            errors.append(f"{file.name}: unsupported file type '.{ext}'.")
            continue
        if file.size > max_bytes:
            errors.append(
                f"{file.name}: file size {round(file.size / (1024 * 1024), 2)} MB exceeds "
                f"the {max_file_size_mb} MB limit."
            )
            continue
        valid_files.append(file)

    return valid_files, errors


def build_user_vector_db(
    uploaded_files,
    session_id,
    parent_dir,
    embedding_function,
    max_pdf_pages=250,
    progress_callback=None,
):
    uploads_root = Path(parent_dir) / "uploads"
    uploads_db_root = Path(parent_dir) / "uploads_vector_db"

    session_upload_dir = uploads_root / session_id
    session_db_dir = uploads_db_root / session_id

    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir)
    if session_db_dir.exists():
        shutil.rmtree(session_db_dir)

    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_db_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    total_files = max(len(uploaded_files), 1)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        file_path = session_upload_dir / _safe_name(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if _file_extension(file_path.name) == "pdf":
            try:
                pages = _pdf_page_count(file_path)
            except Exception as err:
                raise ValueError(f"Failed to read PDF page count for {file_path.name}: {err}") from err
            if pages > max_pdf_pages:
                raise ValueError(
                    f"{file_path.name} has {pages} pages. Maximum allowed is {max_pdf_pages} pages."
                )

        saved_paths.append(file_path)
        if progress_callback:
            progress_callback(int((idx / total_files) * 35), f"Saved {idx}/{total_files} files")

    docs = []
    parse_errors = []
    for idx, file_path in enumerate(saved_paths, start=1):
        try:
            loaded_docs = _load_docs_by_extension(file_path)
            for doc in loaded_docs:
                doc.metadata["session_id"] = session_id
                doc.metadata["uploaded_file"] = file_path.name
            docs.extend(loaded_docs)
        except Exception as err:
            parse_errors.append(f"{file_path.name}: {err}")
        if progress_callback:
            progress_callback(35 + int((idx / total_files) * 40), f"Parsed {idx}/{total_files} files")

    if not docs:
        details = "; ".join(parse_errors) if parse_errors else "No readable content found."
        raise ValueError(f"No content could be extracted from uploaded documents. {details}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        filename = metadata.get("uploaded_file") or metadata.get("filename") or "unknown_file"
        metadata["filename"] = filename
        metadata["page"] = _normalize_page(metadata)
        metadata["chunk_id"] = f"{session_id}_{idx:05d}"
        chunk.metadata = metadata
    if progress_callback:
        progress_callback(85, f"Chunked into {len(chunks)} segments")

    collection_name = f"user_{session_id}"
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(session_db_dir),
        collection_name=collection_name,
    )
    if progress_callback:
        progress_callback(100, "Embedding and indexing complete")

    return {
        "upload_dir": str(session_upload_dir),
        "db_dir": str(session_db_dir),
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "parse_errors": parse_errors,
    }
