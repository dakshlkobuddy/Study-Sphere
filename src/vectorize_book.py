import os
import shutil
import torch
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)


working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv(os.path.join(working_dir, ".env"), override=True)


def resolve_device() -> str:
    device = os.getenv("DEVICE", "cpu").strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return device


DEVICE = resolve_device()

data_dir = f"{parent_dir}/data/class_12"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"
unified_vector_db_path = f"{vector_db_dir}/class_12_unified_vector_db"

try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE},
    )
except NotImplementedError as err:
    if "meta tensor" not in str(err).lower():
        raise
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )

legacy_text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
unified_text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


def vectorize_book_and_store_to_db(subject, vector_db_name):
    subject_dir = f"{data_dir}/{subject}"
    vector_db_path = f"{vector_db_dir}/{vector_db_name}"

    loader = DirectoryLoader(path=subject_dir, glob="*.pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    chunks = legacy_text_splitter.split_documents(documents)

    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=vector_db_path)
    print(f"[ok] Full book stored: {vector_db_path}")


def vectorize_chapters(subject):
    subject = subject.lower()
    subject_dir = f"{data_dir}/{subject}"
    subject_output_dir = f"{chapters_vector_db_dir}/{subject}"
    os.makedirs(subject_output_dir, exist_ok=True)

    for file_name in os.listdir(subject_dir):
        if not file_name.endswith(".pdf"):
            continue

        chapter_name = file_name[:-4]
        chapter_path = f"{subject_dir}/{file_name}"

        loader = UnstructuredFileLoader(chapter_path)
        documents = loader.load()
        chunks = legacy_text_splitter.split_documents(documents)

        chapter_output_path = f"{subject_output_dir}/{chapter_name}"
        os.makedirs(chapter_output_path, exist_ok=True)

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=chapter_output_path,
        )
        print(f"[ok] {chapter_name} -> saved inside {subject}/")

    print(f"[ok] Completed subject: {subject}")


def vectorize_unified_db(subjects, recreate=False):
    os.makedirs(vector_db_dir, exist_ok=True)

    if recreate and os.path.isdir(unified_vector_db_path):
        shutil.rmtree(unified_vector_db_path)

    all_chunks = []
    for subject in subjects:
        subject_lower = subject.lower()
        subject_dir = f"{data_dir}/{subject_lower}"

        if not os.path.isdir(subject_dir):
            print(f"[skip] Missing subject directory: {subject_dir}")
            continue

        for file_name in sorted(os.listdir(subject_dir)):
            if not file_name.endswith(".pdf"):
                continue

            chapter_name = file_name[:-4]
            pdf_path = f"{subject_dir}/{file_name}"
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for page_doc in pages:
                page_doc.metadata["subject"] = subject_lower
                page_doc.metadata["chapter"] = chapter_name
                # Keep original metadata["page"] from PyPDFLoader for citation.

            chunks = unified_text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"[ok] Prepared chunks: {subject_lower} | {chapter_name}")

    if not all_chunks:
        raise ValueError("No chunks generated for unified DB.")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        persist_directory=unified_vector_db_path,
    )
    print(f"[ok] Unified vector DB stored: {unified_vector_db_path}")
