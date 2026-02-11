import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

# Folder structure
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# Load env from both project root and src directory for predictable local runs.
load_dotenv(os.path.join(parent_dir, ".env"))
load_dotenv(os.path.join(working_dir, ".env"), override=True)


def resolve_device() -> str:
    device = os.getenv("DEVICE", "cpu").strip().lower()

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"

    return device


DEVICE = resolve_device()

data_dir = f"{parent_dir}/data/class_12"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"

# Embedding model
try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE}
    )
except NotImplementedError as err:
    if "meta tensor" not in str(err).lower():
        raise

    # Fallback for meta-device initialization bugs in upstream torch/transformers combos.
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=150)


# ------------------------------------------------------
# 1. Vectorize full book (optional)
# ------------------------------------------------------
def vectorize_book_and_store_to_db(subject, vector_db_name):
    subject_dir = f"{data_dir}/{subject}"
    vector_db_path = f"{vector_db_dir}/{vector_db_name}"

    loader = DirectoryLoader(
        path=subject_dir,
        glob="*.pdf",
        loader_cls=UnstructuredFileLoader
    )
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=vector_db_path
    )

    print(f"[✔] Full book stored: {vector_db_path}")


# ------------------------------------------------------
# 2. Vectorize chapters into subject folders
# ------------------------------------------------------
def vectorize_chapters(subject):
    subject = subject.lower()

    subject_dir = f"{data_dir}/{subject}"
    subject_output_dir = f"{chapters_vector_db_dir}/{subject}"
    os.makedirs(subject_output_dir, exist_ok=True)

    for file in os.listdir(subject_dir):
        if not file.endswith(".pdf"):
            continue

        chapter_name = file[:-4]     # remove .pdf
        chapter_path = f"{subject_dir}/{file}"

        loader = UnstructuredFileLoader(chapter_path)
        documents = loader.load()

        chunks = text_splitter.split_documents(documents)

        chapter_output_path = f"{subject_output_dir}/{chapter_name}"
        os.makedirs(chapter_output_path, exist_ok=True)

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=chapter_output_path
        )

        print(f"[✔] {chapter_name} → saved inside {subject}/")

    print(f"\n[✓] Completed subject: {subject}\n")
