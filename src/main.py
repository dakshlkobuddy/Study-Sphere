import os
import hashlib
import uuid
from datetime import datetime, timezone
from io import BytesIO

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app_config import load_project_env, resolve_device, get_unified_vector_db_path
from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link
from observability import get_logger, new_error_id
from qa_engine import (
    answer_from_sources,
    generate_quiz_from_sources,
    FALLBACK_ANSWER,
    UPLOADS_FALLBACK_ANSWER,
)
from upload_utility import build_user_vector_db
from upload_utility import cleanup_old_session_data, validate_uploaded_files


working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = load_project_env(working_dir)
DEVICE = resolve_device()
UNIFIED_VECTOR_DB_PATH = get_unified_vector_db_path(parent_dir)
subjects_list = ["Physics", "Chemistry", "Biology"]
logger = get_logger()


@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": DEVICE},
        )
    except NotImplementedError as err:
        if "meta tensor" not in str(err).lower():
            raise
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )


@st.cache_resource
def get_vectorstore():
    if not os.path.isdir(UNIFIED_VECTOR_DB_PATH):
        raise FileNotFoundError(
            "Unified vector DB not found. Run `python src/vectorize_script.py --unified` first."
        )
    return Chroma(
        persist_directory=UNIFIED_VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )


@st.cache_resource
def get_user_vectorstore(user_db_path, session_id):
    if not user_db_path or not os.path.isdir(user_db_path):
        raise FileNotFoundError("No uploaded document DB found for current session.")
    return Chroma(
        persist_directory=user_db_path,
        embedding_function=get_embeddings(),
        collection_name=f"user_{session_id}",
    )


@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


@st.cache_resource
def get_reranker():
    try:
        return CrossEncoder("BAAI/bge-reranker-base")
    except Exception:
        return None


def _wrap_text(text, width=98):
    words = (text or "").split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current_len + extra <= width:
            current.append(word)
            current_len += extra
        else:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _chat_to_pdf_bytes(chat_history, subject, chapter, mode, language):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, page_height = A4
    y = page_height - 50

    header = (
        f"Study Sphere Chat Export | Subject: {subject or 'N/A'} | Chapter: {chapter or 'N/A'} | "
        f"Mode: {mode} | Language: {language} | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    for line in _wrap_text(header, width=110):
        pdf.drawString(40, y, line)
        y -= 14
    y -= 8

    for msg in chat_history:
        role = msg["role"].upper()
        lines = [f"{role}:"] + _wrap_text(msg["content"], width=110)
        for line in lines:
            if y < 50:
                pdf.showPage()
                y = page_height - 50
            pdf.drawString(40, y, line)
            y -= 14
        y -= 8

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _display_source_name(filename):
    name = (filename or "N/A").strip()
    if name.lower().endswith(".pdf.pdf"):
        return name[:-4]
    return name


st.set_page_config(page_title="Study Sphere", page_icon="S", layout="centered")
st.markdown(
    "<h1 style='margin:0 0 0.4rem 0; line-height:1.2;'>Study Sphere Learning Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 1rem;}
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"]) {margin-bottom: 0.25rem;}
div[data-testid="stChatMessage"] {padding-top: 0.35rem; padding-bottom: 0.35rem;}
</style>
""",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []
if "citation_history" not in st.session_state:
    st.session_state.citation_history = []
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None
if "selected_chapter" not in st.session_state:
    st.session_state.selected_chapter = None
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "uploads_signature_processed" not in st.session_state:
    st.session_state.uploads_signature_processed = ""
if "user_upload_db_path" not in st.session_state:
    st.session_state.user_upload_db_path = None
if "user_upload_collection_name" not in st.session_state:
    st.session_state.user_upload_collection_name = None
if "upload_cleanup_done" not in st.session_state:
    st.session_state.upload_cleanup_done = False


def _assistant_count(messages):
    return sum(1 for msg in messages if msg.get("role") == "assistant")


def _normalize_artifact_history():
    """
    Keep artifacts aligned to assistant messages only.
    Supports migration from old state where artifacts were indexed by all chat messages.
    """
    assistant_total = _assistant_count(st.session_state.chat_history)

    if len(st.session_state.video_history) == len(st.session_state.chat_history):
        st.session_state.video_history = [
            st.session_state.video_history[idx]
            for idx, msg in enumerate(st.session_state.chat_history)
            if msg.get("role") == "assistant"
        ]
    if len(st.session_state.citation_history) == len(st.session_state.chat_history):
        st.session_state.citation_history = [
            st.session_state.citation_history[idx]
            for idx, msg in enumerate(st.session_state.chat_history)
            if msg.get("role") == "assistant"
        ]

    if len(st.session_state.video_history) > assistant_total:
        st.session_state.video_history = st.session_state.video_history[:assistant_total]
    if len(st.session_state.citation_history) > assistant_total:
        st.session_state.citation_history = st.session_state.citation_history[:assistant_total]

    while len(st.session_state.video_history) < assistant_total:
        st.session_state.video_history.append([])
    while len(st.session_state.citation_history) < assistant_total:
        st.session_state.citation_history.append([])


def _render_assistant_tabs(answer_text, citations, videos):
    tabs = st.tabs(["Answer", "Sources", "Videos"])
    with tabs[0]:
        st.markdown(answer_text)
    with tabs[1]:
        with st.expander("Sources", expanded=False):
            if citations:
                for citation in citations:
                    source_name = _display_source_name(citation.get("filename"))
                    page = citation.get("page", "N/A")
                    score = citation.get("score", 0.0)
                    st.markdown(
                        f"Source: {source_name}  \n"
                        f"Page: {page}  \n"
                        f"Relevance Score: {score:.2f}"
                    )
            else:
                st.caption("No source references.")
    with tabs[2]:
        with st.expander("Videos", expanded=False):
            if videos:
                for title, link in videos:
                    st.info(f"{title}\n\nLink: {link}")
            else:
                st.caption("No video references.")

MAX_UPLOAD_FILES = 8
MAX_UPLOAD_FILE_MB = 15
MAX_UPLOAD_PDF_PAGES = 200
SESSION_CLEANUP_AGE_HOURS = 24

if not st.session_state.upload_cleanup_done:
    removed = cleanup_old_session_data(parent_dir, max_age_hours=SESSION_CLEANUP_AGE_HOURS)
    st.session_state.upload_cleanup_done = True
    removed_total = removed["uploads"] + removed["uploads_vector_db"]
    if removed_total > 0:
        st.caption(
            f"Cleanup completed: removed {removed['uploads']} old upload folder(s) and "
            f"{removed['uploads_vector_db']} old vector DB folder(s)."
        )

_normalize_artifact_history()

with st.sidebar:
    st.subheader("Answer Settings")
    source_mode = st.radio(
        "Source Mode",
        ["NCERT", "My Documents"],
        index=0,
        horizontal=True,
    )
    query_mode = st.selectbox(
        "Query Mode",
        ["Explain", "Exam Answer", "Short Notes", "MCQ"],
        index=0,
    )
    output_language = st.radio(
        "Output Language",
        ["English", "Hindi"],
        index=0,
        horizontal=True,
    )
    st.subheader("Chapter Quiz")
    quiz_count = st.slider("Questions", min_value=3, max_value=10, value=5, step=1)
    generate_quiz_clicked = st.button("Generate Quiz From Chapter")
uploaded_files = []

if source_mode == "My Documents":
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help=(
            f"Max {MAX_UPLOAD_FILES} files, {MAX_UPLOAD_FILE_MB} MB each, "
            f"PDF limit {MAX_UPLOAD_PDF_PAGES} pages."
        ),
    )

if source_mode == "My Documents" and uploaded_files:
    valid_files, upload_validation_errors = validate_uploaded_files(
        uploaded_files=uploaded_files,
        allowed_extensions={"pdf", "docx", "txt"},
        max_files=MAX_UPLOAD_FILES,
        max_file_size_mb=MAX_UPLOAD_FILE_MB,
        max_pdf_pages=MAX_UPLOAD_PDF_PAGES,
    )

    if upload_validation_errors:
        st.error("Upload validation failed:")
        for err in upload_validation_errors:
            st.write(f"- {err}")

    if not valid_files:
        st.session_state.user_upload_db_path = None
        st.session_state.user_upload_collection_name = None
        st.session_state.uploads_signature_processed = ""
        get_user_vectorstore.clear()
        uploaded_files = []
    else:
        uploaded_files = valid_files

if source_mode == "My Documents" and uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) uploaded.")
    with st.expander("View uploaded files"):
        for file in uploaded_files:
            size_kb = round(file.size / 1024, 2)
            st.write(f"- {file.name} ({size_kb} KB)")

    signature_source = "|".join(
        f"{file.name}:{file.size}" for file in sorted(uploaded_files, key=lambda f: f.name)
    )
    current_signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()

    if st.session_state.uploads_signature_processed != current_signature:
        try:
            progress = st.progress(0, text="Starting ingestion...")
            with st.spinner("Processing uploaded files (parse, chunk, embed)..."):
                def on_progress(percent, message):
                    progress.progress(max(0, min(100, int(percent))), text=message)

                result = build_user_vector_db(
                    uploaded_files=uploaded_files,
                    session_id=st.session_state.session_id,
                    parent_dir=parent_dir,
                    embedding_function=get_embeddings(),
                    max_pdf_pages=MAX_UPLOAD_PDF_PAGES,
                    progress_callback=on_progress,
                )
            progress.empty()
            db_path = result["db_dir"]
            collection_name = result["collection_name"]
            chunk_count = result["chunk_count"]
            st.session_state.user_upload_db_path = db_path
            st.session_state.user_upload_collection_name = collection_name
            st.session_state.uploads_signature_processed = current_signature
            get_user_vectorstore.clear()
            st.success(
                f"Upload ingestion complete for session `{st.session_state.session_id}` "
                f"collection `{collection_name}` with {chunk_count} chunks."
            )
            if result["parse_errors"]:
                st.warning("Some files could not be parsed:")
                for parse_error in result["parse_errors"]:
                    st.write(f"- {parse_error}")
        except Exception as err:
            error_id = new_error_id()
            logger.exception(
                "upload_ingestion_failed",
                extra={
                    "event": "upload_ingestion_failed",
                    "error_id": error_id,
                    "session_id": st.session_state.session_id,
                },
            )
            st.error(
                "Upload ingestion failed. Check file type/size/page limits and parser support. "
                f"Error ID: {error_id}. Details: {err}"
            )

user_docs_available = bool(
    st.session_state.user_upload_db_path
    and os.path.isdir(st.session_state.user_upload_db_path)
)

selected_subject = None
selected_chapter = None
if source_mode == "NCERT":
    selected_subject = st.selectbox(
        label="Select a Subject from class 12",
        options=subjects_list,
        index=None,
    )

    if selected_subject:
        chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]
        selected_chapter = st.selectbox(
            label=f"Select a Chapter from class 12 - {selected_subject}",
            options=chapter_list,
            index=0,
        )

    if selected_subject and selected_chapter:
        selection_changed = (
            st.session_state.selected_subject != selected_subject
            or st.session_state.selected_chapter != selected_chapter
        )
        if selection_changed:
            st.session_state.selected_subject = selected_subject
            st.session_state.selected_chapter = selected_chapter

assistant_idx = 0
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            citations = []
            videos = []
            if assistant_idx < len(st.session_state.citation_history):
                citations = st.session_state.citation_history[assistant_idx]
            if assistant_idx < len(st.session_state.video_history):
                videos = st.session_state.video_history[assistant_idx]
            _render_assistant_tabs(message["content"], citations, videos)
            assistant_idx += 1
        else:
            st.markdown(message["content"])

chat_ready = selected_subject is not None and selected_chapter is not None
if source_mode == "NCERT":
    chat_ready = selected_subject is not None and selected_chapter is not None
    if not chat_ready:
        st.info("Select subject and chapter first to start chatting.")
else:
    chat_ready = user_docs_available
    if not chat_ready:
        st.info("Upload and process files first, then ask questions in My Documents mode.")

active_vectorstore = None
active_user_vectorstore = None
if source_mode == "NCERT":
    if chat_ready:
        try:
            active_vectorstore = get_vectorstore()
        except FileNotFoundError as err:
            st.error(str(err))
            chat_ready = False
else:
    if chat_ready:
        try:
            active_user_vectorstore = get_user_vectorstore(
                st.session_state.user_upload_db_path,
                st.session_state.session_id,
            )
        except FileNotFoundError as err:
            st.error(str(err))
            chat_ready = False

if chat_ready and generate_quiz_clicked:
    try:
        with st.spinner("Generating chapter quiz..."):
            target_vectorstore = active_vectorstore if source_mode == "NCERT" else active_user_vectorstore
            quiz_markdown = generate_quiz_from_sources(
                st.session_state.selected_subject or "Physics",
                st.session_state.selected_chapter or "All Chapters",
                target_vectorstore,
                get_llm(),
                num_questions=quiz_count,
                output_language=output_language,
                use_metadata_filter=(source_mode == "NCERT"),
            )
        st.subheader("Generated Quiz")
        st.markdown(quiz_markdown)
    except FileNotFoundError as err:
        error_id = new_error_id()
        logger.exception(
            "quiz_generation_failed_file_not_found",
            extra={"event": "quiz_generation_failed", "error_id": error_id, "source_mode": source_mode},
        )
        st.error(f"{err} (Error ID: {error_id})")
    except Exception as err:
        error_id = new_error_id()
        logger.exception(
            "quiz_generation_failed",
            extra={"event": "quiz_generation_failed", "error_id": error_id, "source_mode": source_mode},
        )
        st.error(f"Quiz generation failed. Error ID: {error_id}. Details: {err}")
pdf_data = _chat_to_pdf_bytes(
    st.session_state.chat_history,
    st.session_state.selected_subject,
    st.session_state.selected_chapter,
    query_mode,
    output_language,
)
st.download_button(
    label="Export Session to PDF",
    data=pdf_data,
    file_name="study_sphere_session.pdf",
    mime="application/pdf",
    disabled=len(st.session_state.chat_history) == 0,
)

user_input = st.chat_input("Ask your question from selected sources", disabled=not chat_ready)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            fallback_message = UPLOADS_FALLBACK_ANSWER if source_mode == "My Documents" else FALLBACK_ANSWER
            target_vectorstore = active_vectorstore if source_mode == "NCERT" else active_user_vectorstore
            answer, low_confidence, citations = answer_from_sources(
                user_input=user_input,
                selected_subject=st.session_state.selected_subject or "Physics",
                selected_chapter=st.session_state.selected_chapter or "All Chapters",
                chat_history=st.session_state.chat_history,
                vectorstore=target_vectorstore,
                llm=get_llm(),
                reranker=get_reranker(),
                query_mode=query_mode,
                output_language=output_language,
                use_metadata_filter=(source_mode == "NCERT"),
                fallback_message=fallback_message,
            )
        except FileNotFoundError as err:
            error_id = new_error_id()
            logger.exception(
                "answer_generation_failed_file_not_found",
                extra={"event": "answer_generation_failed", "error_id": error_id, "source_mode": source_mode},
            )
            answer = f"{err} (Error ID: {error_id})"
            low_confidence = True
            citations = []
        except Exception as err:
            error_id = new_error_id()
            logger.exception(
                "answer_generation_failed",
                extra={"event": "answer_generation_failed", "error_id": error_id, "source_mode": source_mode},
            )
            answer = f"Something went wrong while generating the answer. Error ID: {error_id}"
            low_confidence = True
            citations = []

        if low_confidence:
            video_refs = []
        else:
            titles, links = get_yt_video_link(user_input)
            video_refs = []
            if titles and links:
                max_videos = min(3, len(titles), len(links))
                for i in range(max_videos):
                    video_refs.append((titles[i], links[i]))

        _render_assistant_tabs(answer, citations, video_refs)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.video_history.append(video_refs or [])
        st.session_state.citation_history.append(citations or [])
