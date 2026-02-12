import os
import streamlit as st
from sentence_transformers import CrossEncoder

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app_config import load_project_env, resolve_device, get_unified_vector_db_path
from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link
from qa_engine import answer_from_sources


# ------------------------- LOAD ENV ------------------------- #
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = load_project_env(working_dir)


DEVICE = resolve_device()
UNIFIED_VECTOR_DB_PATH = get_unified_vector_db_path(parent_dir)
subjects_list = ["Physics", "Chemistry", "Biology"]


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
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


@st.cache_resource
def get_reranker():
    try:
        return CrossEncoder("BAAI/bge-reranker-base")
    except Exception:
        return None


st.set_page_config(page_title="Study Sphere", page_icon="S", layout="centered")
st.title("Study Sphere")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None
if "selected_chapter" not in st.session_state:
    st.session_state.selected_chapter = None

selected_subject = st.selectbox(
    label="Select a Subject from class 12",
    options=subjects_list,
    index=None,
)

selected_chapter = None
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

for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            videos = st.session_state.video_history[idx]
            if videos:
                st.subheader("Video Reference")
                for title, link in videos:
                    st.info(f"{title}\n\nLink: {link}")

chat_ready = selected_subject is not None and selected_chapter is not None
if not chat_ready:
    st.info("Select subject and chapter first to start chatting.")

user_input = st.chat_input("Ask your question from selected sources", disabled=not chat_ready)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.video_history.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            answer, low_confidence = answer_from_sources(
                user_input,
                st.session_state.selected_subject,
                st.session_state.selected_chapter,
                st.session_state.chat_history,
                get_vectorstore(),
                get_llm(),
                get_reranker(),
            )
        except FileNotFoundError as err:
            answer = str(err)
            low_confidence = True

        st.markdown(answer)

        if low_confidence:
            video_refs = []
        else:
            titles, links = get_yt_video_link(user_input)
            video_refs = []
            if titles and links:
                st.subheader("Video Reference")
                max_videos = min(3, len(titles), len(links))
                for i in range(max_videos):
                    st.info(f"{titles[i]}\n\nLink: {links[i]}")
                    video_refs.append((titles[i], links[i]))

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.video_history.append(video_refs)
