import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link


# ------------------------- LOAD ENV ------------------------- #
load_dotenv()
DEVICE = os.getenv('DEVICE', 'cpu')

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

subjects_list = ["Physics", "Chemistry", "Biology"]


# ------------------------- VECTOR DB PATH ------------------------- #
def get_vector_db_path(chapter, subject):
    subject = subject.lower()

    if chapter == "All Chapters":
        return f"{parent_dir}/vector_db/class_12_{subject}_vector_db"

    return f"{parent_dir}/chapters_vector_db/{subject}/{chapter}"


# ------------------------- SETUP CHAIN ------------------------- #
def setup_chain(selected_chapter, selected_subject):
    vector_db_path = get_vector_db_path(selected_chapter, selected_subject)

    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": DEVICE})
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    memory = ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        ),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return chain


# ------------------------- STREAMLIT UI ------------------------- #
st.set_page_config(
    page_title="Study Sphere",
    page_icon="♻️",
    layout="centered"
)

st.title("📚 Study Sphere")

# Session State Setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []


# ------------------------- SELECT SUBJECT ------------------------- #
selected_subject = st.selectbox(
    label="Select a Subject from class 12",
    options=subjects_list,
    index=None
)

if selected_subject:
    chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]

    selected_chapter = st.selectbox(
        label=f"Select a Chapter from class 12 - {selected_subject}",
        options=chapter_list,
        index=0
    )

    if selected_chapter:
        if st.session_state.get('selected_chapter') != selected_chapter:
            st.session_state.chat_chain = setup_chain(
                selected_chapter, selected_subject
            )

        st.session_state.selected_chapter = selected_chapter


# ------------------------- DISPLAY CHAT HISTORY ------------------------- #
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            videos = st.session_state.video_history[idx]
            if videos:
                st.subheader("Video Reference")
                for title, link in videos:
                    st.info(f"{title}\n\nLink: {link}")


# ------------------------- USER INPUT ------------------------- #
user_input = st.chat_input("Ask Your Doubts Here!")

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.video_history.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.chat_chain({"question": user_input})
        answer = response["answer"]
        st.markdown(answer)

        # -----------------------------------------------------------
        # 100% FINAL VIDEO BLOCKER (NEVER FAILS)
        # -----------------------------------------------------------

        answer_lower = answer.strip().lower()

        block_keywords = [
            "not provided",
            "not mentioned",
            "not in this chapter",
            "not found",
            "no information",
            "not related",
            "doesn't mention",
            "does not mention",
            "the text you provided",
            "the provided text",
            "the given text",
            "the context does not",
            "context does not",
            "irrelevant",
            "cannot be found",
            "not available in this chapter",
        ]

        # Check phrasing
        should_block_video = any(keyword in answer_lower for keyword in block_keywords)

        # Also block if retrieved text is irrelevant / too small
        retrieved_text = ""
        if "source_documents" in response:
            retrieved_text = " ".join(
                [d.page_content for d in response["source_documents"]]
            )
            if len(retrieved_text.strip()) < 50:
                should_block_video = True

        # FINAL DECISION
        if should_block_video:
            video_refs = []   # do NOT show any video section OR message
        else:
            titles, links = get_yt_video_link(user_input)
            video_refs = []

            if titles and links:
                st.subheader("Video Reference")
                max_videos = min(3, len(titles), len(links))

                for i in range(max_videos):
                    st.info(f"{titles[i]}\n\nLink: {links[i]}")
                    video_refs.append((titles[i], links[i]))
            else:
                video_refs = []   # silently ignore

        # Save to streamlit session
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.video_history.append(video_refs)