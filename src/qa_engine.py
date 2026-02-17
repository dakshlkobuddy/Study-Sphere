import math
import difflib
import re

from retrieval_utility import (
    retrieve_with_mmr_and_rerank,
    build_metadata_filter,
    get_subject_retrieval_config,
    rerank_documents,
)


FALLBACK_ANSWER = "I don't know from the selected source material."
UPLOADS_FALLBACK_ANSWER = "Not found in uploaded docs."
LOW_RELIABILITY_MESSAGE = "I couldn't find reliable info in selected sources."
FALLBACK_ANSWER_HI = "चयनित स्रोत सामग्री में उत्तर नहीं मिला।"
UPLOADS_FALLBACK_ANSWER_HI = "चयनित अपलोडेड दस्तावेज़ों में उत्तर नहीं मिला।"
LOW_RELIABILITY_MESSAGE_HI = "चयनित स्रोतों में विश्वसनीय जानकारी नहीं मिली।"


def _mode_instruction(query_mode):
    mode = (query_mode or "Explain").strip().lower()
    if mode == "exam answer":
        return "Write a structured exam-style answer with key points and concise conclusion."
    if mode == "short notes":
        return "Write short notes in 5-8 crisp bullet points."
    if mode == "mcq":
        return (
            "Answer as one multiple-choice question with exactly 4 options (A-D), "
            "then provide the correct option and one-line explanation."
        )
    return "Explain clearly in simple student-friendly language."


def _language_instruction(language):
    lang = (language or "English").strip().lower()
    if lang == "hindi":
        return "Return the full answer in Hindi."
    return "Return the full answer in English."


def format_context(scored_docs, limit=3):
    context_parts = []
    for idx, (doc, _) in enumerate(scored_docs[:limit], start=1):
        metadata = doc.metadata or {}
        chapter_name = metadata.get("chapter", "Unknown Chapter")
        page_number = metadata.get("page")
        if isinstance(page_number, int):
            page_number += 1
        else:
            page_number = "N/A"
        context_parts.append(
            f"[S{idx}] Chapter: {chapter_name} | Page: {page_number}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_parts)


def build_citations(scored_docs, limit=4):
    citations = []
    for doc, score in scored_docs[:limit]:
        metadata = doc.metadata or {}
        page = metadata.get("page")
        if isinstance(page, int):
            page = page + 1
        elif page is None:
            page = metadata.get("page_number", "N/A")

        citations.append(
            {
                "filename": metadata.get("filename") or metadata.get("uploaded_file") or "N/A",
                "chapter": metadata.get("chapter", "N/A"),
                "page": page if page is not None else "N/A",
                "chunk_id": metadata.get("chunk_id", "N/A"),
                "score": float(score),
                "snippet": doc.page_content[:220].replace("\n", " ").strip(),
            }
        )
    return citations


def format_recent_history(messages, max_pairs=3):
    if not messages:
        return ""
    recent = messages[-(max_pairs * 2) :]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def should_use_fallback(scored_docs, confidence, threshold):
    if not scored_docs:
        return True
    return confidence < threshold


def _extract_vocab(text):
    return set(re.findall(r"[a-z]{4,}", (text or "").lower()))


def _build_source_vocab(
    user_input,
    selected_subject,
    selected_chapter,
    vectorstore,
    use_metadata_filter,
):
    search_kwargs = {"query": user_input, "k": 14}
    if use_metadata_filter:
        search_kwargs["filter"] = build_metadata_filter(selected_subject, selected_chapter)
    docs = vectorstore.similarity_search(**search_kwargs)

    vocab = set()
    for doc in docs:
        vocab.update(_extract_vocab(doc.page_content))
    vocab.update(_extract_vocab(selected_subject))
    vocab.update(_extract_vocab(selected_chapter))
    return vocab


def _correct_query_with_vocab(user_input, vocab):
    if not vocab:
        return user_input

    def _replace(match):
        word = match.group(0)
        lower = word.lower()
        if len(lower) < 5 or not lower.isalpha() or lower in vocab:
            return word
        suggestion = difflib.get_close_matches(lower, vocab, n=1, cutoff=0.82)
        return suggestion[0] if suggestion else word

    return re.sub(r"\b[A-Za-z]{5,}\b", _replace, user_input)


def _try_typo_retry(
    user_input,
    selected_subject,
    selected_chapter,
    vectorstore,
    reranker,
    use_metadata_filter,
):
    vocab = _build_source_vocab(
        user_input=user_input,
        selected_subject=selected_subject,
        selected_chapter=selected_chapter,
        vectorstore=vectorstore,
        use_metadata_filter=use_metadata_filter,
    )
    corrected_query = _correct_query_with_vocab(user_input, vocab)
    if corrected_query.strip().lower() == user_input.strip().lower():
        return None, None, None, None

    scored_docs, confidence, retrieval_cfg = _retrieve_scored_docs(
        user_input=corrected_query,
        selected_subject=selected_subject,
        selected_chapter=selected_chapter,
        vectorstore=vectorstore,
        reranker=reranker,
        use_metadata_filter=use_metadata_filter,
    )
    return corrected_query, scored_docs, confidence, retrieval_cfg


def _retrieve_scored_docs(
    user_input,
    selected_subject,
    selected_chapter,
    vectorstore,
    reranker,
    use_metadata_filter,
):
    if use_metadata_filter:
        scored_docs, confidence, retrieval_cfg = retrieve_with_mmr_and_rerank(
            vectorstore=vectorstore,
            query=user_input,
            subject=selected_subject,
            chapter=selected_chapter,
            reranker=reranker,
        )
        return scored_docs, confidence, retrieval_cfg

    retrieval_cfg = get_subject_retrieval_config(selected_subject or "physics")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retrieval_cfg["candidate_k"],
            "fetch_k": retrieval_cfg["fetch_k"],
            "lambda_mult": retrieval_cfg["lambda_mult"],
        },
    )
    docs = [doc for doc in retriever.invoke(user_input) if doc.page_content.strip()]
    scored_docs = rerank_documents(
        query=user_input,
        docs=docs,
        reranker=reranker,
        top_n=retrieval_cfg["rerank_top_n"],
    )
    confidence = 0.0
    if scored_docs:
        confidence = 1.0 / (1.0 + math.exp(-scored_docs[0][1]))
    return scored_docs, confidence, retrieval_cfg


def answer_from_sources(
    user_input,
    selected_subject,
    selected_chapter,
    chat_history,
    vectorstore,
    llm,
    reranker,
    query_mode="Explain",
    output_language="English",
    use_metadata_filter=True,
    fallback_message=FALLBACK_ANSWER,
    low_reliability_message=LOW_RELIABILITY_MESSAGE,
):
    query_for_prompt = user_input
    scored_docs, confidence, retrieval_cfg = _retrieve_scored_docs(
        user_input=user_input,
        selected_subject=selected_subject,
        selected_chapter=selected_chapter,
        vectorstore=vectorstore,
        reranker=reranker,
        use_metadata_filter=use_metadata_filter,
    )

    initial_fallback = should_use_fallback(
        scored_docs=scored_docs,
        confidence=confidence,
        threshold=retrieval_cfg["confidence_threshold"],
    ) or (scored_docs and confidence < 0.5)
    if initial_fallback:
        corrected_query, retry_docs, retry_confidence, retry_cfg = _try_typo_retry(
            user_input=user_input,
            selected_subject=selected_subject,
            selected_chapter=selected_chapter,
            vectorstore=vectorstore,
            reranker=reranker,
            use_metadata_filter=use_metadata_filter,
        )
        if corrected_query is not None:
            scored_docs = retry_docs
            confidence = retry_confidence
            retrieval_cfg = retry_cfg
            query_for_prompt = corrected_query

    if scored_docs and confidence < 0.5:
        return low_reliability_message, True, []

    if should_use_fallback(
        scored_docs=scored_docs,
        confidence=confidence,
        threshold=retrieval_cfg["confidence_threshold"],
    ):
        return fallback_message, True, []

    context_text = format_context(scored_docs, limit=3)
    citations = build_citations(scored_docs)
    history_text = format_recent_history(chat_history)
    mode_instruction = _mode_instruction(query_mode)
    language_instruction = _language_instruction(output_language)
    prompt = f"""
You are a study assistant. Answer ONLY from the provided context.
If context is insufficient, say exactly: {fallback_message}
Do not invent facts. Keep answer concise and accurate.
{mode_instruction}
{language_instruction}

Conversation history:
{history_text}

Question:
{query_for_prompt}

Context:
{context_text}
"""
    response = llm.invoke(prompt)
    answer_text = response.content.strip()
    fallback_used = answer_text.lower().startswith(fallback_message.lower())
    return answer_text, fallback_used, citations


def answer_from_multiple_sources(
    user_input,
    selected_subject,
    selected_chapter,
    chat_history,
    source_specs,
    llm,
    reranker,
    query_mode="Explain",
    output_language="English",
    fallback_message=FALLBACK_ANSWER,
):
    merged_docs = []
    confidence_threshold = 0.0

    for spec in source_specs:
        docs, _, cfg = _retrieve_scored_docs(
            user_input=user_input,
            selected_subject=selected_subject,
            selected_chapter=selected_chapter,
            vectorstore=spec["vectorstore"],
            reranker=reranker,
            use_metadata_filter=spec["use_metadata_filter"],
        )
        confidence_threshold = max(confidence_threshold, cfg["confidence_threshold"])
        merged_docs.extend(doc for doc, _ in docs)

    reranked = rerank_documents(
        query=user_input,
        docs=merged_docs,
        reranker=reranker,
        top_n=3,
    )
    confidence = 0.0
    if reranked:
        confidence = 1.0 / (1.0 + math.exp(-reranked[0][1]))

    if reranked and confidence < 0.5:
        return LOW_RELIABILITY_MESSAGE, True, []

    if should_use_fallback(
        scored_docs=reranked,
        confidence=confidence,
        threshold=confidence_threshold,
    ):
        return fallback_message, True, []

    context_text = format_context(reranked, limit=3)
    citations = build_citations(reranked)
    history_text = format_recent_history(chat_history)
    mode_instruction = _mode_instruction(query_mode)
    language_instruction = _language_instruction(output_language)
    prompt = f"""
You are a study assistant. Answer ONLY from the provided context.
If context is insufficient, say exactly: {fallback_message}
Do not invent facts. Keep answer concise and accurate.
{mode_instruction}
{language_instruction}

Conversation history:
{history_text}

Question:
{user_input}

Context:
{context_text}
"""
    response = llm.invoke(prompt)
    answer_text = response.content.strip()
    fallback_used = answer_text.lower().startswith(fallback_message.lower())
    return answer_text, fallback_used, citations


def generate_quiz_from_sources(
    selected_subject,
    selected_chapter,
    vectorstore,
    llm,
    num_questions=5,
    output_language="English",
    use_metadata_filter=True,
):
    if use_metadata_filter:
        metadata_filter = build_metadata_filter(selected_subject, selected_chapter)
        docs = vectorstore.similarity_search(
            query="key concepts, definitions, formulas and examples",
            k=14,
            filter=metadata_filter,
        )
    else:
        docs = vectorstore.similarity_search(
            query="key concepts, definitions, formulas and examples",
            k=14,
        )
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        return FALLBACK_ANSWER

    context_text = format_context([(doc, 0.0) for doc in docs], limit=10)
    language_instruction = _language_instruction(output_language)
    prompt = f"""
Create a chapter quiz from the context.
Generate exactly {int(num_questions)} questions.
Each question should have 4 options (A-D), mention the correct option, and include a one-line explanation.
{language_instruction}
If context is insufficient, say exactly: {FALLBACK_ANSWER}

Context:
{context_text}
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_quiz_from_multiple_sources(
    selected_subject,
    selected_chapter,
    source_specs,
    llm,
    num_questions=5,
    output_language="English",
):
    merged_docs = []
    for spec in source_specs:
        if spec["use_metadata_filter"]:
            metadata_filter = build_metadata_filter(selected_subject, selected_chapter)
            docs = spec["vectorstore"].similarity_search(
                query="key concepts, definitions, formulas and examples",
                k=10,
                filter=metadata_filter,
            )
        else:
            docs = spec["vectorstore"].similarity_search(
                query="key concepts, definitions, formulas and examples",
                k=10,
            )
        merged_docs.extend([doc for doc in docs if doc.page_content.strip()])

    if not merged_docs:
        return FALLBACK_ANSWER

    context_text = format_context([(doc, 0.0) for doc in merged_docs], limit=12)
    language_instruction = _language_instruction(output_language)
    prompt = f"""
Create a chapter quiz from the context.
Generate exactly {int(num_questions)} questions.
Each question should have 4 options (A-D), mention the correct option, and include a one-line explanation.
{language_instruction}
If context is insufficient, say exactly: {FALLBACK_ANSWER}

Context:
{context_text}
"""
    response = llm.invoke(prompt)
    return response.content.strip()
