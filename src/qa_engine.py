from retrieval_utility import retrieve_with_mmr_and_rerank


FALLBACK_ANSWER = "I don't know from the selected source material."


def format_context(scored_docs, limit=6):
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


def answer_from_sources(
    user_input,
    selected_subject,
    selected_chapter,
    chat_history,
    vectorstore,
    llm,
    reranker,
):
    scored_docs, confidence, retrieval_cfg = retrieve_with_mmr_and_rerank(
        vectorstore=vectorstore,
        query=user_input,
        subject=selected_subject,
        chapter=selected_chapter,
        reranker=reranker,
    )

    if should_use_fallback(
        scored_docs=scored_docs,
        confidence=confidence,
        threshold=retrieval_cfg["confidence_threshold"],
    ):
        return FALLBACK_ANSWER, True

    context_text = format_context(scored_docs, limit=6)
    history_text = format_recent_history(chat_history)
    prompt = f"""
You are a study assistant. Answer ONLY from the provided context.
If context is insufficient, say exactly: {FALLBACK_ANSWER}
Do not invent facts. Keep answer concise and accurate.

Conversation history:
{history_text}

Question:
{user_input}

Context:
{context_text}
"""
    response = llm.invoke(prompt)
    answer_text = response.content.strip()
    fallback_used = answer_text.lower().startswith(FALLBACK_ANSWER.lower())
    return answer_text, fallback_used

