import math
import re


DEFAULT_RETRIEVAL_CONFIG = {
    "output_k": 6,
    "candidate_k": 16,
    "fetch_k": 32,
    "lambda_mult": 0.5,
    "rerank_top_n": 6,
    "confidence_threshold": 0.55,
}

SUBJECT_RETRIEVAL_CONFIG = {
    "physics": {
        "output_k": 6,
        "candidate_k": 18,
        "fetch_k": 36,
        "lambda_mult": 0.35,
        "rerank_top_n": 6,
        "confidence_threshold": 0.58,
    },
    "chemistry": {
        "output_k": 6,
        "candidate_k": 16,
        "fetch_k": 32,
        "lambda_mult": 0.45,
        "rerank_top_n": 6,
        "confidence_threshold": 0.56,
    },
    "biology": {
        "output_k": 6,
        "candidate_k": 14,
        "fetch_k": 28,
        "lambda_mult": 0.6,
        "rerank_top_n": 6,
        "confidence_threshold": 0.53,
    },
}


def get_subject_retrieval_config(subject):
    cfg = dict(DEFAULT_RETRIEVAL_CONFIG)
    cfg.update(SUBJECT_RETRIEVAL_CONFIG.get(subject.lower(), {}))
    return cfg


def build_metadata_filter(subject, chapter):
    subject_filter = {"subject": subject.lower()}
    if chapter == "All Chapters":
        return subject_filter
    return {"$and": [subject_filter, {"chapter": chapter}]}


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _deduplicate_docs(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            (doc.metadata or {}).get("subject"),
            (doc.metadata or {}).get("chapter"),
            (doc.metadata or {}).get("page"),
            doc.page_content[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def rerank_documents(query, docs, reranker, top_n):
    if not docs:
        return []

    if reranker is None:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored_docs = []
        for doc in docs:
            doc_tokens = set(re.findall(r"[a-z0-9]+", doc.page_content.lower()))
            overlap = len(query_tokens.intersection(doc_tokens))
            denom = max(len(query_tokens), 1)
            scored_docs.append((doc, float(overlap / denom)))
    else:
        pairs = [(query, doc.page_content[:1200]) for doc in docs]
        scores = reranker.predict(pairs)
        scored_docs = [(doc, float(score)) for doc, score in zip(docs, scores)]

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    return scored_docs[:top_n]


def retrieve_with_mmr_and_rerank(vectorstore, query, subject, chapter, reranker):
    cfg = get_subject_retrieval_config(subject)
    metadata_filter = build_metadata_filter(subject, chapter)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": cfg["candidate_k"],
            "fetch_k": cfg["fetch_k"],
            "lambda_mult": cfg["lambda_mult"],
            "filter": metadata_filter,
        },
    )
    docs = retriever.invoke(query)
    docs = [doc for doc in docs if doc.page_content.strip()]
    docs = _deduplicate_docs(docs)

    reranked_docs = rerank_documents(
        query=query,
        docs=docs,
        reranker=reranker,
        top_n=cfg["rerank_top_n"],
    )

    if not reranked_docs:
        return [], 0.0, cfg

    confidence = _sigmoid(reranked_docs[0][1])
    return reranked_docs[: cfg["output_k"]], confidence, cfg
