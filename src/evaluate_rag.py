import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from retrieval_utility import retrieve_with_mmr_and_rerank


def resolve_paths():
    working_dir = Path(__file__).resolve().parent
    parent_dir = working_dir.parent
    eval_dir = parent_dir / "eval"
    results_dir = eval_dir / "results"
    return working_dir, parent_dir, eval_dir, results_dir


def resolve_device() -> str:
    device = os.getenv("DEVICE", "cpu").strip().lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return device


def load_embeddings(device):
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
        )
    except NotImplementedError as err:
        if "meta tensor" not in str(err).lower():
            raise
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )


def load_dataset(dataset_path):
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_metadata_filter(subject, chapter):
    subject_filter = {"subject": subject.lower()}
    if chapter == "All Chapters":
        return subject_filter
    return {"$and": [subject_filter, {"chapter": chapter}]}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokenize(text):
    return normalize(text).split()


def token_f1(reference, prediction):
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    pred_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    overlap = 0
    for token, count in ref_counts.items():
        overlap += min(count, pred_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(text, expected_keywords):
    if not expected_keywords:
        return 0.0
    norm_text = normalize(text)
    matched = 0
    for keyword in expected_keywords:
        if normalize(keyword) in norm_text:
            matched += 1
    return matched / len(expected_keywords)


def groundedness(answer, contexts):
    answer_tokens = [t for t in tokenize(answer) if len(t) > 2]
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(" ".join(contexts)))
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


def confidence_from_similarity(top_score):
    return 1.0 / (1.0 + math.exp(-top_score))


def retrieval_baseline(vectorstore, question, subject, chapter, k=8):
    metadata_filter = build_metadata_filter(subject, chapter)
    scored = vectorstore.similarity_search_with_relevance_scores(
        query=question,
        k=k,
        filter=metadata_filter,
    )
    scored = [(doc, float(score)) for doc, score in scored if doc.page_content.strip()]
    return scored


def retrieval_improved(vectorstore, reranker, question, subject, chapter):
    return retrieve_with_mmr_and_rerank(
        vectorstore=vectorstore,
        query=question,
        subject=subject,
        chapter=chapter,
        reranker=reranker,
    )


def answer_with_llm_or_extractive(llm, question, contexts):
    if not contexts:
        return "I don't know from the selected source material."

    if llm is None:
        extract = contexts[0].strip()
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:2])[:400]

    context_block = "\n\n".join(contexts[:6])
    prompt = f"""
You are evaluating a RAG system. Answer ONLY from the context below.
If context is insufficient, say exactly: I don't know from the selected source material.

Question: {question}

Context:
{context_block}
"""
    return llm.invoke(prompt).content.strip()


def evaluate_pipeline(dataset, vectorstore, llm, pipeline_name, reranker=None):
    rows = []
    for sample in dataset:
        question = sample["question"]
        subject = sample["subject"]
        chapter = sample["chapter"]

        if pipeline_name == "baseline":
            scored_docs = retrieval_baseline(vectorstore, question, subject, chapter, k=8)
            confidence = confidence_from_similarity(scored_docs[0][1]) if scored_docs else 0.0
            threshold = 0.55
            selected_docs = scored_docs[:6]
        else:
            selected_docs, confidence, cfg = retrieval_improved(
                vectorstore=vectorstore,
                reranker=reranker,
                question=question,
                subject=subject,
                chapter=chapter,
            )
            threshold = cfg["confidence_threshold"]

        contexts = [doc.page_content for doc, _ in selected_docs]
        retrieval_hit = keyword_coverage(" ".join(contexts), sample["expected_keywords"]) > 0.0

        if not selected_docs or confidence < threshold:
            answer = "I don't know from the selected source material."
        else:
            answer = answer_with_llm_or_extractive(llm, question, contexts)

        ground = groundedness(answer, contexts)
        f1 = token_f1(sample["reference_answer"], answer)
        kw = keyword_coverage(answer, sample["expected_keywords"])
        accuracy = 0.7 * f1 + 0.3 * kw

        rows.append(
            {
                "id": sample["id"],
                "retrieval_hit": 1 if retrieval_hit else 0,
                "groundedness": ground,
                "answer_accuracy": accuracy,
                "confidence": confidence,
                "used_fallback": answer.lower().startswith("i don't know from the selected source material"),
            }
        )

    n = max(len(rows), 1)
    return {
        "samples": len(rows),
        "hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
        "groundedness": sum(r["groundedness"] for r in rows) / n,
        "answer_accuracy": sum(r["answer_accuracy"] for r in rows) / n,
        "fallback_rate": sum(1 for r in rows if r["used_fallback"]) / n,
        "details": rows,
    }


def update_readme(readme_path, metrics):
    start = "<!-- EVAL_RESULTS_START -->"
    end = "<!-- EVAL_RESULTS_END -->"

    baseline = metrics["baseline"]
    improved = metrics["improved"]
    delta_hit = improved["hit_rate"] - baseline["hit_rate"]
    delta_ground = improved["groundedness"] - baseline["groundedness"]
    delta_acc = improved["answer_accuracy"] - baseline["answer_accuracy"]

    block = (
        f"{start}\n"
        f"| Pipeline | Hit Rate | Groundedness | Answer Accuracy | Fallback Rate |\n"
        f"|---|---:|---:|---:|---:|\n"
        f"| Baseline | {baseline['hit_rate']:.3f} | {baseline['groundedness']:.3f} | {baseline['answer_accuracy']:.3f} | {baseline['fallback_rate']:.3f} |\n"
        f"| Improved | {improved['hit_rate']:.3f} | {improved['groundedness']:.3f} | {improved['answer_accuracy']:.3f} | {improved['fallback_rate']:.3f} |\n"
        f"| Delta | {delta_hit:+.3f} | {delta_ground:+.3f} | {delta_acc:+.3f} | {(improved['fallback_rate'] - baseline['fallback_rate']):+.3f} |\n"
        f"\n"
        f"Last updated: {metrics['run_at_utc']}\n"
        f"{end}"
    )

    original = readme_path.read_text(encoding="utf-8")
    if start in original and end in original:
        updated = re.sub(
            rf"{re.escape(start)}[\s\S]*?{re.escape(end)}",
            block,
            original,
            count=1,
        )
    else:
        updated = original.rstrip() + "\n\n## Evaluation Snapshot\n" + block + "\n"
    readme_path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs improved RAG pipeline")
    parser.add_argument(
        "--dataset",
        default="eval/ncert_qa_small.jsonl",
        help="Path to labeled QA dataset (.jsonl)",
    )
    parser.add_argument(
        "--vector-db",
        default="vector_db/class_12_unified_vector_db",
        help="Path to unified Chroma DB",
    )
    parser.add_argument(
        "--skip-readme-update",
        action="store_true",
        help="Do not update README evaluation snapshot block",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Evaluate improved pipeline without cross-encoder (uses lexical fallback reranking)",
    )
    args = parser.parse_args()

    working_dir, parent_dir, _, results_dir = resolve_paths()
    load_dotenv(parent_dir / ".env")
    load_dotenv(working_dir / ".env", override=True)

    dataset_path = parent_dir / args.dataset
    vector_db_path = parent_dir / args.vector_db
    readme_path = parent_dir / "README.md"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not vector_db_path.exists():
        raise FileNotFoundError(f"Unified vector DB not found at: {vector_db_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    device = resolve_device()
    embeddings = load_embeddings(device)
    vectorstore = Chroma(persist_directory=str(vector_db_path), embedding_function=embeddings)

    reranker = None
    if not args.disable_reranker:
        try:
            reranker = CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            reranker = None

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) if groq_api_key else None

    dataset = load_dataset(dataset_path)
    baseline = evaluate_pipeline(dataset, vectorstore, llm, pipeline_name="baseline")
    improved = evaluate_pipeline(dataset, vectorstore, llm, pipeline_name="improved", reranker=reranker)

    payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset),
        "samples": len(dataset),
        "baseline": baseline,
        "improved": improved,
    }

    latest_path = results_dir / "latest_metrics.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    history_path = results_dir / "metrics_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    if not args.skip_readme_update:
        update_readme(readme_path, payload)

    print(f"Saved: {latest_path}")
    print(f"Appended: {history_path}")
    print("Baseline:", {k: round(v, 3) for k, v in baseline.items() if isinstance(v, float)})
    print("Improved:", {k: round(v, 3) for k, v in improved.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
