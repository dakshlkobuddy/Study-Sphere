# Study Sphere
Study Sphere is an intelligent learning companion built with Streamlit, Groq LLM, ChromaDB, and LangChain. It supports class-12 subject/chapter Q&A using a unified metadata-filtered RAG pipeline.

## Core Features
- Subject and chapter aware Q&A (`Physics`, `Chemistry`, `Biology`)
- Unified Chroma vector DB with metadata filters (`subject`, `chapter`, `page`)
- MMR retrieval + cross-encoder reranking
- Confidence-based fallback: `I don't know from the selected source material.`
- Optional YouTube video references for high-confidence answers

## Build Unified Vector DB
```bash
python src/vectorize_script.py --unified --recreate-unified
```

## Run App
```bash
streamlit run src/main.py
```

## Automated Tests
Run locally:
```bash
pytest -q
```

Coverage in this suite includes:
- Unit tests for chapter discovery (`chatbot_utility`)
- Unit tests for DB-path and env loading (`app_config`)
- Unit tests for fallback behavior (`qa_engine`)
- Integration test for one end-to-end query against a tiny fixture Chroma DB

CI:
- GitHub Actions workflow: `.github/workflows/tests.yml`
- Runs on every push and pull request

## Evaluation Framework
The project includes an automated RAG evaluation utility with a labeled NCERT-style QA dataset.

- Dataset: `eval/ncert_qa_small.jsonl`
- Evaluator: `src/evaluate_rag.py`
- Metrics:
- `hit_rate`: retrieval contains expected concept keywords
- `groundedness`: fraction of answer tokens supported by retrieved context
- `answer_accuracy`: weighted score using token F1 + keyword coverage
- `fallback_rate`: share of questions answered with fallback string

Run evaluation:
```bash
python src/evaluate_rag.py
```

If cross-encoder download is blocked/offline:
```bash
python src/evaluate_rag.py --disable-reranker
```

Outputs:
- Latest run: `eval/results/latest_metrics.json`
- History log: `eval/results/metrics_history.jsonl`
- README snapshot auto-updated between markers below

## Evaluation Snapshot
<!-- EVAL_RESULTS_START -->
| Pipeline | Hit Rate | Groundedness | Answer Accuracy | Fallback Rate |
|---|---:|---:|---:|---:|
| Baseline | 0.889 | 0.874 | 0.441 | 0.056 |
| Improved | 0.944 | 0.813 | 0.367 | 0.167 |
| Delta | +0.056 | -0.060 | -0.074 | +0.111 |

Last updated: 2026-02-12T09:02:14.791761+00:00
<!-- EVAL_RESULTS_END -->
