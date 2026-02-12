from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from qa_engine import answer_from_sources, FALLBACK_ANSWER


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(lower)),
            float(lower.count("electric")),
            float(lower.count("field")),
            float(lower.count("charge")),
        ]


@dataclass
class _Response:
    content: str


class FakeLLM:
    def invoke(self, prompt):
        if "electric field" in prompt.lower():
            return _Response("Electric field is force per unit positive test charge.")
        return _Response(FALLBACK_ANSWER)


def test_end_to_end_query_against_fixture_db(tmp_path):
    persist_dir = tmp_path / "tiny_db"
    docs = [
        Document(
            page_content=(
                "Electric field at a point is defined as force experienced by a unit positive "
                "test charge placed at that point."
            ),
            metadata={
                "subject": "physics",
                "chapter": "1. Electric Charges and Fields",
                "page": 0,
            },
        )
    ]
    embeddings = FakeEmbeddings()
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    answer, used_fallback = answer_from_sources(
        user_input="What is electric field?",
        selected_subject="Physics",
        selected_chapter="1. Electric Charges and Fields",
        chat_history=[],
        vectorstore=vectorstore,
        llm=FakeLLM(),
        reranker=None,
    )

    assert used_fallback is False
    assert "force per unit" in answer.lower()

