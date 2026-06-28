"""Hybrid retrieval over chunks created by ``parsing_pipeline.ingestion``.

The parsing pipeline stores semantic chunks in Chroma with:
    - collection name: ``scraped``
    - IDs: ``{source_file}::{chunk_index}``
    - metadata: ``{"source": source_file, "chunk": chunk_index}``

This module reads that same persisted collection and combines Chroma vector
search with BM25 keyword search via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parsing_pipeline.ingestion import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_ID,
)


TOP_K = 5
OVERSAMPLING_FACTOR = 4
VECTOR_PENALTY = 60.0
FULLTEXT_PENALTY = 60.0
VECTOR_WEIGHT = 1.0
FULLTEXT_WEIGHT = 1.0


class HybridSearchRetriever:
    def __init__(
        self,
        top_k: int = TOP_K,
        oversampling_factor: int = OVERSAMPLING_FACTOR,
        vector_penalty: float = VECTOR_PENALTY,
        fulltext_penalty: float = FULLTEXT_PENALTY,
        vector_weight: float = VECTOR_WEIGHT,
        fulltext_weight: float = FULLTEXT_WEIGHT,
        use_vector: bool = True,
    ) -> None:
        self.top_k = top_k
        self.oversampling_factor = oversampling_factor
        self.vector_penalty = vector_penalty
        self.fulltext_penalty = fulltext_penalty
        self.vector_weight = vector_weight
        self.fulltext_weight = fulltext_weight
        self.use_vector = use_vector

        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_collection(name=COLLECTION_NAME)
        self._vector_collection = None

        self._documents = self._load_documents()
        self._bm25_retriever = self._build_bm25_retriever()

    @property
    def collection_count(self) -> int:
        return len(self._documents)

    def _load_documents(self) -> list[Document]:
        data = self._collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        documents: list[Document] = []
        for chunk_id, text, metadata in zip(ids, texts, metadatas):
            if not text:
                continue

            doc_metadata = dict(metadata or {})
            doc_metadata["id"] = chunk_id
            documents.append(Document(page_content=text, metadata=doc_metadata))

        documents.sort(
            key=lambda doc: (
                str(doc.metadata.get("source", "")),
                int(doc.metadata.get("chunk", 0)),
            )
        )
        return documents

    def _build_bm25_retriever(self) -> BM25Retriever | None:
        if not self._documents:
            return None

        retriever = BM25Retriever.from_documents(self._documents)
        retriever.k = self._candidate_count()
        return retriever

    def _candidate_count(self) -> int:
        if not self._documents:
            return 0
        return min(len(self._documents), self.top_k * self.oversampling_factor)

    def _get_vector_collection(self):
        """Open the same collection with the same embedding function as ingestion."""
        if self._vector_collection is None:
            embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_ID)
            self._vector_collection = self._client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
            )
        return self._vector_collection

    def _vector_search(self, question: str, k: int) -> list[Document]:
        results = self._get_vector_collection().query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        texts = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        documents: list[Document] = []
        for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            doc_metadata = dict(metadata or {})
            doc_metadata["id"] = chunk_id
            doc_metadata["vector_distance"] = distance
            documents.append(Document(page_content=text, metadata=doc_metadata))
        return documents

    def retrieve_by_embedding(self, text: str) -> list[Document]:
        text = text.strip()
        candidates = self._candidate_count()
        if not text or candidates == 0:
            return []
        return self._vector_search(text, candidates)[: self.top_k]

    def _bm25_search(self, question: str) -> list[Document]:
        if self._bm25_retriever is None:
            return []
        return self._bm25_retriever.invoke(question)

    @staticmethod
    def _document_key(doc: Document) -> str:
        chunk_id = doc.metadata.get("id")
        if chunk_id:
            return str(chunk_id)

        source = doc.metadata.get("source")
        chunk = doc.metadata.get("chunk")
        if source is not None and chunk is not None:
            return f"{source}::{chunk}"

        return doc.page_content

    @classmethod
    def _reciprocal_rank_fusion(
        cls,
        result_lists: list[list[Document]],
        weights: list[float],
        penalties: list[float],
    ) -> list[tuple[Document, float]]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list, weight, penalty in zip(result_lists, weights, penalties):
            for rank, doc in enumerate(result_list, start=1):
                key = cls._document_key(doc)
                scores[key] = scores.get(key, 0.0) + weight / (rank + penalty)
                doc_map.setdefault(key, doc)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(doc_map[key], score) for key, score in ranked]

    def _fused_documents(self, question: str) -> list[tuple[Document, float]]:
        question = question.strip()
        candidates = self._candidate_count()
        if not question or candidates == 0:
            return []

        vector_docs: list[Document] = []
        if self.use_vector:
            try:
                vector_docs = self._vector_search(question, candidates)
            except Exception as exc:
                warnings.warn(
                    f"Vector search is unavailable; falling back to BM25 only. ({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
        fulltext_docs = self._bm25_search(question)

        return self._reciprocal_rank_fusion(
            result_lists=[vector_docs, fulltext_docs],
            weights=[self.vector_weight, self.fulltext_weight],
            penalties=[self.vector_penalty, self.fulltext_penalty],
        )

    def retrieve(self, question: str) -> list[Document]:
        return [doc for doc, _score in self._fused_documents(question)[: self.top_k]]

    def retrieve_sources(self, question: str, max_docs: int) -> list[str]:
        """Single-query convenience wrapper around :meth:`retrieve_sources_multi`."""
        return self.retrieve_sources_multi([question], max_docs)

    def retrieve_sources_multi(self, queries: list[str], max_docs: int) -> list[str]:
        """Rank source documents across many sub-queries (e.g. RFP chunks).

        Each query is scored with hybrid fusion (vector + BM25 + RRF). Every
        source document is then scored by its single best-matching chunk across
        all queries (best-match aggregation), so a document that strongly answers
        even one requirement ranks high even if it's irrelevant to the rest.

        Ranking stays at chunk granularity (precise), but the caller feeds each
        document's full text, so a fact is never dropped just because its chunk
        ranked low. Returns up to ``max_docs`` source filenames, highest first.
        """
        best: dict[str, float] = {}
        for query in queries:
            for doc, score in self._fused_documents(query):
                source = str(doc.metadata.get("source", ""))
                if source and score > best.get(source, float("-inf")):
                    best[source] = score

        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return [source for source, _score in ranked[:max_docs]]

    def document_text(self, source: str) -> str:
        """Stitch the stored chunks for a source back into one ordered string.

        Fallback for when the original Markdown file is not on disk; chunk text
        carries breadcrumb prefixes, so this is rougher than the source file.
        """
        chunks = [d for d in self._documents if str(d.metadata.get("source", "")) == source]
        chunks.sort(key=lambda doc: int(doc.metadata.get("chunk", 0)))
        return "\n\n".join(doc.page_content for doc in chunks)

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        formatted: list[str] = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk")
            chunk_label = f" #{chunk}" if chunk is not None else ""
            label = f"[Source {i}: {source}{chunk_label}]"

            metadata_text = "\n".join(
                f"{key}: {value}" for key, value in doc.metadata.items()
            )
            formatted.append(
                f"{label}\n"
                f"Metadata:\n{metadata_text}\n\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

