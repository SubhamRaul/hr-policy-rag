from __future__ import annotations

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from .config import get_settings
from .ingestion import Chunk


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    model = get_embedding_model()
    embeddings = model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "document": c.document,
                "section": c.section,
                "chunk_type": c.chunk_type,
            }
            for c in chunks
        ],
        embeddings=embeddings,
    )
    return len(chunks)


def query_chunks(question: str, n_results: int) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    model = get_embedding_model()
    query_embedding = model.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        items.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "document": metadata.get("document", "unknown"),
                "section": metadata.get("section", "Document"),
                "chunk_type": metadata.get("chunk_type", "prose"),
                "distance": float(distance),
                "semantic_similarity": max(
                    0.0, min(1.0, 1.0 - float(distance))
                ),
            }
        )
    return items


def collection_count() -> int:
    return get_collection().count()


def clear_collection() -> None:
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    get_collection.cache_clear()
