from __future__ import annotations

import re
from dataclasses import dataclass

from .config import get_settings
from .store import query_chunks


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}", re.I)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    section: str
    chunk_type: str
    text: str
    semantic_similarity: float
    lexical_score: float
    hybrid_score: float


def tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 2
    }


def lexical_overlap(question: str, text: str) -> float:
    q = tokens(question)
    if not q:
        return 0.0
    t = tokens(text)
    return len(q & t) / len(q)


def retrieve(question: str) -> list[RetrievedChunk]:
    settings = get_settings()
    candidates = query_chunks(question, settings.top_k)

    ranked: list[RetrievedChunk] = []
    for item in candidates:
        lexical = lexical_overlap(question, item["text"])
        semantic = item["semantic_similarity"]
        hybrid = 0.75 * semantic + 0.25 * lexical
        ranked.append(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document=item["document"],
                section=item["section"],
                chunk_type=item["chunk_type"],
                text=item["text"],
                semantic_similarity=semantic,
                lexical_score=lexical,
                hybrid_score=hybrid,
            )
        )

    ranked.sort(key=lambda x: x.hybrid_score, reverse=True)
    return ranked[: settings.final_k]


def evidence_is_strong(results: list[RetrievedChunk]) -> bool:
    if not results:
        return False
    settings = get_settings()
    best = results[0]
    return (
        best.semantic_similarity >= settings.min_semantic_sim
        and best.hybrid_score >= settings.min_hybrid_score
    )
