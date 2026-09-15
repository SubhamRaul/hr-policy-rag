from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .ingestion import chunk_document, extract_text
from .llm import answer_question, validate_llm_answer
from .models import Citation, HealthResponse, QueryRequest, QueryResponse, UploadResponse
from .retrieval import evidence_is_strong, retrieve
from .store import collection_count, upsert_chunks , clear_collection


app = FastAPI(
    title="Internal HR FAQ & Policy Assistant",
    version="1.0.0",
    description="Grounded RAG over uploaded HR policy documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}


def safe_filename(name: str | None) -> str:
    original = Path(name or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", original)
    return cleaned or "upload"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=collection_count())

@app.delete("/admin/clear")
def clear_policies() -> dict:
    try:
        clear_collection()
        return {
            "message": "All indexed policy chunks have been cleared.",
            "indexed_chunks": collection_count(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not clear policy collection: {exc}",
        ) from exc

@app.post("/admin/upload", response_model=UploadResponse)
async def upload_policy(file: UploadFile = File(...)) -> UploadResponse:
    filename = safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .md, .txt, or .pdf.",
        )

    settings = get_settings()
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )

    try:
        text = extract_text(filename, data)
        chunks = chunk_document(text, filename)
        count = upsert_chunks(chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not index document: {exc}",
        ) from exc

    return UploadResponse(document=filename, chunks_indexed=count)


@app.post("/query", response_model=QueryResponse)
def query_policy(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()

    if not question:
        return QueryResponse(
            answer="I don't have enough information in the uploaded policies. Please contact HR.",
            citations=[],
            refused=True,
            reason="Question is empty.",
        )

    results = retrieve(question)

    if not evidence_is_strong(results):
        return QueryResponse(
            answer="I don't have enough information in the uploaded policies. Please contact HR.",
            citations=[],
            refused=True,
            reason="No sufficiently relevant policy evidence was retrieved.",
        )

    try:
        llm_result = answer_question(question, results)
        validated = validate_llm_answer(llm_result, results)

    except Exception as exc:
        print(f"LLM ERROR: {type(exc).__name__}: {exc}")

        return QueryResponse(
            answer="I don't have enough information in the uploaded policies. Please contact HR.",
            citations=[],
            refused=True,
            reason=f"Grounded answer generation failed: {type(exc).__name__}: {exc}",
        )

    if validated is None or not validated.answerable:
        return QueryResponse(
            answer="I don't have enough information in the uploaded policies. Please contact HR.",
            citations=[],
            refused=True,
            reason="The retrieved policy evidence was not sufficient to answer.",
        )

    by_id = {r.chunk_id: r for r in results}
    citations = [
        Citation(
            document=by_id[chunk_id].document,
            section=by_id[chunk_id].section,
            chunk_id=chunk_id,
        )
        for chunk_id in validated.citation_chunk_ids
    ]

    return QueryResponse(
        answer=validated.answer.strip(),
        citations=citations,
        refused=False,
        reason=None,
    )
