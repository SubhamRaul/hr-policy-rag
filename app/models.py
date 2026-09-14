from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    section: str
    chunk_id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    refused: bool
    reason: str | None = None


class UploadResponse(BaseModel):
    document: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    citation_chunk_ids: list[str] = []
