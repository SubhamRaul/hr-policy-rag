# Internal HR FAQ & Policy Assistant

A small, grounded Retrieval-Augmented Generation (RAG) service for answering employee questions **only from uploaded HR policy documents**.

## What it demonstrates

- Upload `.md`, `.txt`, and `.pdf` policy files.
- Section-aware and table-aware chunking.
- Local embeddings using `sentence-transformers`.
- Persistent vector search with ChromaDB.
- Hybrid retrieval using semantic similarity and keyword overlap.
- Grounded Gemini responses with local Ollama fallback and citations.
- Safe refusal when the answer is not found in the policies.
- Admin endpoint to clear all indexed policy chunks.
- FastAPI backend with a simple Streamlit UI.

## Architecture

```text
                    ┌──────────────────┐
                    │ Admin: upload    │
                    │ .md/.txt/.pdf    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Parser + section │
                    │ / table-aware    │
                    │ chunker          │
                    └────────┬─────────┘
                             │
                    local embeddings
                             │
                             ▼
                    ┌──────────────────┐
                    │ ChromaDB         │
                    │ persistent       │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
             Upload / Upsert       Clear all policies
                  │                     │
                  └──────────┬──────────┘
                             │ top-k
                             ▼
                       Hybrid retrieval
                      (semantic + keyword)
                             │
                      confidence gate
                      ┌─────┴─────┐
                    weak        strong
                     │             │
                     ▼             ▼
                   REFUSE     Grounded LLM
                                     │
                              ┌──────┴──────┐
                              │             │
                           Gemini        Ollama
                           primary       fallback
                              │             │
                              └──────┬──────┘
                                     │
                              Pydantic JSON
                                     │
                              citation validator
                                     │
                                     ▼
                                JSON response
```

## Stack

- Python 3.11+
- FastAPI
- Streamlit
- ChromaDB
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local embeddings
- Google Gemini API as the primary LLM
- Ollama as the local fallback LLM
- Pydantic for structured output
- PyPDF for PDF text extraction

The default LLM is Gemini. If Gemini is unavailable because of quota, API, or connection errors, the application falls back to a local Ollama model.

## 1. Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini key.

Install and run Ollama locally, then pull the fallback model configured in your environment variables.

## 2. Start the API

```bash
uvicorn app.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

## 3. Start the UI

In another terminal:

```bash
streamlit run ui/streamlit_app.py
```

## API examples

Upload:

```bash
curl -X POST "http://127.0.0.1:8000/admin/upload" \
  -F "file=@E:/policies/leave_policy.md"
```

Ask:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many casual leave days can be carried forward?"}'
```

Example response:

```json
{
  "answer": "Employees may carry forward up to 5 unused casual leave days.",
  "citations": [
    {
      "document": "leave_policy.md",
      "section": "Casual Leave",
      "chunk_id": "..."
    }
  ],
  "refused": false,
  "reason": null
}
```
Clear indexed policies:

```bash
curl -X DELETE "http://127.0.0.1:8000/admin/clear"
```

Example response:

```json
{
  "message": "All indexed policy chunks have been cleared.",
  "indexed_chunks": 0
}
```

Unknown questions return a safe refusal instead of a general-knowledge answer.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes for Gemini | — | Gemini API key |
| `LLM_MODEL` | No | `gemini-3.5-flash` | Primary Gemini model |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Local Ollama server URL |
| `OLLAMA_MODEL` | No | `llama3.2:3b` | Local fallback model |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_DIR` | No | `.chroma` | Persistent vector store |
| `COLLECTION_NAME` | No | `hr_policies` | Chroma collection |
| `TOP_K` | No | `8` | Initial retrieval candidates |
| `FINAL_K` | No | `5` | Context chunks sent to the LLM |
| `MIN_SEMANTIC_SIM` | No | `0.30` | Retrieval safety threshold |
| `MIN_HYBRID_SCORE` | No | `0.28` | Combined retrieval threshold |
| `MAX_UPLOAD_MB` | No | `10` | Upload size limit |

