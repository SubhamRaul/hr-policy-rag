# Internal HR FAQ & Policy Assistant

A small, grounded Retrieval-Augmented Generation (RAG) service for answering employee questions **only from uploaded HR policy documents**.

## What it demonstrates

- Upload `.md`, `.txt`, and `.pdf` policy files.
- Section-aware and table-aware chunking.
- Local embeddings using `sentence-transformers`.
- Persistent vector search with ChromaDB.
- Hybrid retrieval using semantic similarity and keyword overlap.
- Grounded Gemini responses with citations.
- Safe refusal when the answer is not found in the policies.
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
                             │ top-k
                             ▼
                       Hybrid retrieval
                       semantic + keyword
                             │
                      confidence gate
                       ┌─────┴─────┐
                    weak          strong
                     │              │
                     ▼              ▼
                  REFUSE       Gemini JSON
                               grounded only
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
- Google Gemini API for the final answer
- Pydantic for structured output
- PyPDF for PDF text extraction

The default LLM is `gemini-3.5-flash`. Embeddings are local, so no embedding API key is required.

## 1. Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini key.

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

Unknown questions return a safe refusal instead of a general-knowledge answer.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes for answering | — | Gemini API key |
| `LLM_MODEL` | No | `gemini-2.5-flash-lite` | Final answer model |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_DIR` | No | `.chroma` | Persistent vector store |
| `COLLECTION_NAME` | No | `hr_policies` | Chroma collection |
| `TOP_K` | No | `8` | Initial retrieval candidates |
| `FINAL_K` | No | `5` | Context chunks sent to the LLM |
| `MIN_SEMANTIC_SIM` | No | `0.30` | Retrieval safety threshold |
| `MIN_HYBRID_SCORE` | No | `0.28` | Combined retrieval threshold |
| `MAX_UPLOAD_MB` | No | `10` | Upload size limit |


## Known limitation

PDFs are processed using PyPDF text extraction. Scanned PDFs and complex tables may not be extracted correctly.
