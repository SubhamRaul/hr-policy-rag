# DESIGN.md — Internal HR FAQ & Policy Assistant

## 1. Architecture and data flow

The goal is to answer questions only when the uploaded policy corpus contains enough evidence. Grounding, citations and refusal are treated as first-class requirements.

```text
UPLOAD
  │
  ├── validate extension / empty file
  ├── parse markdown/text/PDF
  └── identify section headings and table blocks
             │
             ▼
        CHUNK + METADATA
             │
             ├── document
             ├── section
             ├── chunk_id
             ├── chunk_type
             └── text
             │
             ▼
     LOCAL EMBEDDING MODEL
             │
             ▼
       PERSISTENT CHROMA
             │
             ├── Upload / Upsert
             ├── Clear all indexed policies
             │
QUERY ───────┘
  │
  ├── validate non-empty question
  ├── retrieve semantic top-k
  ├── calculate lexical overlap
  ├── hybrid rerank
  └── confidence gate
        │
        ├── weak evidence ────► deterministic refusal
        │
        ▼
grounded LLM prompt
        │
      Gemini
        │ │    
        │ └──failure / quota exceeded
        │          │
        │          ▼          
        │────── Ollama                    
        │          
        ▼          
Pydantic structured JSON
        │
        ▼
  validate citation chunk IDs
        │
        ├── invalid citation / malformed result ─► refusal
        ▼
  answer + citations[]
```

### Components

- `app/config.py`: stores the application settings, such as API keys, model names, etc.
- `app/ingestion.py`: parsing and chunking.
- `app/models.py`: Pydantic models define the expected request and response structures.
- `app/store.py`: Manages the embedding model, persistent ChromaDB collection, chunk storage, retrieval queries, and collection clearing.
- `app/retrieval.py`: hybrid retrieval and confidence gate.
- `app/llm.py`: grounded LLM generation using Gemini with local Ollama fallback and structured output.
- `app/main.py`: Handles API endpoints, file uploads, policy indexing, querying, and collection management.
- `ui/streamlit_app.py`: user interface.

## 2. Chunking and retrieval

### Chunking

The ingestion pipeline is section-aware.

1. Markdown headings (`#`, `##`, etc.) update the current section.
2. Normal prose is grouped into chunks around 900 characters with modest overlap.
3. Markdown tables are recognized as table blocks and kept together whenever possible.
4. Very large tables are split by rows while repeating the header.
5. Each chunk stores `document`, `section`, `chunk_type`, and a `chunk_id`.

This avoids blindly splitting a policy rule from its heading and protects structured benefits tables.

### Retrieval

The first stage retrieves 8 semantic candidates from Chroma using cosine distance.

For each candidate:

```text
semantic_similarity = 1 - cosine_distance
lexical_score       = query_token_overlap(candidate_text)
hybrid_score        = 0.75 * semantic_similarity + 0.25 * lexical_score
```

The final context contains at most 5 candidates after reranking.

Semantic search handles paraphrases such as "unused leave next year" vs "carry forward". Lexical matching helps exact policy terms.

### Confidence gate

The retrieved evidence is considered sufficiently relevant only when the strongest candidate satisfies both thresholds:

- its semantic similarity is at least `MIN_SEMANTIC_SIM`; and
- its hybrid score is at least `MIN_HYBRID_SCORE`.

This gate prevents obviously unrelated questions from being forwarded to the LLM.

The values are configuration parameters and should be calibrated against an evaluation set rather than treated as universal constants.

## 3. Grounding and anti-hallucination

The system uses multiple controls to reduce unsupported answers and ensure that responses are based only on retrieved policy evidence.

### A. Retrieval gate

Weak evidence is rejected before generation.

### B. Prompt restriction

The LLM receives the user question, selected policy chunks, and explicit grounding instructions:

- use only those chunks;
- never use general knowledge;
- never infer missing policy;
- return `answerable=false` when evidence is insufficient;
- cite only supplied chunk IDs.

### C. Structured output

The expected result is:

```json
{
  "answerable": true,
  "answer": "...",
  "citation_chunk_ids": ["..."]
}
```

Pydantic validates this structure.

### D. Citation validation

The backend does not trust arbitrary citation strings. Every returned citation ID must belong to the retrieved chunk set. The backend then resolves that ID to the actual uploaded document and section.

### E. Final refusal

If the model says `answerable=false`, has no citations for an answer, uses an unknown citation, or fails validation, the service returns:

> I don't have enough information in the uploaded policies. Please contact HR.

This makes “I don't know” a normal successful outcome.

### LLM provider fallback

Gemini is used as the primary LLM provider. If Gemini fails because of quota limits, API errors, or another generation error, the application can fall back to a local Ollama model.


## 4. Schema and APIs

### `POST /admin/upload`

Multipart upload.

Accepted: `.md`, `.txt`, `.pdf`.

The uploaded file is parsed, chunked, embedded, and stored in ChromaDB.

Response:

```json
{
  "document": "leave_policy.md",
  "chunks_indexed": 7
}
```

### `POST /query`

Request:

```json
{
  "question": "Does the Standard health tier cover dental implants?"
}
```

Response:

```json
{
  "answer": "Dental implants are not covered under the Standard tier.",
  "citations": [
    {
      "document": "health_benefits.md",
      "section": "Dental Benefits",
      "chunk_id": "..."
    }
  ],
  "refused": false,
  "reason": null
}
```

Unknown response:

```json
{
  "answer": "I don't have enough information in the uploaded policies. Please contact HR.",
  "citations": [],
  "refused": true,
  "reason": "No sufficiently relevant policy evidence was retrieved."
}
```

### `GET /health`

Returns service/index status.

### `DELETE /admin/clear`

Clears all indexed policy chunks from the ChromaDB collection.

```json
{
  "message": "All indexed policy chunks have been cleared.",
  "indexed_chunks": 0
}
```

## 5. Trade-offs

### Local embeddings vs hosted embeddings

**Chosen:** Local `sentence-transformers` embeddings.

used local embedding model so that the application does not need another external API for creating embeddings. It also avoids additional embedding costs.

The main disadvantage is that the model needs to be downloaded and uses local CPU and memory.

### Hybrid retrieval vs vector-only retrieval

**Chosen:** Hybrid retrieval.

The system combines semantic similarity with simple lexical matching.

Semantic search helps when the question uses different words from the policy. Lexical matching helps when the question contains exact terms such as policy names or benefit names.


### Section-aware chunking vs fixed-size chunking

**Chosen:** Section-aware and table-aware chunking.

Instead of splitting the document blindly into fixed-size pieces, the application tries to keep headings, paragraphs, and tables together.

This helps preserve the meaning of policy rules and makes the document section available for citations.

The disadvantage is that the chunking logic is more complex than basic fixed-size splitting.

### Deterministic refusal vs allowing the LLM to answer everything

**Chosen:** Retrieval confidence checks before calling the LLM.

The application checks whether the retrieved policy chunks are relevant enough. If the evidence is too weak, it returns a refusal instead of asking the LLM to guess.

This improves safety, but strict thresholds may sometimes reject a question that could have been answered. These thresholds should be tested and adjusted using sample questions.

## 6. What I Would Improve With More Time

If I had two more weeks, I would focus on testing the current system and improving the areas that may cause incorrect answers.

1. **Retrieval Evaluation & Weight Optimization** I would use a dataset to validate the semantic similarity and lexical overlap weights, and tune the hybrid retrieval score and confidence thresholds based on results.

2. **Improve document management** I would add support for replacing, updating, and deleting individual policy documents. When a policy is replaced or deleted, its previously indexed chunks should also be removed from ChromaDB

3. **Improve multi-part question handling** I would test questions that contain more than one request and make sure each part receives relevant evidence.

4. **Add user feedback** The UI could include simple positive and negative feedback buttons. This would help identify questions where retrieval or answer generation needs improvement.

5. **Add authentication** In a real HR application, different users would need different permissions. For example, only authorized HR users should be able to upload or update policies.
