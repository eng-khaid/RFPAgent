# RFP RAG Evaluator

This folder is for the uploaded-RFP workflow.

Flow:

1. Convert the uploaded RFP to text.
2. Embed the uploaded RFP as one temporary query.
3. Retrieve matching evidence from `parsing_pipeline/chroma_db`.
4. Ask DeepSeek `deepseek-chat` to judge feasibility using the retrieved evidence.
5. Return JSON that matches `userinterface/hak .html`.

The existing `parsing_pipeline/` remains the knowledge-base builder. This
folder only handles user-uploaded RFP analysis.

## Run The API

```powershell
uv run uvicorn rfp_rag.api:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://localhost:8000/
```

The frontend posts uploaded files to:

```text
POST http://localhost:8000/api/process
```

The default retrieval mode is a single temporary RFP embedding query against
the existing Chroma knowledge base. The uploaded RFP is not stored.

The default judge is DeepSeek:

```text
DEEPSEEK_API_KEY=...
```

The API loads that key from the project `.env`.

If the embedding model is not available locally yet, use this local fallback:

```powershell
$env:RFP_RAG_USE_VECTOR="false"
uv run uvicorn rfp_rag.api:app --reload --host 127.0.0.1 --port 8000
```

For UI/backend testing without calling DeepSeek:

```powershell
$env:RFP_RAG_USE_LLM="false"
uv run uvicorn rfp_rag.api:app --reload --host 127.0.0.1 --port 8000
```
