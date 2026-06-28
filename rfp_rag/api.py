from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from rfp_rag import cache
from rfp_rag.evaluator import RfpRagEvaluator
from rfp_rag.llm_judge import DEEPSEEK_MODEL

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_FILE = ROOT / "userinterface" / "hak.html"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown"}

app = FastAPI(title="RFPAgent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_evaluator() -> RfpRagEvaluator:
    use_vector = os.getenv("RFP_RAG_USE_VECTOR", "true").lower() in {"1", "true", "yes"}
    use_llm = os.getenv("RFP_RAG_USE_LLM", "true").lower() in {"1", "true", "yes"}
    top_k = int(os.getenv("RFP_RAG_TOP_K", "6"))
    max_docs = int(os.getenv("RFP_RAG_MAX_DOCS", "5"))
    return RfpRagEvaluator(
        top_k=top_k, use_vector=use_vector, use_llm=use_llm, max_docs=max_docs
    )


@app.get("/")
def frontend() -> FileResponse:
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend HTML file not found.")
    return FileResponse(FRONTEND_FILE)


@app.get("/api/health")
def health() -> dict:
    evaluator = get_evaluator()
    return {
        "status": "ok",
        "knowledge_base_chunks": evaluator.retriever.collection_count,
        "retrieval_mode": "hybrid_vector_bm25" if evaluator.use_vector else "bm25_only",
        "judge_mode": "deepseek-chat" if evaluator.use_llm else "draft_without_llm",
    }


@app.post("/api/process")
async def process_rfp(
    file: UploadFile = File(...),
    lang: str = "ar",
    refresh: bool = False,
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    language = lang.lower() if lang.lower() in {"ar", "en"} else "ar"

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 25 MB.")

    evaluator = get_evaluator()
    # Language is part of the key: the same RFP yields a separate cached result
    # per output language.
    signature = f"{_result_signature(evaluator)}|lang={language}"
    cache_key = cache.key_with_signature(content, signature)

    # Cache hit: skip docling + retrieval + the LLM call entirely.
    if not refresh:
        cached = cache.get("result", cache_key)
        if cached is not None:
            return _finalize(dict(cached), file.filename, cached=True)

    tmp_path = _write_temp_upload(content=content, suffix=suffix)
    try:
        result = evaluator.evaluate_file(str(tmp_path), language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process RFP: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    response = result.to_frontend_dict()
    # Store the raw result; filename-specific fields are stamped on every reply.
    cache.put("result", cache_key, response)
    return _finalize(response, file.filename, cached=False)


@app.post("/api/cache/clear")
def clear_cache() -> dict:
    return {"status": "ok", "cleared": cache.clear()}


def _result_signature(evaluator: RfpRagEvaluator) -> str:
    """Config that affects the output; a change here invalidates cached results."""
    return (
        f"llm={evaluator.use_llm}|vector={evaluator.use_vector}"
        f"|max_docs={evaluator.max_docs}|model={DEEPSEEK_MODEL}"
        f"|kb={evaluator.retriever.collection_count}|company={evaluator.company_name}"
    )


def _finalize(response: dict, filename: str | None, cached: bool) -> dict:
    if response.get("meta", {}).get("title") == "RFP feasibility review":
        response["meta"]["title"] = Path(filename or "RFP").stem
    response.setdefault("meta", {})["uploaded_file"] = filename
    response["meta"]["cached"] = cached
    return response


def _write_temp_upload(content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)
