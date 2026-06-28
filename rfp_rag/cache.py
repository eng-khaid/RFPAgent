"""Tiny persistent disk cache for the RFP pipeline.

Two layers use it (see ``document_loader`` and ``api``):
  - ``conversion``: file bytes -> extracted Markdown (skips slow docling).
  - ``result``: file bytes + config -> full evaluation JSON (skips docling AND
    the LLM call).

Values are stored as one JSON file per key under ``rfp_rag/.cache/<namespace>/``.
Set ``RFP_RAG_CACHE=off`` (or ``0``/``false``/``no``) to disable both layers.
Delete the ``.cache`` directory to clear everything.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def enabled() -> bool:
    return os.getenv("RFP_RAG_CACHE", "on").lower() not in {"0", "off", "false", "no"}


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def key_with_signature(content: bytes, signature: str) -> str:
    """Cache key from file bytes plus a config signature.

    Changing the signature (model, KB size, flags, ...) yields a new key, so a
    config change never returns a stale result.
    """
    return hashlib.sha256(file_hash(content).encode() + b"|" + signature.encode()).hexdigest()


def _path(namespace: str, key: str) -> Path:
    return CACHE_DIR / namespace / f"{key}.json"


def get(namespace: str, key: str) -> Any | None:
    if not enabled():
        return None
    path = _path(namespace, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # treat a corrupt entry as a miss


def put(namespace: str, key: str, value: Any) -> None:
    if not enabled():
        return
    path = _path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def clear() -> int:
    """Delete all cached entries. Returns how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.rglob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
