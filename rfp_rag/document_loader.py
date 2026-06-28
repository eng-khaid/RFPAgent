from pathlib import Path

from docling.document_converter import DocumentConverter

from rfp_rag import cache

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def load_rfp_text(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"RFP file not found: {file_path}")

    content = file_path.read_bytes()
    # Conversion is a pure function of the file bytes, so cache it by content
    # hash: the same upload skips docling (the slow, model-loading step) entirely.
    key = cache.file_hash(content)
    cached = cache.get("conversion", key)
    if cached is not None:
        return cached

    if file_path.suffix.lower() in TEXT_SUFFIXES:
        text = content.decode("utf-8")
    else:
        text = DocumentConverter().convert(str(file_path)).document.export_to_markdown()

    cache.put("conversion", key, text)
    return text
