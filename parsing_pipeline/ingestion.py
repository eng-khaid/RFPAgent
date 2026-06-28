import html
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docling.document_converter import DocumentConverter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SCRAPED_DIR = Path(__file__).parent / "docs_md"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "scraped"

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}

# How aggressively to split. "percentile" creates a breakpoint wherever the
# distance between two sentences exceeds this percentile of all distances.
# Lower amount -> more breakpoints -> more, smaller chunks. (Default is 95.)
BREAKPOINT_THRESHOLD_TYPE = "percentile"
BREAKPOINT_THRESHOLD_AMOUNT = 85

# Chunks shorter than this (after cleaning) are stray fragments like "Proven."
# and add useless vectors, so we drop them.
MIN_CHUNK_CHARS = 30

# Matches a Markdown heading anywhere, not just at line start: SemanticChunker
# joins sentences with spaces, so a heading after a period (".. text. ## Next")
# is no longer line-anchored. Requires whitespace/start before the #'s and a
# space after, so anchors like "(#services)" never false-match.
_HEADING_RE = re.compile(r"(?:^|\s)(#{1,6})[ \t]+([^\n#]+)")
_HEADING_START_RE = re.compile(r"(#{1,6})[ \t]+")          # heading at the chunk's start
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)   # e.g. <!-- image -->
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # e.g. ![alt](src)


def build_chunker() -> SemanticChunker:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
    )


def _load_text(source: str) -> str:
    path = Path(source)
    if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
        return path.read_text(encoding="utf-8")
    # PDF, DOCX, HTML, URL, ... -> convert to Markdown first.
    return DocumentConverter().convert(source).document.export_to_markdown()


def _clean_markdown(text: str) -> str:
    text = _HTML_COMMENT_RE.sub("", text)   # drop <!-- image --> placeholders
    text = _IMAGE_MD_RE.sub("", text)       # drop ![alt](src) images
    text = html.unescape(text)              # &amp; -> &, &nbsp; -> space, etc.
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip()


def _clean_heading(text: str) -> str:
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.split())


def _enrich_chunks(chunks: list[str]) -> list[str]:
    stack: dict[int, str] = {}
    enriched: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        headings = [(len(m.group(1)), _clean_heading(m.group(2))) for m in _HEADING_RE.finditer(text)]

        # If the chunk *opens* with a heading, drop the breadcrumb down to its
        # parents (so we don't repeat the chunk's own heading or a stale sibling).
        # A heading that appears only later in the chunk doesn't count -- the
        # leading content still belongs under the full current path.
        opener = _HEADING_START_RE.match(text)
        cutoff = len(opener.group(1)) if opener else 7
        crumbs = [stack[lvl] for lvl in sorted(stack) if lvl < cutoff]
        breadcrumb = " > ".join(crumbs)

        if len(text) >= MIN_CHUNK_CHARS:
            enriched.append(f"{breadcrumb}\n\n{text}" if breadcrumb else text)

        # Update the heading path with every heading in this chunk (in order):
        # a heading at level N replaces level N and clears anything deeper.
        for level, htext in headings:
            stack[level] = htext
            for deeper in [lvl for lvl in stack if lvl > level]:
                del stack[deeper]
    return enriched


def chunk_document(source: str, chunker: SemanticChunker | None = None) -> list[str]:
    """Convert a document and return the list of semantic chunks ready to embed.

    `source` can be a file path or URL (Markdown, PDF, DOCX, HTML, ...).
    The returned chunks are cleaned, breadcrumb-prefixed, and noise-filtered.
    """
    chunker = chunker or build_chunker()
    text = _clean_markdown(_load_text(source))
    raw_chunks = chunker.split_text(text)
    return _enrich_chunks(raw_chunks)


def chunk_scraped(scraped_dir: Path = SCRAPED_DIR) -> dict[str, list[str]]:
    chunker = build_chunker()  # build once, reuse across all files
    results: dict[str, list[str]] = {}
    for md_path in sorted(scraped_dir.glob("*.md")):
        results[md_path.name] = chunk_document(str(md_path), chunker=chunker)
    return results


def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent Chroma collection.

    Chroma embeds documents with the SAME model used for chunking, so the
    token budget and the vectors stay consistent.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_ID)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_scraped(scraped_dir: Path = SCRAPED_DIR) -> chromadb.Collection:
    collection = get_collection()
    all_chunks = chunk_scraped(scraped_dir)

    total = 0
    for name, texts in all_chunks.items():
        if not texts:
            continue
        # Stable IDs (file::index) so re-running updates instead of duplicating.
        ids = [f"{name}::{i}" for i in range(len(texts))]
        metadatas = [{"source": name, "chunk": i} for i in range(len(texts))]
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        total += len(texts)
        print(f"  {name}: {len(texts)} chunks")

    print(f"Indexed {total} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    # Chunk text often contains characters (em-dashes, Arabic, ...) the Windows
    # console codec can't encode; force UTF-8 so prints don't crash.
    sys.stdout.reconfigure(encoding="utf-8")

    # Chunk the scraped Markdown and store the chunks in Chroma.
    collection = index_scraped()
    print(f"Collection now holds {collection.count()} chunks total.\n")

    # Quick sanity check: run a semantic query against what we just stored.
    results = collection.query(query_texts=["What does the company do?"], n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"--- {meta['source']} #{meta['chunk']} ---")
        print(doc[:200], "...\n")
