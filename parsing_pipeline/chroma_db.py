"""Showcase the chunks stored in the Chroma collection.

This is a read-only viewer for the vector store that `ingestion.py` builds.
Run `python ingestion.py` first to populate the collection, then use this to
inspect what got stored.

Examples:
    python chroma_db.py                 # list every chunk (preview)
    python chroma_db.py --full          # list every chunk (full text)
    python chroma_db.py --source adi.sa.md
    python chroma_db.py --query "What does the company do?" -n 3
"""

import argparse
import sys
import textwrap
from pathlib import Path

# Allow running as `python parsing_pipeline/chroma_db.py` from the repo root
# as well as `python chroma_db.py` from inside the package.
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import CHROMA_DIR, COLLECTION_NAME, get_collection  # noqa: E402

PREVIEW_CHARS = 280
RULE = "=" * 70


def _print_chunk(chunk_id: str, meta: dict, text: str, full: bool) -> None:
    source = meta.get("source", "?")
    idx = meta.get("chunk", "?")
    print(f"\n[{source} #{idx}]  id={chunk_id}  ({len(text)} chars)")
    print("-" * 70)
    if full:
        print(textwrap.indent(text, "    "))
    else:
        preview = text[:PREVIEW_CHARS].rstrip()
        ellipsis = " …" if len(text) > PREVIEW_CHARS else ""
        print(textwrap.indent(preview + ellipsis, "    "))


def showcase(source: str | None, full: bool) -> None:
    collection = get_collection()
    total = collection.count()
    print(RULE)
    print(f"Collection '{COLLECTION_NAME}'  ->  {total} chunks")
    print(f"Stored at: {CHROMA_DIR}")
    print(RULE)

    if total == 0:
        print("\nNo chunks yet. Run `python ingestion.py` to populate the store.")
        return

    where = {"source": source} if source else None
    data = collection.get(where=where, include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    if not ids:
        print(f"\nNo chunks found for source '{source}'.")
        return

    # Sort by (source, chunk index) for a stable, readable ordering.
    order = sorted(range(len(ids)), key=lambda i: (metas[i].get("source", ""), metas[i].get("chunk", 0)))

    # Per-source counts.
    counts: dict[str, int] = {}
    for m in metas:
        counts[m.get("source", "?")] = counts.get(m.get("source", "?"), 0) + 1
    print("\nChunks per source:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")

    print(f"\nShowing {len(ids)} chunk(s){' (full text)' if full else ' (preview)'}:")
    for i in order:
        _print_chunk(ids[i], metas[i], docs[i], full)


def search(query: str, n_results: int) -> None:
    collection = get_collection()
    if collection.count() == 0:
        print("No chunks to search. Run `python ingestion.py` first.")
        return

    print(RULE)
    print(f"Query: {query!r}  (top {n_results})")
    print(RULE)
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[None] * len(docs)])[0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        score = f"  (cosine distance {dist:.4f})" if dist is not None else ""
        print(f"\n#{rank}  [{meta.get('source')} #{meta.get('chunk')}]{score}")
        print("-" * 70)
        preview = doc[:PREVIEW_CHARS].rstrip()
        ellipsis = " …" if len(doc) > PREVIEW_CHARS else ""
        print(textwrap.indent(preview + ellipsis, "    "))


def main() -> None:
    # Chunk text often contains characters the Windows console can't encode.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Showcase chunks stored in Chroma.")
    parser.add_argument("--full", action="store_true", help="print full chunk text instead of a preview")
    parser.add_argument("--source", help="only show chunks from this source file (e.g. adi.sa.md)")
    parser.add_argument("--query", help="run a semantic search instead of listing chunks")
    parser.add_argument("-n", "--n-results", type=int, default=3, help="results to return for --query")
    args = parser.parse_args()

    if args.query:
        search(args.query, args.n_results)
    else:
        showcase(args.source, args.full)


if __name__ == "__main__":
    main()
