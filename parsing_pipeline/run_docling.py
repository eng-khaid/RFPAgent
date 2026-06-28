from pathlib import Path

from docling.document_converter import DocumentConverter

DOCS_DIR = Path(__file__).parent.parent / "docs"
OUT_DIR = Path(__file__).parent / "scraped_txt"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        return

    converter = DocumentConverter()

    for pdf in pdfs:
        out_path = OUT_DIR / f"{pdf.stem}.md"
        try:
            md = converter.convert(pdf).document.export_to_markdown()
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {pdf.name}: {exc}")
            continue
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {len(md):>6} chars -> {out_path.relative_to(Path(__file__).parent)}")

    print(f"\nDone. {len(pdfs)} PDF(s) processed into {OUT_DIR}")


if __name__ == "__main__":
    main()
