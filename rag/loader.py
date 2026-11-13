from pathlib import Path
import pymupdf

def load_pdfs(folder="data/papers"):
    """Return a list of {paper, page, text}, one per page."""
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append({"paper": pdf.stem, "page": i, "text": page.get_text()})
    return pages

if __name__ == "__main__":
    pages = load_pdfs()
    print(f"{len(pages)} pages from {len({p['paper'] for p in pages})} papers")
    print(pages[0]["text"][:500])
