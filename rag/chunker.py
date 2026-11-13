import re
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def clean(text):
    text = re.sub(r"-\n(\w)", r"\1", text)  # join words split across lines
    text = re.sub(r"\s+", " ", text)        # newlines and extra spaces -> one space
    return text.strip()

def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                  # don't cut a word in half
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)  # start at a word, not mid-word
        if space != -1:
            start = space + 1
    return chunks

def chunk_pages(pages):
    """Pages -> chunks with {id, paper, page, text}. Skips the reference list."""
    chunks, finished = [], set()
    for p in pages:
        if p["paper"] in finished:
            continue
        parts = re.split(r"(?:^|\n)\s*(?:\d+\.\s*)?References\s*\n", p["text"], maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            finished.add(p["paper"])
        for i, piece in enumerate(split_text(clean(parts[0]))):
            chunks.append({"id": f"{p['paper']}_p{p['page']}_c{i}",
                           "paper": p["paper"], "page": p["page"], "text": piece})
    return chunks

if __name__ == "__main__":
    from collections import Counter
    from rag.loader import load_pdfs
    chunks = chunk_pages(load_pdfs())
    print(f"{len(chunks)} chunks, avg {sum(len(c['text']) for c in chunks) // len(chunks)} chars")
    for paper, n in Counter(c["paper"] for c in chunks).items():
        last = max(c["page"] for c in chunks if c["paper"] == paper)
        print(f"  {paper}: {n} chunks, last page used {last}")
    print("\nExample:", chunks[10])
