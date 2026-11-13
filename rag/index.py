import chromadb
from sentence_transformers import SentenceTransformer
from rag.config import EMBED_MODEL, EMBED_DEVICE, DB_DIR, COLLECTION

_model = None

def embed(texts):
    """List of strings -> list of normalized vectors."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

def get_collection():
    return chromadb.PersistentClient(path=DB_DIR).get_or_create_collection(COLLECTION)

def build_index(chunks):
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # start fresh each time
    except Exception:
        pass
    col = client.create_collection(COLLECTION)
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{"paper": c["paper"], "page": c["page"]} for c in chunks],
    )
    return col

if __name__ == "__main__":
    from rag.loader import load_pdfs
    from rag.chunker import chunk_pages
    col = build_index(chunk_pages(load_pdfs()))
    print(f"Indexed {col.count()} chunks into {DB_DIR}/")
