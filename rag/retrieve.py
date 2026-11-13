from rag.config import TOP_K, QUERY_PREFIX
from rag.index import embed, get_collection

def retrieve(question, k=TOP_K):
    """Question -> top-k chunks with paper, page and cosine similarity."""
    res = get_collection().query(query_embeddings=embed([QUERY_PREFIX + question]), n_results=k)
    return [
        {"id": i, "text": doc, "paper": meta["paper"], "page": meta["page"],
         "score": round(1 - dist / 2, 3)}  # Chroma gives squared L2; for unit vectors cos = 1 - d/2
        for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                      res["metadatas"][0], res["distances"][0])
    ]

if __name__ == "__main__":
    import sys
    for r in retrieve(" ".join(sys.argv[1:])):
        print(f"{r['score']}  {r['paper']}  p{r['page']}\n    {r['text'][:150]}...\n")
