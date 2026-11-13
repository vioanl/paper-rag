# paper-rag

Ask questions about a set of research papers and get answers that cite the paper and page each part came from.

A small retrieval-augmented generation (RAG) pipeline in plain Python:

- **Loading:** PDFs read with PyMuPDF; reference lists removed
- **Chunking:** ~1000-character chunks with 200-character overlap, kept within one page so each chunk has an exact page number
- **Embeddings:** `BAAI/bge-small-en-v1.5` (sentence-transformers), stored in Chroma
- **Retrieval:** top-5 chunks by cosine similarity
- **Generation:** Gemini (Claude also supported, set in `rag/config.py`), instructed to answer only from the retrieved chunks and cite every sentence

## Run

```bash
pip install -r requirements.txt
# put PDFs in data/papers/ and your key in .env (GEMINI_API_KEY=... or ANTHROPIC_API_KEY=...)
python -m rag.index
python ask.py "your question"
```

## Example

```
$ python ask.py "How does correlation alignment transfer structural knowledge from teacher to student?"

Correlation alignment transfers structural knowledge from the teacher to the student by aligning
second-order statistics, which capture fine-grained spatial correlations and local structural
dependencies between spatial features within an image [2][5]. Specifically, a contrastive
correlation loss uses a thresholding mechanism acting as a structural confidence filter: when the
teacher's spatial correlation is strong, the student is encouraged to increase its corresponding
spatial correlation, whereas weak teacher correlations suppress the alignment to avoid transferring
spurious architectural noise [2]. This process ensures the student inherits the teacher's detailed,
high-confidence understanding of spatial textures and relational characteristics rather than just
final output labels [1][2][4].

Sources:
  [1] ivc2026_cakd_wildfire_retrieval, p3
  [2] ivc2026_cakd_wildfire_retrieval, p5
  [4] cakd_image_retrieval_conf, p5
  [5] ivc2026_cakd_wildfire_retrieval, p4
```

## Evaluation

In progress: ground-truth question set, retrieval precision/recall, and LLM-as-judge scoring of correctness and hallucination.
