CHUNK_SIZE = 1000     # characters, about one paragraph
CHUNK_OVERLAP = 200   # characters shared between neighbouring chunks

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cpu"  # "cuda" if you want a GPU
DB_DIR = "chroma_db"
COLLECTION = "papers"

TOP_K = 5
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LLM_PROVIDER = "gemini"                # or "claude"
LLM_MODEL = "gemini-3.1-flash-lite"    # or e.g. "claude-sonnet-5-5"
JUDGE_MODEL = "gemini-3.8-flash"       # a different model than the one answering
MAX_TOKENS = 4096
