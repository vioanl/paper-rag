import re, sys
from rag.retrieve import retrieve
from rag.generate import generate

def answer(question):
    chunks = retrieve(question)
    return generate(question, chunks), chunks

if __name__ == "__main__":
    text, chunks = answer(" ".join(sys.argv[1:]))
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    print(text, "\n\nSources:")
    for n in cited:
        c = chunks[n - 1]
        print(f"  [{n}] {c['paper']}, p{c['page']}")
