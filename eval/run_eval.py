import json, time
from rag.config import TOP_K, LLM_MODEL, JUDGE_MODEL
from rag.retrieve import retrieve
from rag.generate import generate, call_llm, format_sources
from eval.questions import load_questions

JUDGE_SYSTEM = """You grade answers from a question-answering system over research papers.
You get the question, a reference answer written by the papers' author, the sources the system retrieved, and the system's answer.
Return ONLY a JSON object, no other text:
{"correctness": <integer 1-5>, "hallucination": <true or false>, "reason": "<one short sentence>"}
correctness: how well the system answer matches the reference answer. 5 = same key facts and no errors; 3 = partly right or missing key facts; 1 = wrong or no real answer.
If the reference says the papers don't cover the question, give 5 if the system declined to answer and 1 if it gave an answer.
hallucination: true if the system answer states anything that is not supported by the sources. A refusal is not a hallucination."""

def with_retry(fn, *args, tries=4, wait=60):
    """Retry on rate limits or 'high demand' errors from the API."""
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retrying in {wait}s: {str(e)[:80]}")
            time.sleep(wait)

def retrieval_metrics(q, chunks):
    """Paper-level precision, recall and hit for one question."""
    relevant = set(q["relevant_papers"])
    retrieved = [c["paper"] for c in chunks]
    precision = sum(p in relevant for p in retrieved) / len(retrieved)
    recall = len(relevant & set(retrieved)) / len(relevant)
    return {"precision": precision, "recall": recall, "hit": float(recall > 0)}

def judge(q, chunks, answer):
    prompt = (f"Question: {q['question']}\n\nReference answer: {q['reference_answer']}\n\n"
              f"Sources:\n{format_sources(chunks)}\n\nSystem answer: {answer}")
    text = with_retry(call_llm, JUDGE_SYSTEM, prompt, JUDGE_MODEL)
    result = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return {"correctness": int(result["correctness"]),
            "hallucination": str(result["hallucination"]).lower() == "true",
            "reason": str(result.get("reason", "")).replace("|", "/")}

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def write_report(rows):
    ret = [r for r in rows if "precision" in r]
    unans = [r for r in rows if "precision" not in r]
    summary = {
        f"Retrieval precision@{TOP_K}": mean([r["precision"] for r in ret]),
        f"Retrieval recall@{TOP_K}": mean([r["recall"] for r in ret]),
        f"Retrieval hit@{TOP_K}": mean([r["hit"] for r in ret]),
        "Answer correctness (mean, 1-5)": mean([r["correctness"] for r in rows]),
        "Answers judged correct (score >= 4)": mean([r["correctness"] >= 4 for r in rows]),
        "Hallucination rate": mean([r["hallucination"] for r in rows]),
        "Correct refusals (unanswerable questions)": mean([r["correctness"] >= 4 for r in unans]),
    }
    lines = ["# Evaluation results", "",
             f"{len(rows)} questions, {len(unans)} of them not answerable from the papers. "
             f"Answer model: `{LLM_MODEL}`. Judge model: `{JUDGE_MODEL}`.", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in summary.items()]
    lines += ["", f"| Question | P@{TOP_K} | R@{TOP_K} | Correctness | Hallucination | Judge's reason |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        p = f"{r['precision']:.2f}" if "precision" in r else "-"
        rec = f"{r['recall']:.2f}" if "recall" in r else "-"
        lines.append(f"| {r['id']} | {p} | {rec} | {r['correctness']} | "
                     f"{'yes' if r['hallucination'] else 'no'} | {r['reason']} |")
    open("eval/results.md", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:5 + len(summary)]))

def main():
    path = "eval/results.json"
    try:
        rows = json.load(open(path))  # resume: keep questions already evaluated
    except FileNotFoundError:
        rows = []
    done = {r["id"] for r in rows}
    for q in load_questions():
        if q["id"] in done:
            continue
        chunks = retrieve(q["question"])
        answer = with_retry(generate, q["question"], chunks)
        row = {"id": q["id"], "question": q["question"], "answer": answer, **judge(q, chunks, answer)}
        if q["relevant_papers"]:
            row.update(retrieval_metrics(q, chunks))
        rows.append(row)
        json.dump(rows, open(path, "w"), indent=2)  # save progress after every question
        print(f"{q['id']}  correctness={row['correctness']}  hallucination={row['hallucination']}")
        time.sleep(15)  # stay under free-tier rate limits
    rows.sort(key=lambda r: r["id"])
    write_report(rows)

if __name__ == "__main__":
    main()
