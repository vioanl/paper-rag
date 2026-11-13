def load_questions(path="eval/questions.txt"):
    """Read Q/P/A blocks from the text file into a list of dicts."""
    questions = []
    for block in open(path).read().split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l and not l.startswith("#")]
        if not lines:
            continue
        fields = {l[0]: l[2:].strip() for l in lines}
        questions.append({
            "id": f"q{len(questions) + 1:02d}",
            "question": fields["Q"],
            "relevant_papers": [p.strip() for p in fields.get("P", "").split(",") if p.strip()],
            "reference_answer": fields["A"],
        })
    return questions

if __name__ == "__main__":
    from collections import Counter
    qs = load_questions()
    print(len(qs), "questions")
    print(Counter(p for q in qs for p in q["relevant_papers"]))
    print("still to fill:", [q["id"] for q in qs if "WRITE" in q["question"] + q["reference_answer"]])
