import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

API_URL = "http://localhost:8000/query"
JUDGE_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "llama3.2:3b"

DONT_KNOW_MARKERS = ["don't know", "do not know", "cannot find", "couldn't find",
                     "not aware", "no information", "not covered"]


def looks_like_refusal(answer):
    a = answer.lower()
    return any(m in a for m in DONT_KNOW_MARKERS)


def judge_grounded(question, answer, sources_text):
    """Ask the model itself whether the answer is actually supported by the sources."""
    prompt = (
        "You are a strict fact-checker. Given SOURCES and an ANSWER, reply with "
        "exactly one word: SUPPORTED if every factual claim in the answer is backed "
        "by the sources, or UNSUPPORTED if the answer contains claims not present "
        "in the sources.\n\n"
        f"SOURCES:\n{sources_text}\n\nQUESTION: {question}\nANSWER: {answer}\n\nVerdict:"
    )
    r = requests.post(JUDGE_URL, json={
        "model": JUDGE_MODEL, "stream": False, "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=120)
    r.raise_for_status()
    verdict = r.json()["message"]["content"].strip().upper()
    return "SUPPORTED" in verdict


def main():
    cases = [json.loads(l) for l in open("evals/testset.jsonl", encoding="utf-8") if l.strip()]
    refusal_correct = refusal_total = 0
    grounded_correct = grounded_total = 0

    for c in cases:
        r = requests.post(API_URL, json={"question": c["question"]}, timeout=180)
        r.raise_for_status()
        result = r.json()
        answer = result["answer"]
        refused = looks_like_refusal(answer)

        if not c["should_answer"]:
            refusal_total += 1
            ok = refused
            refusal_correct += ok
            print(f"{'PASS' if ok else 'FAIL'} [refusal]  {c['question']}")
            if not ok:
                print(f"   expected a refusal, got: {answer[:150]}")
        else:
            if refused:
                print(f"FAIL [wrongly refused]  {c['question']}")
                continue
            grounded_total += 1
            sources_text = "\n".join(
                f"[{s['n']}] ({s['heading']})" for s in result["sources"]
            )
            ok = judge_grounded(c["question"], answer, sources_text)
            grounded_correct += ok
            print(f"{'PASS' if ok else 'FAIL'} [grounded]  {c['question']}")
            if not ok:
                print(f"   answer: {answer[:200]}")

    print(f"\nRefusal accuracy: {refusal_correct}/{refusal_total}"
          f" = {refusal_correct/refusal_total:.0%}" if refusal_total else "\nNo refusal cases")
    print(f"Grounded-answer rate: {grounded_correct}/{grounded_total}"
          f" = {grounded_correct/grounded_total:.0%}" if grounded_total else "No answer cases")


if __name__ == "__main__":
    main()