import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.main import embed_query, retrieve  # reuse the real app code

def main():
    cases = [json.loads(l) for l in open("evals/testset.jsonl", encoding="utf-8") if l.strip()]
    hits, checked = 0, 0
    for c in cases:
        if not c["expected_source_contains"]:
            continue
        checked += 1
        rows = retrieve(embed_query(c["question"]))
        found = any(c["expected_source_contains"] in r[0] for r in rows)
        hits += found
        print(f"{'PASS' if found else 'FAIL'}  {c['question']}")
        if not found:
            print(f"   expected source containing: {c['expected_source_contains']}")
            print(f"   got: {[r[0] for r in rows]}")
    print(f"\nRetrieval accuracy: {hits}/{checked} = {hits/checked:.0%}")

if __name__ == "__main__":
    main()