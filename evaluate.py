import json
from rag_chain import rag_with_sources

def load_test_set(path="data/eval_questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]

def keyword_match(answer, keywords):
    """Check if all keywords appear in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    return len(matched) == len(keywords), matched

def evaluate():
    test_set = load_test_set()
    results = []
    
    print(f"Running evaluation on {len(test_set)} questions...\n")
    
    for test in test_set:
        qid = test["id"]
        question = test["question"]
        expected_kw = test["expected_keywords"]
        
        result = rag_with_sources(question)
        answer = result["answer"]
        sources = result["sources"]
        
        # Answer correctness
        correct, matched = keyword_match(answer, expected_kw)
        
        # Source check
        if test["expected_source_type"] == "none":
            source_ok = len(sources) == 0
        else:
            source_ok = len(sources) > 0
        
        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "correct": correct,
            "matched_keywords": matched,
            "source_ok": source_ok,
            "source_count": len(sources)
        })
        
        status = "✅" if correct and source_ok else "❌"
        print(f"{status} [{qid}] {question}")
        print(f"    Answer correct: {correct} (matched: {matched}/{expected_kw})")
        print(f"    Source correct: {source_ok} ({len(sources)} sources)")
        print()
    
    # Final report
    correct_count = sum(1 for r in results if r["correct"])
    source_count = sum(1 for r in results if r["source_ok"])
    total = len(results)
    
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Answer accuracy:   {correct_count}/{total} = {correct_count/total*100:.1f}%")
    print(f"Source accuracy:   {source_count}/{total} = {source_count/total*100:.1f}%")
    
    return results

if __name__ == "__main__":
    evaluate()