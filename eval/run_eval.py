"""
Evaluation harness — runs 10 test questions and reports pass/fail.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --verbose

Each test case specifies:
  - question, country, language
  - expected_citation_ids: at least one of these must appear in citations (OR logic)
  - expected_keywords:     all of these must appear in the answer (case-insensitive AND logic)
  - expect_empty:          if True, expect zero citations (isolation / fallback test)
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import ask_async  # noqa: E402

TEST_CASES = [
    # 1 — Country A, English: account closure
    {
        "id": "TC01",
        "description": "Country A / en — account closure rules",
        "question": "How do I close my account?",
        "country": "A",
        "language": "en",
        "expected_citation_ids": ["a_faq_account_en", "a_tc_en_v2"],
        "expected_keywords": ["support", "balance"],
        "expect_empty": False,
    },
    # 2 — Country A, Hindi: return policy
    {
        "id": "TC02",
        "description": "Country A / hi — return policy in Hindi",
        "question": "आपकी वापसी नीति क्या है?",
        "country": "A",
        "language": "hi",
        "expected_citation_ids": ["a_faq_returns_hi", "a_tc_hi_v2"],
        "expected_keywords": ["48"],
        "expect_empty": False,
    },
    # 3 — Country B, Spanish: account closure (different rules from A)
    {
        "id": "TC03",
        "description": "Country B / es — account closure in Spanish",
        "question": "¿Cómo cierro mi cuenta?",
        "country": "B",
        "language": "es",
        "expected_citation_ids": ["b_faq_account_es", "b_tc_es_v3"],
        "expected_keywords": ["saldo"],
        "expect_empty": False,
    },
    # 4 — Country B, English: return policy (7 days — NOT 48 hours like Country A)
    {
        "id": "TC04",
        "description": "Country B / en — return window is 7 days (not 48h)",
        "question": "What is your return policy?",
        "country": "B",
        "language": "en",
        "expected_citation_ids": ["b_faq_returns_en", "b_tc_en_v3"],
        "expected_keywords": ["7 days"],
        "expect_empty": False,
    },
    # 5 — Country C, French Canadian: return policy
    {
        "id": "TC05",
        "description": "Country C / fr_CA — return policy in French",
        "question": "Quelle est votre politique de retour?",
        "country": "C",
        "language": "fr_CA",
        "expected_citation_ids": ["c_faq_returns_fr", "c_tc_fr_v1"],
        "expected_keywords": ["14"],
        "expect_empty": False,
    },
    # 6 — Country C, English: payment methods
    {
        "id": "TC06",
        "description": "Country C / en — payment methods",
        "question": "What payment methods do you accept?",
        "country": "C",
        "language": "en",
        "expected_citation_ids": ["c_faq_payment_en", "c_tc_en_v1"],
        "expected_keywords": ["credit card"],
        "expect_empty": False,
    },
    # 7 — Country D, English: delivery time
    {
        "id": "TC07",
        "description": "Country D / en — delivery time",
        "question": "When will my order be delivered?",
        "country": "D",
        "language": "en",
        "expected_citation_ids": ["d_faq_delivery_en", "d_tc_en_v1"],
        "expected_keywords": ["business day"],
        "expect_empty": False,
    },
    # 8 — ISOLATION TEST: Country A asking in Spanish (no Spanish content for A)
    {
        "id": "TC08",
        "description": "ISOLATION: Country A / es — must return empty (no Spanish in Country A)",
        "question": "¿Cuál es su política de devoluciones?",
        "country": "A",
        "language": "es",
        "expected_citation_ids": [],
        "expected_keywords": [],
        "expect_empty": True,
    },
    # 9 — ISOLATION TEST: Country D asking in French (no French content for D)
    {
        "id": "TC09",
        "description": "ISOLATION: Country D / fr_CA — must return empty",
        "question": "Comment fermer mon compte?",
        "country": "D",
        "language": "fr_CA",
        "expected_citation_ids": [],
        "expected_keywords": [],
        "expect_empty": True,
    },
    # 10 — Country B, English: pending orders on account closure
    {
        "id": "TC10",
        "description": "Country B / en — pending orders don't block closure (contradicts Country A)",
        "question": "Can I cancel my account if I have pending orders?",
        "country": "B",
        "language": "en",
        "expected_citation_ids": ["b_faq_account_en", "b_tc_en_v3"],
        "expected_keywords": ["refund"],
        "expect_empty": False,
    },
]


def run_single(tc: dict, verbose: bool = False) -> tuple[bool, str]:
    """Run one test case. Returns (passed, reason)."""
    try:
        result = asyncio.run(
            ask_async(
                question=tc["question"],
                country=tc["country"],
                language=tc["language"],
            )
        )
    except Exception as e:
        return False, f"Exception: {e}"

    citations = result.get("citations", [])
    answer = result.get("answer", "")
    citation_ids = {c["content_id"] for c in citations}

    if tc["expect_empty"]:
        if citations:
            return False, (
                f"Expected empty citations but got: {list(citation_ids)[:3]}"
            )
        return True, "OK (empty as expected)"

    # Check at least one expected citation ID is present
    if tc["expected_citation_ids"]:
        found = citation_ids & set(tc["expected_citation_ids"])
        if not found:
            return False, (
                f"No expected citation found. Got: {list(citation_ids)}. "
                f"Expected one of: {tc['expected_citation_ids']}"
            )

    # Check all expected keywords appear in answer
    answer_lower = answer.lower()
    missing_kw = [
        kw for kw in tc["expected_keywords"] if kw.lower() not in answer_lower
    ]
    if missing_kw:
        return False, f"Missing keywords in answer: {missing_kw}"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print answers")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = []
    passed = 0

    print("=" * 70)
    print("EVALUATION HARNESS — Multi-Country Content Q&A")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['description']}")
        ok, reason = run_single(tc, verbose=args.verbose)
        status = "PASS" if ok else "FAIL"
        print(f"  Status : {status}")
        print(f"  Reason : {reason}")

        if args.verbose:
            try:
                result = asyncio.run(
                    ask_async(tc["question"], tc["country"], tc["language"])
                )
                print(f"  Answer : {result['answer'][:200]}")
                print(f"  Trace  : {result['trace']}")
            except Exception:
                pass

        if ok:
            passed += 1

        results.append({"id": tc["id"], "passed": ok, "reason": reason})

    total = len(TEST_CASES)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} passed")
    print("=" * 70)

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
