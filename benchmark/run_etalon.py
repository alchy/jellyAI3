"""Etalon — normativní test kvality odpovědí. Spusť před/po změně a porovnej.

`.venv/bin/python benchmark/run_etalon.py`

Sada `etalon.jsonl`: {q, expect:[přijatelné podřetězce], reject?:[zakázané
podřetězce], cat, gap?}. Skóre = úspěšnost na JÁDRU (bez „gap" dotazů). „gap"
jsou známé neúspěchy (tracking); když gap začne procházet, je to POKROK
(GAP-FIXED). Shoda je case-insensitive podřetězec; `reject` hlídá, že odpověď
NEobsahuje šum/hádání (negativní očekávání — poctivost > recall).
"""
import json
import os

from config import Config
from jellyai.tasks import make_graph_answerer

ETALON = os.path.join(os.path.dirname(__file__), "etalon.jsonl")


def _matches(answer, expect, reject=()):
    """True, když odpověď obsahuje některý očekávaný a žádný zakázaný podřetězec."""
    low = answer.lower()
    return (any(e.lower() in low for e in expect)
            and not any(r.lower() in low for r in reject))


def main():
    with open(ETALON, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    answerer = make_graph_answerer(Config())
    rows, passed, failed, gap_open, gap_fixed = [], 0, 0, 0, 0
    for item in items:
        answerer.reset()
        answer = answerer.answer(item["q"], []).text
        ok = _matches(answer, item["expect"], item.get("reject", ()))
        if "gap" in item:
            status = "GAP-FIXED ✅" if ok else "gap"
            gap_fixed += ok
            gap_open += not ok
        else:
            status = "PASS" if ok else "FAIL ✗"
            passed += ok
            failed += not ok
        rows.append((status, item["q"], answer))

    for status, question, answer in rows:
        print(f"[{status:12}] {question:36} → {answer[:42]}")
    core = passed + failed
    pct = 100 * passed // core if core else 0
    print(f"\nJÁDRO: {passed}/{core} ({pct} %)   "
          f"GAP: {gap_fixed} opraveno / {gap_open} otevřeno")
    return passed, failed, gap_fixed, gap_open


if __name__ == "__main__":
    main()
