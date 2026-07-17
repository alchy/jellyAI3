"""Focus — normativní test ZAOSTŘENÍ AKTIVACE. Spusť před/po změně a porovnej.

`.venv/bin/python benchmark/run_focus.py`

Po každé odpovědi drží answerer konverzační aktivační pole
(`answerer.context.scores` — dict {uzel: jas}). Benchmark měří, jestli je pole
správně ZAOSTŘENÉ: uzly, které k otázce právem patří (téma + odpověď, např.
„Božena Němcová" a „Babička"), musí být mezi top-K nejjasnějšími uzly. Když
je pole rozostřené (očekávané uzly přesvítí šum), řádek padá — i kdyby textová
odpověď náhodou vyšla správně.

Sada `focus.jsonl`: {q, expect_nodes:[id uzlů], k}. Řádek PASS, když VŠECHNY
expect_nodes leží v top-K uzlů seřazených podle jasu sestupně. Baseline je
100 % (normativ) — jde o guardrail: pokles znamená regresi zaostření.
"""
import json
import os

from config import Config
from jellyai.tasks import make_graph_answerer

FOCUS = os.path.join(os.path.dirname(__file__), "focus.jsonl")


def _run_item(answerer, item):
    """Zodpoví položku; {"dialog": [q1, q2…]} je navazující rozhovor (bez
    resetu mezi tahy) — hodnotí se aktivační pole po posledním tahu."""
    questions = item.get("dialog") or [item["q"]]
    for question in questions:
        answerer.answer(question, [])
    item.setdefault("q", " → ".join(questions))


def main():
    """Projede focus sadu proti aktuálnímu grafu a vypíše PASS/FAIL + skóre."""
    with open(FOCUS, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    answerer = make_graph_answerer(Config())
    rows, passed = [], 0
    for item in items:
        answerer.reset()
        _run_item(answerer, item)
        k = item.get("k", 5)
        top = sorted(answerer.context.scores,
                     key=answerer.context.scores.get, reverse=True)[:k]
        missing = [node for node in item["expect_nodes"] if node not in top]
        passed += not missing
        rows.append(("PASS" if not missing else "FAIL ✗", item["q"], missing))

    for status, question, missing in rows:
        detail = f" → chybí: {', '.join(missing)}" if missing else ""
        print(f"[{status:6}] {question}{detail}")
    total = len(rows)
    pct = 100 * passed // total if total else 0
    print(f"\nFOCUS: {passed}/{total} ({pct} %)")
    return passed, total


if __name__ == "__main__":
    main()
