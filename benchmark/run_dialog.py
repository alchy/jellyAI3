"""Dialog — normativní test AUTOMATU IRIS. Spusť před/po změně a porovnej.

`.venv/bin/python benchmark/run_dialog.py`

Sada `dialog.jsonl`: řádek = scénář {name, turns:[{u, expect:[povinné
podřetězce], kind?, reject?}…]}. Scénář je souvislý rozhovor: automat se mezi
tahy NErestartuje (zaostření a rozpracovaný dialog nesou kontext dál); mezi
scénáři se staví ČERSTVÝ automat s čerstvě načteným grafem (fakta Mnemos
z minulého scénáře nesmí prosakovat). Hodiny jsou fixní (pátek 17. července
2026, poledne), aby Chronos i Mnemos odpovídaly deterministicky.

Tah PASS, když odpověď obsahuje VŠECHNY expect podřetězce (case-insensitive),
ŽÁDNÝ reject a druh tahu (answer/dialog) sedí — je-li předepsán. Baseline je
100 % (normativ): pokles znamená regresi dialogového chování.
"""
import json
import os

from datetime import datetime

from config import Config
from jellyai.iris import IrisAutomaton
from jellyai.tasks import make_graph_answerer

DIALOG = os.path.join(os.path.dirname(__file__), "dialog.jsonl")


def _now():
    """Fixní „teď" benchmarku — pátek 17. července 2026, poledne."""
    return datetime(2026, 7, 17, 12, 0)


def _turn_ok(response, turn):
    """True, když odpověď tahu splní expect + reject + kind (je-li uveden)."""
    low = response.text.lower()
    return (all(e.lower() in low for e in turn["expect"])
            and not any(r.lower() in low for r in turn.get("reject", ()))
            and turn.get("kind", response.kind) == response.kind)


def main():
    """Projede scénáře proti aktuálnímu grafu a vypíše PASS/FAIL + souhrn."""
    with open(DIALOG, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    config = Config()
    rows, passed, total, scen_passed = [], 0, 0, 0
    for scenario in scenarios:
        # čerstvý automat = čistý stav dialogu i grafu (bez cizích faktů)
        iris = IrisAutomaton(make_graph_answerer(config), clock=_now)
        clean = True
        for turn in scenario["turns"]:
            response = iris.turn(turn["u"])
            ok = _turn_ok(response, turn)
            passed += ok
            total += 1
            clean &= ok
            rows.append(("PASS" if ok else "FAIL ✗", scenario["name"],
                         turn["u"], response.text))
        scen_passed += clean

    for status, name, question, answer in rows:
        print(f"[{status:6}] {name}/{question:36} → {answer[:42]}")
    pct = 100 * passed // total if total else 0
    print(f"\nDIALOG: {passed}/{total} tahů ({pct} %), "
          f"scénářů {scen_passed}/{len(scenarios)}")
    return passed, total


if __name__ == "__main__":
    main()
