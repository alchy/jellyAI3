"""Zápisový etalon — normativní test PARSOVÁNÍ výroků Mnemos (BACKLOG #37).

`.venv/bin/python benchmark/run_mnemos.py`

Dotazová strana má guardrail čtyřnásobný (etalon/focus/dialog/coverage);
ZÁPISOVÁ strana (Mnemos: výrok uživatele → fakt paměti) selhala v provozu
bez měření (#31–35). Tento benchmark ji měří na TÉMŽE soukolí jako runtime:
`parse_clauses` s balíčkem karet a vetem `_known_word` živého automatu,
fixní hodiny (pátek 17. července 2026, poledne — jako dialog benchmark).
Souvětí = fakt na klauzuli (#46 fáze 4 v2): řádkové klíče měří PRVNÍ
parse, klíč `clauses` pak počet i obsah všech klauzulí.

Sada `mnemos.jsonl`: řádek = {u, kind|null, predicate?, objects?(podmnožina),
reject_objects?, places?, subject?, needs_subject?, time?, nom_*?, cat, gap?}.
`kind: null` = výrok NEMÁ být rozpoznán (dotaz, holá entita). Predikát se
srovnává bez ohledu na velikost písmen (kapitalizace začátku věty není chyba
zápisu); objekty PŘESNÝM tvarem (Pavel ≠ Pavla). „gap" = známý nedostatek
(tracking, nemaže se); když projde, hlásí se GAP-FIXED (pokrok).

`--nom` navíc prožene výrok nominativizací (`_nominativize_statement`) a
zkontroluje `nom_objects`/`nom_places`/`nom_subject` — VYŽADUJE běžící ÚFAL
služby (morpho/UDPipe); bez nich se nom očekávání přeskočí a jen započítají.
"""
import argparse
import json
import os

from datetime import datetime

from config import Config
from jellyai.iris import IrisAutomaton
from jellyai.iris.subsystems.mnemos import parse_clauses
from jellyai.tasks import make_graph_answerer

MNEMOS = os.path.join(os.path.dirname(__file__), "mnemos.jsonl")

ALLOWED_KEYS = {"u", "kind", "predicate", "objects", "reject_objects",
                "places", "subject", "needs_subject", "time",
                "nom_objects", "nom_places", "nom_subject", "cat", "gap",
                "note", "clauses"}

# klíče pod-řádku v `clauses` (souvětí = fakt na klauzuli, #46 fáze 4 v2)
CLAUSE_KEYS = {"kind", "predicate", "objects", "reject_objects",
               "places", "subject", "needs_subject", "time"}


def _now():
    """Fixní „teď" benchmarku — pátek 17. července 2026, poledne."""
    return datetime(2026, 7, 17, 12, 0)


def load_items(path):
    """Načte a ZVALIDUJE sadu: povinné `u`, jen povolené klíče (překlep
    v klíči by jinak tiše prošel jako splněné očekávání)."""
    with open(path, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    for item in items:
        unknown = set(item) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"neznámé klíče {sorted(unknown)}: {item}")
        if "u" not in item:
            raise ValueError(f"řádek bez výroku `u`: {item}")
        for sub in item.get("clauses", ()):
            unknown = set(sub) - CLAUSE_KEYS
            if unknown:
                raise ValueError(
                    f"neznámé klíče klauzule {sorted(unknown)}: {item}")
    return items


def row_ok(statement, item):
    """Porovná parse s očekáváními řádku.

    Returns:
        tuple: (bool, str) — splněno + důvod prvního nesouladu („" při shodě).
    """
    if statement is None:
        if item.get("kind", "") is None:
            return True, ""
        return False, "parse vrátil None"
    if "kind" in item and item["kind"] != statement.get("kind"):
        return False, f"kind {statement.get('kind')} ≠ {item['kind']}"
    if "predicate" in item and (statement.get("predicate") or "").lower() \
            != item["predicate"].lower():
        return False, f"predicate {statement.get('predicate')!r}"
    objects = statement.get("objects", ())
    missing = [o for o in item.get("objects", ()) if o not in objects]
    if missing:
        return False, f"objects bez {missing}"
    hit = [o for o in item.get("reject_objects", ()) if o in objects]
    if hit:
        return False, f"reject_objects zasaženo {hit}"
    places = statement.get("places", ())
    missing = [p for p in item.get("places", ()) if p not in places]
    if missing:
        return False, f"places bez {missing}"
    if "subject" in item and statement.get("subject") != item["subject"]:
        return False, f"subject {statement.get('subject')!r}"
    if "needs_subject" in item \
            and bool(statement.get("needs_subject")) != item["needs_subject"]:
        return False, f"needs_subject {statement.get('needs_subject')}"
    if "time" in item and statement.get("time") != item["time"]:
        return False, f"time {statement.get('time')!r}"
    return True, ""


def clauses_ok(parses, item):
    """Očekávání `clauses`: počet parsů sedí a každá klauzule projde
    týmž checkerem jako řádek (`row_ok`).

    Returns:
        tuple: (bool, str) — splněno + důvod prvního nesouladu.
    """
    expected = item.get("clauses", ())
    if len(parses) != len(expected):
        return False, f"klauzulí {len(parses)} ≠ {len(expected)}"
    for i, (parsed, sub) in enumerate(zip(parses, expected), 1):
        ok, why = row_ok(parsed, {"u": item["u"], **sub})
        if not ok:
            return False, f"klauzule {i}: {why}"
    return True, ""


def _nom_item(item):
    """Nom očekávání řádku přeložená na běžné klíče checkeru."""
    mapped = {"u": item["u"]}
    for src, dst in (("nom_objects", "objects"), ("nom_places", "places"),
                     ("nom_subject", "subject")):
        if src in item:
            mapped[dst] = item[src]
    return mapped if len(mapped) > 1 else None


def main():
    """Projede sadu proti živému soukolí Mnemos a vypíše PASS/FAIL/gap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nom", action="store_true",
                        help="měř i nominativizaci (běžící ÚFAL služby)")
    args = parser.parse_args()
    items = load_items(MNEMOS)
    iris = IrisAutomaton(make_graph_answerer(Config()), clock=_now)
    rows, passed, failed, gap_open, gap_fixed, nom_skipped = [], 0, 0, 0, 0, 0
    for item in items:
        parses = parse_clauses(item["u"], _now(), iris.deck,
                               is_node=iris._known_word)  # pylint: disable=protected-access
        statement = parses[0] if parses else None
        ok, why = row_ok(statement, item)
        if ok and "clauses" in item:
            ok, why = clauses_ok(parses, item)
        nom = _nom_item(item)
        nom_measured = True
        if nom is not None and not args.nom:
            nom_measured = False
            nom_skipped += 1
        elif nom is not None and ok and statement is not None:
            nominativized = iris._nominativize_statement(item["u"], statement)  # pylint: disable=protected-access
            ok, why = row_ok(nominativized, nom)
            why = why and f"nom: {why}"
        if "gap" in item:
            # gap se uzavírá jen PLNĚ změřený (nom očekávání bez --nom nestačí)
            fixed = ok and nom_measured
            status = "GAP-FIXED ✅" if fixed else "gap"
            gap_fixed += fixed
            gap_open += not fixed
        else:
            status = "PASS" if ok else "FAIL ✗"
            passed += ok
            failed += not ok
        got = "—" if statement is None else (
            f"{statement['predicate']}: {', '.join(statement['objects'])}")
        rows.append((status, item["u"], why or got))

    for status, utterance, detail in rows:
        print(f"[{status:12}] {utterance:42} → {detail[:46]}")
    core = passed + failed
    pct = 100 * passed // core if core else 0
    skipped = f"   nom neměřeno: {nom_skipped}" if nom_skipped else ""
    print(f"\nZÁPIS: {passed}/{core} ({pct} %)   "
          f"GAP: {gap_fixed} opraveno / {gap_open} otevřeno{skipped}")
    return passed, failed, gap_fixed, gap_open


if __name__ == "__main__":
    main()
