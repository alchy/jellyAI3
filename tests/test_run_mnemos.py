"""Zápisový etalon (BACKLOG #37) — kontrakt runneru `benchmark/run_mnemos.py`.

Benchmark měří ZÁPISOVOU cestu Mnemos (výrok → parse), která v provozu selhala
(#31–35), zatímco dotazová strana má guardrail čtyřnásobný. Testuje se čistý
checker řádku (`row_ok`) a validace schématu sady (`load_items`) — samotné
parsování měří benchmark, ne tento test.
"""

import json
import os

import pytest

from benchmark.run_mnemos import MNEMOS, clauses_ok, load_items, row_ok

STATEMENT = {"kind": "event", "predicate": "prší", "objects": ["Venku"],
             "places": [], "time": "17. července 2026 12:00",
             "card": "statement-event", "needs_subject": False}


def test_expected_none_matches_unrecognized_utterance():
    """`kind: null` = výrok NEMÁ být rozpoznán (dotaz, holá entita)."""
    ok, _ = row_ok(None, {"u": "Kdo je Karel?", "kind": None})
    assert ok
    ok, why = row_ok(STATEMENT, {"u": "Kdo je Karel?", "kind": None})
    assert not ok and "kind" in why


def test_expected_kind_fails_when_parse_returns_none():
    ok, why = row_ok(None, {"u": "Venku prší.", "kind": "event"})
    assert not ok and "None" in why


def test_kind_and_predicate_match():
    assert row_ok(STATEMENT, {"u": "…", "kind": "event", "predicate": "prší"})[0]
    ok, why = row_ok(STATEMENT, {"u": "…", "kind": "observation"})
    assert not ok and "kind" in why
    ok, why = row_ok(STATEMENT, {"u": "…", "predicate": "bydlet"})
    assert not ok and "predicate" in why


def test_predicate_comparison_is_case_insensitive():
    """Kapitalizace predikátu („Máme" na začátku věty) není chyba zápisu."""
    assert row_ok(STATEMENT, {"u": "…", "predicate": "Prší"})[0]


def test_objects_are_required_subset_and_reject_forbids():
    fact = dict(STATEMENT, objects=["Karel", "Praze"])
    assert row_ok(fact, {"u": "…", "objects": ["Karel"]})[0]
    ok, why = row_ok(fact, {"u": "…", "objects": ["Emil"]})
    assert not ok and "objects" in why
    ok, why = row_ok(fact, {"u": "…", "reject_objects": ["Praze"]})
    assert not ok and "reject" in why


def test_places_subject_time_needs_subject():
    fact = dict(STATEMENT, places=["Praze"], subject="Karel",
                needs_subject=True, time="17. července 2026")
    assert row_ok(fact, {"u": "…", "places": ["Praze"], "subject": "Karel",
                         "needs_subject": True,
                         "time": "17. července 2026"})[0]
    assert not row_ok(fact, {"u": "…", "places": ["Brně"]})[0]
    assert not row_ok(fact, {"u": "…", "subject": "uživatel"})[0]
    assert not row_ok(fact, {"u": "…", "time": "16. července 2026"})[0]
    assert not row_ok(fact, {"u": "…", "needs_subject": False})[0]


def test_clauses_expectations_check_each_parse():
    """Klíč `clauses` (#46 fáze 4 v2): souvětí = fakt na klauzuli — počet
    parsů musí sedět a každý se měří týmž checkerem jako řádek."""
    first = dict(STATEMENT, predicate="jí", objects=["Roník", "stravu"])
    second = dict(STATEMENT, predicate="má", objects=["Roník", "maso"])
    item = {"u": "…", "clauses": [
        {"predicate": "jí", "objects": ["Roník"]},
        {"predicate": "má", "objects": ["maso"], "reject_objects": ["jí"]}]}
    assert clauses_ok([first, second], item)[0]
    ok, why = clauses_ok([first], item)
    assert not ok and "klauzul" in why
    ok, why = clauses_ok([first, dict(second, objects=["jí"])], item)
    assert not ok


def test_load_items_rejects_unknown_keys(tmp_path):
    """Překlep v klíči řádku (`objets`) nesmí tiše projít jako splněný."""
    path = tmp_path / "rows.jsonl"
    path.write_text('{"u": "Venku prší.", "objets": ["Venku"]}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(str(path))
    path.write_text('{"kind": "event"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(str(path))    # chybí "u"


def test_shipped_rows_conform_to_schema():
    """Ostrá sada `mnemos.jsonl` existuje a validuje (žádné překlepy klíčů)."""
    assert os.path.exists(MNEMOS)
    items = load_items(MNEMOS)
    assert len(items) >= 30, "paleta má být široká (viz zadání #37)"
    assert any("gap" in item for item in items), "pasti #31–35 patří do sady"
    for item in items:
        json.dumps(item)     # serializovatelné = čisté JSONL
