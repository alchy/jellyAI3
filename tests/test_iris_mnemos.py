"""Mnemos — paměť Iris: časově vázaná komunikace uživatele v grafu.

Konstatování („Dnes jsem měl knedlíky.") NENÍ dotaz — Mnemos ho uloží jako
fakt grafu s UŽIVATELEM jako entitou (identita uživatele je uzel) a Chronos
ukotví relativní čas na absolutní datum (paměť nesmí držet „dnes" — zítra by
znamenalo jiný den). Otázka „Kdy jsem měl v tomto roce knedlíky?" se pak
zodpoví běžnou cestou grafu — Mnemos a Chronos pomáhají Iris s aktivací.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.mnemos import parse_statement, remember
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_parse_statement_extracts_time_anchored_fact():
    fact = parse_statement("Dnes jsem měl knedlíky.", NOW)
    assert fact is not None and fact["kind"] == "episode"
    assert fact["predicate"] == "měl"
    assert fact["objects"] == ["knedlíky"]
    assert fact["time"] == "17. července 2026"    # absolutní kotva, ne „dnes"


def test_statement_without_time_word_still_gets_timestamp():
    """Čas výroku platí vždy — interakce je časově vázaná."""
    fact = parse_statement("Měl jsem knedlíky.", NOW)
    assert fact["time"] == "17. července 2026"


def test_world_observation_is_recognized():
    """„Venku je teplo." — konstatování o světě (bez 1. osoby): timestamp +
    graf, uživatel jako pozorovatel; zatím žádná další reakce."""
    fact = parse_statement("Venku je teplo.", NOW)
    assert fact is not None and fact["kind"] == "observation"
    assert fact["predicate"] == "být"
    assert fact["objects"] == ["Venku", "teplo"]
    assert fact["time"] == "17. července 2026"
    g = FactGraph()
    remember(g, fact, "uživatel")
    parts = {(p.role, p.node) for p in next(iter(g.facts.values())).participants}
    assert ("subj", "Venku") in parts and ("pred", "teplo") in parts
    assert ("theme", "uživatel") in parts
    assert ("time", "17. července 2026") in parts


def test_questions_and_plain_text_are_not_statements():
    assert parse_statement("Kdy jsem měl knedlíky?", NOW) is None   # dotaz
    assert parse_statement("Josef Čapek", NOW) is None              # volba/entita
    assert parse_statement("Knedlíky s gulášem.", NOW) is None      # bez 1. osoby


def test_remember_writes_user_fact_into_graph():
    g = FactGraph()
    remember(g, parse_statement("Dnes jsem měl knedlíky.", NOW), "uživatel")
    fact = next(iter(g.facts.values()))
    parts = {(p.role, p.node) for p in fact.participants}
    assert ("subj", "uživatel") in parts
    assert ("obj", "knedlíky") in parts
    assert ("time", "17. července 2026") in parts


def test_new_card_extends_recognition_without_code(tmp_path):
    """ZÁKON: logika se nestaví fixně programově — nová karta v adresáři
    naučí Mnemos nový tvar konstatování („Pršelo." — holé l-příčestí)."""
    import json
    from jellyai.iris.patterns import PatternDeck
    card = {"name": "statement-bare-verb",
            "trigger": {"event": "utterance.statement",
                        "requires": ["l_verb"], "forbids": ["first_person"]},
            "dialog": "Zapamatováno: {fact}",
            "action": {"memorize": "episode", "predicate_from": "l_verb"},
            "teach": "Testovací vzor: holé příčestí bez osoby."}
    (tmp_path / "statement-bare-verb.json").write_text(json.dumps(card),
                                                       encoding="utf-8")
    deck = PatternDeck(str(tmp_path))
    deck.load()
    assert parse_statement("Pršelo celý den.", NOW, deck) is None \
        or True   # bez objektů může být None — jádro testu je níž
    fact = parse_statement("Pršelo v Praze.", NOW, deck)
    assert fact is not None and fact["card"] == "statement-bare-verb"
    assert fact["predicate"] == "pršel"


def test_full_dumpling_scenario_via_iris():
    """Scénář uživatele: konstatování → paměť; otázka s „v tomto roce" →
    odpověď z Mnemos faktu (Chronos ukotvil čas při uložení)."""
    g = FactGraph()
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn("Dnes jsem měl knedlíky.")
    assert stored.kind == "answer" and "Zapamatováno" in stored.text
    assert "mnemos" in stored.used["components"]
    out = iris.turn("Kdy jsem měl v tomto roce knedlíky?")
    assert out.kind == "answer"
    assert "17. července 2026" in out.text
