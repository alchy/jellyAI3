"""Klauzulátor v2 (#46 fáze 4): souvětí = fakt na klauzuli.

Zadání user (2026-07-19 + docs/JAK-PSAT-FAKTA.md „jedna věta = jeden
fakt"): „Roník jí stravu, má však rád i maso." jsou DVA fakty. Rozpad
na klauzule vyhrává jen když KAŽDÁ klauzule parsuje čistě (zdroj
predikátu malými písmeny — kapitalizovaný l-lookalike „Pavla" by z
výčtu udělal paskvil); jinak platí celek. Klauzule bez podmětu dědí
podmět předchozí (spec §3).
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.subsystems.mnemos import parse_clauses
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_compound_sentence_yields_fact_per_clause():
    parses = parse_clauses(
        "Roník jí i vegetariánskou stravu, má však rád i maso.", NOW, None)
    assert len(parses) == 2
    first, second = parses
    assert first["predicate"] == "jí"
    assert first["objects"] == ["Roník", "vegetariánskou", "stravu"]
    assert second["predicate"] == "má"
    assert second["objects"] == ["Roník", "rád", "maso"]  # podmět zděděn


def test_enumeration_comma_does_not_split():
    """Výčet „Potkal jsem Karla, Pavla a Marii." zůstává JEDNÍM faktem —
    „Pavla" je kapitalizovaný l-lookalike, klauzulí být nesmí."""
    parses = parse_clauses("Potkal jsem Karla, Pavla a Marii.", NOW, None)
    assert len(parses) == 1
    assert parses[0]["predicate"] == "potkal"
    assert {"Karla", "Pavla", "Marii"} <= set(parses[0]["objects"])


def test_confirmation_comma_keeps_whole_sentence():
    """„Ano, měl rád knedlíky." není souvětí — „Ano." neparsuje, platí
    celek (připsání osobě řeší automat, ne klauzulátor)."""
    parses = parse_clauses("Ano, měl rád knedlíky.", NOW, None)
    assert len(parses) == 1
    assert parses[0]["kind"] == "attributed"
    assert parses[0]["needs_subject"] is True


def test_vsuvka_definition_parses_clean():
    """Vsuvka „To co Roník jí je strava." (spec §3, gap #45): definiční
    tvar s klauzulí uvnitř — predikát je sloveso vnitřní klauzule (jí),
    účastníci Roník + strava; vztažné „co" ani spona v objektech nejsou."""
    parses = parse_clauses("To co Roník jí je strava.", NOW, None)
    assert len(parses) == 1
    assert parses[0]["predicate"] == "jí"
    assert parses[0]["objects"] == ["Roník", "strava"]


def test_final_copula_homograph_stays_participant():
    """Gap #45 byt≡být: „V Plzni má pronajatý byt." — koncové „byt"
    sponou být nemůže (spona stojí UPROSTŘED, spec §5); s predikátem
    ze vzoru (má) je to účastník a místo drží roli loc."""
    from jellyai.iris.subsystems.mnemos import parse_statement

    p = parse_statement("V Plzni má pronajatý byt.", NOW, None)
    assert p is not None and p["kind"] == "event"
    assert p["predicate"] == "má"
    assert "byt" in p["objects"] and "má" not in p["objects"]
    assert p["places"] == ["Plzni"]


def test_medial_copula_still_dropped_from_objects():
    """Pravidlo je POZIČNÍ: spona uprostřed („Niki je však většinou
    v Plzni.") účastníkem nezůstává — observation drží."""
    from jellyai.iris.subsystems.mnemos import parse_statement

    p = parse_statement("Niki je však většinou v Plzni.", NOW, None)
    assert p is not None and p["kind"] == "observation"
    assert "je" not in p["objects"]


def test_multi_memorize_stores_both_clauses():
    """Automat uloží OBA fakty souvětí, potvrzení je vyjmenuje a druhá
    klauzule je dotazatelná („Co má Roník?" → maso)."""
    g = FactGraph()
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()))
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn(
        "Roník jí i vegetariánskou stravu, má však rád i maso.")
    assert "Zapamatováno" in stored.text
    assert "jí" in stored.text and "má" in stored.text
    assert isinstance(stored.memorized, list) and len(stored.memorized) == 2
    out = iris.turn("Co má Roník?")
    assert "maso" in out.text
