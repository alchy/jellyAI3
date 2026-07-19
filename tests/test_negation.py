"""Negace dějů (BACKLOG #24 jádro) — negovaný fakt je evidence OPAKU.

„Prší?" s faktem `neprší(čas T)` → „Ne, od T neprší." Rozhoduje NEJNOVĚJŠÍ
evidence: nedatovaný (korpusový) fakt prohrává s datovanou pamětí; remíza
časů → pozdější zápis (uživatel se opravil). Pár predikát↔negace je
mechanismus (`negation_prefix` z jazykových dat), text odpovědi šablona
`negative_existence_answer` v cs.json — nikoli řetězec v kódu.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _event(predicate, time=None):
    parts = [Participant("obj", "Venku", "concept"),
             Participant("theme", "uživatel", "person")]
    if time is not None:
        parts.append(Participant("time", time, "time"))
    return make_fact(predicate, parts)


def _answerer(*facts):
    g = FactGraph()
    for fact in facts:
        g.add_fact(fact)
    return GraphAnswerer(g, FakeUfalClient(),
                         ExtractiveAnswerer(AnswererConfig()),
                         query_mode="templates", clock=lambda: NOW)


def test_negated_fact_answers_no_with_since():
    a = _answerer(_event("neprší", "17. července 2026 11:00"))
    out = a.answer("Prší?", [])
    assert out.text.startswith("Ne")
    assert "neprší" in out.text
    assert "17. července 2026 11:00" in out.text


def test_latest_evidence_wins_negative():
    a = _answerer(_event("prší", "16. července 2026"),
                  _event("neprší", "17. července 2026 11:00"))
    assert a.answer("Prší?", []).text.startswith("Ne")


def test_latest_evidence_wins_positive():
    """Po „neprší" začalo znovu pršet — poslední slovo má poslední evidence."""
    a = _answerer(_event("neprší", "16. července 2026"),
                  _event("prší", "17. července 2026 11:00"))
    assert a.answer("Prší?", []).text == "Ano"


def test_dated_negative_beats_undated_positive():
    """Nedatovaný korpusový fakt je starší než jakákoli datovaná paměť."""
    a = _answerer(_event("prší"),
                  _event("neprší", "17. července 2026 11:00"))
    assert a.answer("Prší?", []).text.startswith("Ne")


def test_same_moment_later_statement_wins():
    """Fixní hodiny dají oběma výrokům týž timestamp — rozhodne pořadí
    zápisu (deník je chronologický)."""
    a = _answerer(_event("prší", "17. července 2026 12:00"),
                  _event("neprší", "17. července 2026 12:00"))
    assert a.answer("Prší?", []).text.startswith("Ne")


def test_positive_only_still_answers_yes():
    a = _answerer(_event("prší", "17. července 2026 11:00"))
    assert a.answer("Prší?", []).text == "Ano"
