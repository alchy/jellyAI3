"""Vzorové karty dotazů (#46 fáze 2a) — zavírají #44 z živého dialogu.

„Kdo bydlí v Petrovicích?" dosud neměl šablonu (build_query → None) a
fakty paměti (osoba jako obj, místo loc, uživatel theme) nenašel ani
UDPipe fallback. Vzorová karta `q-kdo-sloveso-misto` staví Pattern
z tříd lexeru; answerer navíc dostal guardy: pozorovatel (uživatel
v roli theme) ani časová kotva nejsou odpověď na kdo/co díru.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _bydli(person, place):
    return make_fact("bydlí", [
        Participant("obj", person, "concept"),
        Participant("loc", place, "geo"),
        Participant("theme", "uživatel", "person"),
        Participant("time", "17. července 2026 12:00", "time")])


def _answerer(*facts):
    g = FactGraph()
    for fact in facts:
        g.add_fact(fact)
    return GraphAnswerer(g, FakeUfalClient(),
                         ExtractiveAnswerer(AnswererConfig()),
                         query_mode="templates", clock=lambda: NOW)


def test_kdo_verb_place_finds_inhabitants_not_user():
    a = _answerer(_bydli("Marcela", "Petrovice"), _bydli("Roník", "Petrovice"))
    text = a.answer("Kdo bydlí v Petrovicích?", []).text
    assert "Marcela" in text and "Roník" in text
    assert "uživatel" not in text         # pozorovatel není odpověď (#34)
    assert "července" not in text         # časová kotva není odpověď


def test_kdo_dalsi_variant_matches_same_pattern():
    a = _answerer(_bydli("Marcela", "Petrovice"))
    text = a.answer("Kdo další bydlí v Petrovicích?", []).text
    assert "Marcela" in text


def test_cim_instrument_question():
    a = _answerer(make_fact("krmí", [
        Participant("obj", "Marcela", "concept"),
        Participant("obj", "Roníka", "concept"),
        Participant("obj", "konzervami", "concept"),
        Participant("theme", "uživatel", "person"),
        Participant("time", "17. července 2026 12:00", "time")]))
    text = a.answer("Čím krmí Marcela Roníka?", []).text
    assert "konzervami" in text
    assert "uživatel" not in text and "července" not in text


def test_place_hole_still_works_after_guards():
    """Guardy nesmí rozbít směr „Kde bydlí X?" (loc díra z týchž faktů)."""
    a = _answerer(_bydli("Marcela", "Petrovice"))
    assert a.answer("Kde bydlí Marcela?", []).text == "Petrovice"
