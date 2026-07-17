"""Tvrdý časový filtr (S2, backlog #10) — interval z otázky FILTRUJE fakty.

Dosud interval jen rozsvěcel časové uzly (soft ranking); teď otázka
s časovým primitivem VYŘADÍ fakty, jejichž časový účastník do intervalu
nespadá. Fakty bez časového účastníka filtr nechává (nedatované nelze
vyloučit časem). E2E případ zadavatele: kovářova kobyla, 21.1.1900 —
pozor, rok 1900 JE 19. století (1801–1900).
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.iris.subsystems.chronos import resolve_temporal
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_explicit_date_is_temporal_primitive():
    """„21.1.1900" i „21. ledna 1900" → denní interval (absolutní kotva)."""
    for text in ("21.1.1900 chodila kobyla", "21. ledna 1900 chodila kobyla"):
        interval = resolve_temporal(text, NOW)
        assert interval is not None and interval.granularity == "day"
        assert interval.start == datetime(1900, 1, 21)
        assert interval.contains_date({"rok": "1900", "měsíc": "leden",
                                       "den": "21"})


def _kobyla_answerer():
    g = FactGraph()
    g.add_fact(make_fact("chodit", [
        Participant("subj", "kovářova kobyla", "concept"),
        Participant("obj", "bosa", "concept"),
        Participant("time", "21. ledna 1900", "time")]))
    return GraphAnswerer(g, FakeUfalClient(),
                         ExtractiveAnswerer(AnswererConfig()),
                         query_mode="templates", clock=lambda: NOW)


def test_interval_hard_filters_existence():
    """Fakt z 21.1.1900: „v 19. století?" → Ano (1900 ⊂ 1801–1900);
    „ve 20. století?" → poctivé nenašel — filtr fakt VYŘADÍ."""
    a = _kobyla_answerer()
    out = a.answer("Chodila kovářova kobyla bosa v 19. století?", [])
    assert out.text == "Ano"
    a.reset()
    out = a.answer("Chodila kovářova kobyla bosa ve 20. století?", [])
    assert out.text != "Ano"


def test_undated_facts_survive_filter():
    """Fakt BEZ časového účastníka interval nevyřadí — nedatované
    nelze vyloučit časem."""
    g = FactGraph()
    g.add_fact(make_fact("chodit", [
        Participant("subj", "kovářova kobyla", "concept"),
        Participant("obj", "bosa", "concept")]))
    a = GraphAnswerer(g, FakeUfalClient(),
                      ExtractiveAnswerer(AnswererConfig()),
                      query_mode="templates", clock=lambda: NOW)
    out = a.answer("Chodila kovářova kobyla bosa v 19. století?", [])
    assert out.text == "Ano"


def test_dated_hole_answers_respect_filter():
    """I díra (kdy/kde…) čte jen fakty uvnitř intervalu: dva narodit-fakty
    různých let — otázka s „v 19. století" vybere ten z 1820."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [
        Participant("subj", "Božena", "person"),
        Participant("loc", "Vídeň", "geo"),
        Participant("time", "4. února 1820", "time")]))
    g.add_fact(make_fact("narodit", [
        Participant("subj", "Božena", "person"),
        Participant("loc", "Praha", "geo"),
        Participant("time", "4. února 1920", "time")]))
    a = GraphAnswerer(g, FakeUfalClient(),
                      ExtractiveAnswerer(AnswererConfig()),
                      query_mode="templates", clock=lambda: NOW)
    out = a.answer("Kde se narodila Božena v 19. století?", [])
    assert "Vídeň" in out.text and "Praha" not in out.text
