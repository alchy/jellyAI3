"""Připomínky — Chronos časování (krátkodobá paměť, vlastní vlákno hodin).

„Připomeň mi za deset minut vybrat maso z trouby" → termín (resolve_due),
úkol, sklad; dozrání odpálí tep Chronos NEBO začátek tahu. Bez termínu se
automat ptá (dialog > figly). Scénáře nesou karty reminder-set/when/due.
"""

from datetime import datetime, timedelta

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.chronos import ChronosTicker, format_due, resolve_due
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_resolve_due_offset_clock_date_and_advance():
    assert resolve_due("za deset minut vybrat maso", NOW) \
        == NOW + timedelta(minutes=10)
    assert resolve_due("za hodinu", NOW) == NOW + timedelta(hours=1)
    assert resolve_due("v 18:30 zavolat", NOW) == NOW.replace(hour=18,
                                                              minute=30)
    assert resolve_due("v 9:00", NOW) == NOW.replace(
        hour=9, minute=0) + timedelta(days=1)          # po termínu → zítra
    assert resolve_due("má Pavel svátek 21. června", NOW) \
        == datetime(2027, 6, 21, 9)                    # letos už bylo → napřesrok
    assert resolve_due("svátek 21. srpna den předem", NOW) \
        == datetime(2026, 8, 20, 9)                    # předstih
    assert resolve_due("zítra koupit rohlíky", NOW) == datetime(2026, 7, 18, 9)
    assert resolve_due("koupit rohlíky", NOW) is None  # bez času


def test_format_due_today_vs_date():
    assert format_due(NOW.replace(hour=12, minute=10), NOW) == "12:10"
    assert format_due(datetime(2026, 8, 20, 9), NOW) == "20. srpna 9:00"
    assert format_due(datetime(2027, 6, 21, 9), NOW) == "21. června 2027 9:00"


def _iris(clock, path=None):
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    return IrisAutomaton(answerer, clock=clock, reminders_path=path)


def test_reminder_set_and_fire_on_turn():
    """Naplánování s termínem → potvrzení; po dozrání se text připomínky
    předřadí odpovědi PŘÍŠTÍHO tahu (plynutí hodin bez tikeru)."""
    moment = [NOW]
    iris = _iris(lambda: moment[0])
    out = iris.turn("Připomeň mi za deset minut vybrat maso z trouby.")
    assert "Připomenu 12:10" in out.text and "vybrat maso z trouby" in out.text
    assert "reminder-set" in out.used["patterns"]
    moment[0] = NOW + timedelta(minutes=11)
    out = iris.turn("Kolik je hodin?")
    assert "Připomínám: vybrat maso z trouby" in out.text
    assert "12:11" in out.text                       # odpověď tahu následuje
    assert not iris.reminders                        # krátkodobá paměť: pryč


def test_reminder_without_time_asks_then_completes():
    """Bez termínu se automat ZEPTÁ (karta reminder-when); další odpověď
    („za hodinu") termín doplní a připomínka se naplánuje."""
    iris = _iris(lambda: NOW)
    out = iris.turn("Připomeň mi, že mám koupit rohlíky.")
    assert out.kind == "dialog" and "koupit rohlíky" in out.text
    assert "reminder-when" in out.used["patterns"]
    out = iris.turn("za hodinu")
    assert "Připomenu 13:00" in out.text and "koupit rohlíky" in out.text


def test_reminder_persists_across_instances(tmp_path):
    """Sklad JSONL přežije restart — nová instance odpálí, co dozrálo."""
    path = str(tmp_path / "reminders.jsonl")
    _iris(lambda: NOW, path).turn("Připomeň mi za pět minut čaj.")
    later = _iris(lambda: NOW + timedelta(minutes=6), path)
    fired = later.fire_due()
    assert fired and "čaj" in fired[0]
    assert later.fire_due() == []                    # jednou a dost


def test_chronos_ticker_beats_and_notifies():
    """Tep Chronos (vlastní vlákno hodin): fire → notify; chyba notifikace
    časovač nezabije."""
    heard = []
    ticker = ChronosTicker(lambda: ["⏰ Připomínám: čaj"],
                           heard.append, interval=999)
    ticker.beat()
    assert heard == ["⏰ Připomínám: čaj"]
    boom = ChronosTicker(lambda: ["x"],
                         lambda m: (_ for _ in ()).throw(RuntimeError()),
                         interval=999)
    boom.beat()                                      # nesmí vyhodit
