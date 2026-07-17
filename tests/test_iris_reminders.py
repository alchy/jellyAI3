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
from jellyai.iris.subsystems.chronos import ChronosTicker, format_due, resolve_due
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


def test_resolve_due_soon_fractions_and_day_parts():
    """„za chvíli/čtvrt hodiny" (jazykové tabulky minut) a denní části:
    „zítra ráno" = 7:00, „ráno v šest" = 6:00, „večer v 8" = 20:00."""
    assert resolve_due("za chvíli", NOW) == NOW + timedelta(minutes=15)
    assert resolve_due("za čtvrt hodiny", NOW) == NOW + timedelta(minutes=15)
    assert resolve_due("za půl hodiny", NOW) == NOW + timedelta(minutes=30)
    assert resolve_due("až zítra ráno", NOW) == datetime(2026, 7, 18, 7)
    assert resolve_due("vzbuď mě ráno v šest", NOW) == datetime(2026, 7, 18, 6)
    assert resolve_due("večer v 8", NOW) == datetime(2026, 7, 17, 20)


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


def test_reminder_without_time_defaults_then_reschedules():
    """Bez termínu se rovnou plánuje VÝCHOZÍ ofset (karta reminder-default,
    za čtvrt hodiny) s nabídkou změny; navazující „až zítra ráno" připomínku
    PŘEPLÁNUJE — nevznikne druhá."""
    iris = _iris(lambda: NOW)
    out = iris.turn("Připomeň mi, že mám koupit rohlíky.")
    assert out.kind == "answer" and "koupit rohlíky" in out.text
    assert "12:15" in out.text and "reminder-default" in out.used["patterns"]
    assert len(iris.reminders) == 1
    out = iris.turn("až zítra ráno")
    assert "Připomenu 18. července 7:00" in out.text
    assert len(iris.reminders) == 1                  # přeplánováno, ne přidáno
    assert iris.reminders[0]["due"].startswith("2026-07-18T07:00")


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
