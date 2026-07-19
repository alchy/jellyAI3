"""Triage telemetrie (BACKLOG #38) — stopa tahu do JSONL + shlukování.

Provoz plní zápisový/dotazový etalon průběžně: každý tah zapíše
strukturovanou stopu (otázka → karty → odpověď + assurance) a
`./jelly triage` shlukuje tahy s miss/nízkou assurance — místo
ručního lovení pastí ve screenshotech.
"""

import json
from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.triage import clusters, is_miss, report
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _iris(telemetry_path=None):
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()))
    return IrisAutomaton(answerer, clock=lambda: NOW,
                         telemetry_path=telemetry_path)


def test_turn_writes_telemetry_row(tmp_path):
    path = str(tmp_path / "telemetry.jsonl")
    iris = _iris(telemetry_path=path)
    iris.turn("Kdo napsal R.U.R.?")
    iris.turn("Kdo napsal Hordubala?")        # poctivý miss
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(rows) == 2
    assert rows[0]["q"] == "Kdo napsal R.U.R.?"
    assert "Karel" in rows[0]["answer"]
    assert rows[0]["kind"] == "answer" and rows[0]["assurance"] > 0
    assert rows[0]["patterns"]                # vystřelené karty ve stopě
    assert rows[0]["ts"].startswith("2026-07-17")


def test_no_telemetry_without_path(tmp_path):
    """Benchmarky a testy automat staví BEZ telemetry_path — nic se nepíše."""
    iris = _iris()
    iris.turn("Kdo napsal R.U.R.?")
    assert not list(tmp_path.iterdir())


def test_miss_detection_and_clustering():
    rows = [
        {"q": "Kdo napsal R.U.R.?", "kind": "answer", "answer": "Karel Čapek",
         "assurance": 0.9, "patterns": ["q-otaz-minuly"]},
        {"q": "Kdo je Roník?", "kind": "answer",
         "answer": "Nepodařilo se mi zaostřit dotaz dostatečně.",
         "assurance": 0.2, "patterns": ["assurance-fail"]},
        {"q": "Mám dnes něco naplánováno?", "kind": "answer",
         "answer": "V textu jsem odpověď nenašel.", "assurance": 0.0,
         "patterns": []},
        {"q": "Máš dnes něco v plánu?", "kind": "answer",
         "answer": "V textu jsem odpověď nenašel.", "assurance": 0.0,
         "patterns": []},
    ]
    assert not is_miss(rows[0])
    assert all(is_miss(r) for r in rows[1:])
    shluky = clusters(rows)
    assert shluky[0]["count"] == 2            # největší shluk první
    assert "Mám dnes něco naplánováno?" in shluky[0]["examples"]
    text = report(rows)
    assert "2×" in text and "Roník" in text
