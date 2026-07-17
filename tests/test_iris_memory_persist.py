"""Perzistence Mnemos — paměť uživatele přežije restart služby.

Konstatování se kromě grafu v paměti PŘIPÍŠE do `memory.jsonl` (auditovatelný
deník); nový automat nad čerstvě načteným grafem deník přehraje a paměť
obnoví — včetně slovníku predikátů pro parser. Deník přežije i přestavbu
korpusového grafu (fakta uživatele nejsou v anotacích).
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _iris(memory_path):
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    return IrisAutomaton(answerer, clock=lambda: NOW,
                         memory_path=memory_path)


def test_memory_survives_restart(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    first = _iris(path)
    stored = first.turn("Dnes jsem měl knedlíky.")
    assert "Zapamatováno" in stored.text
    # „restart": nový automat, čerstvý (prázdný) graf, týž deník
    reborn = _iris(path)
    out = reborn.turn("Kdy jsem měl v tomto roce knedlíky?")
    assert out.kind == "answer"
    assert "17. července 2026" in out.text


def test_no_memory_path_keeps_memory_volatile(tmp_path):
    first = _iris(None)
    first.turn("Dnes jsem měl knedlíky.")
    reborn = _iris(None)
    out = reborn.turn("Kdy jsem měl v tomto roce knedlíky?")
    assert "17. července 2026" not in out.text
