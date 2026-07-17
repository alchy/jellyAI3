"""Topos — kontejnment míst (S3): gazetteer + místní filtr odpovědí.

Zrcadlo Chronosu na ose prostoru: „Pršelo v Čechách?" s faktem
prší(… Praze …) → Ano, protože Praha ⊂ Čechy (gazetteer); „na Moravě"
poctivě nenajde. Jména se porovnávají kmenově (v Praze ↔ Praha).
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.subsystems.topos import (area_keys, load_gazetteer,
                                           place_within)
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)
GAZ = load_gazetteer("data/sub_topos_gazetteer.jsonl")


def test_gazetteer_containment_transitive_and_inflected():
    """Kontejnment jde po řetězu rodičů a snese skloněný povrch."""
    assert place_within("Praze", "Čechy", GAZ)          # přímý rodič
    assert place_within("Praha", "Česko", GAZ)          # tranzitivně
    assert place_within("Kafarnaum", "Izrael", GAZ)     # Galilea → Izrael
    assert place_within("Čechách", "Čechy", GAZ)        # oblast sama v sobě
    assert not place_within("Brno", "Čechy", GAZ)       # Morava ≠ Čechy
    assert "cech" in {k[:4] for k in area_keys(GAZ)} or area_keys(GAZ)


def test_rain_in_prague_answers_by_containment():
    """E2E zadavatele: „V Praze prší." → „Pršelo v Čechách?" → Ano
    (Praha ⊂ Čechy); „na Moravě?" → poctivé nenašel — oblast je FILTR
    Toposu, ne účastník existence."""
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates", clock=lambda: NOW)
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn("V Praze prší.")
    assert "Zapamatováno" in stored.text
    assert iris.turn("Pršelo v Praze?").text == "Ano"
    assert iris.turn("Pršelo v Čechách?").text == "Ano"
    out = iris.turn("Pršelo na Moravě?")
    assert out.text != "Ano"
