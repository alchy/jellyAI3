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


def test_mnemos_place_roundtrip_with_containment():
    """Vklad přes Mnemos (zadání): „Marcela bydlí v Petrovicích." →
    místo dostane roli loc/geo → „Kde bydlí Marcela?" odpoví místem
    a „Bydlí Marcela v Čechách?" → Ano kontejnmentem (Petrovice ⊂
    Plzeň ⊂ Čechy); „na Moravě?" poctivě ne."""
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             clock=lambda: NOW)
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    assert "Zapamatováno" in iris.turn("Marcela bydlí v Petrovicích.").text
    assert "Petrovicích" in iris.turn("Kde bydlí Marcela?").text
    assert iris.turn("Bydlí Marcela v Čechách?").text == "Ano"
    assert iris.turn("Bydlí Marcela na Moravě?").text != "Ano"


def test_nested_places_teach_containment(tmp_path):
    """Zadání: „Zapamatuj si, že Pavla a Matěj bydlí na Barrandově
    v Praze." — dva podměty v JEDNOM faktu, obě místa role loc, a PŘESAH
    do Toposu: vnořená místa učí kontejnment (Barrandov ⊂ Praha) →
    „Bydlí Matěj v Česku?" → Ano přes naučený řetěz."""
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             clock=lambda: NOW)
    seed = tmp_path / "gaz.jsonl"                            # čistý tmp seed
    seed.write_text('{"place": "Praha", "in": "Čechy"}\n'
                    '{"place": "Čechy", "in": "Česko"}\n', encoding="utf-8")
    answerer._gazetteer_path = str(seed)
    answerer._gazetteer = load_gazetteer(str(seed))
    answerer._area_keys = area_keys(answerer._gazetteer)
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    out = iris.turn("Zapamatuj si, že Pavla a Matěj bydlí na Barrandově "
                    "v Praze.")
    assert "Zapamatováno" in out.text
    assert "Barrandově" in iris.turn("Kde bydlí Matěj?").text
    assert iris.turn("Bydlí Pavla v Praze?").text == "Ano"
    assert iris.turn("Bydlí Matěj v Česku?").text == "Ano"   # naučený řetěz
    learned = (tmp_path / "gaz.jsonl").read_text(encoding="utf-8")
    assert "Barrandově" in learned and "Praze" in learned


def test_rain_in_prague_answers_by_containment():
    """E2E zadavatele: „V Praze prší." → „Pršelo v Čechách?" → Ano
    (Praha ⊂ Čechy); „na Moravě?" → poctivé nenašel — oblast je FILTR
    Toposu, ne účastník existence."""
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             clock=lambda: NOW)
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn("V Praze prší.")
    assert "Zapamatováno" in stored.text
    assert iris.turn("Pršelo v Praze?").text == "Ano"
    assert iris.turn("Pršelo v Čechách?").text == "Ano"
    out = iris.turn("Pršelo na Moravě?")
    assert out.text != "Ano"


def test_keys_druhy_orez_lokativ_singularu():
    """Dávka D (nález při T7/B7): „Jeruzalém" se v gazetteeru ořízne na
    „jeruzal" (-ém), ale lokativ „Jeruzalémě" jen na „jeruzalem" — klíče
    se minuly a místní filtr se u těchto míst nikdy nenárokoval. Druhý
    ořez (s délkovou pojistkou ≥ 4) je srovná; „Betlémě" týž vzor."""
    from jellyai.iris.subsystems.topos import _key, _keys
    assert _key("Jeruzalém") in _keys("Jeruzalémě")
    assert _key("Betlém") in _keys("Betlémě")
    assert _key("Praha") in _keys("Praze")        # palatalizace drží dál
    assert _key("Plzeň") in _keys("Plzni")        # epenteze drží dál
