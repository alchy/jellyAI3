"""QueryAssurance — číselná jistota zaostření subjektu (0–1).

Řídí přechody automatu Iris: nad prahem se odpovídá, pod prahem se vede
dialog (dialog > figly). Skóre skládá kvalitu jmenné evidence (váhy pater
rozlišení), počet rovnocenných soupeřů a aktivaci vítěze z konverzace.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.iris.assurance import assurance
from jellyai.ufal_client import FakeUfalClient


def test_full_exact_single_candidate_is_confident():
    assert assurance(quality=1.0, rivals=0) >= 0.9


def test_equal_rivals_halve_confidence():
    """Dva rovnocenní kandidáti („Kdo je Čapek?") → jistota volby ~ 1/2."""
    assert assurance(quality=1.0, rivals=1) == 0.5


def test_weak_evidence_with_rival_is_uncertain():
    assert assurance(quality=0.25, rivals=1) < 0.5


def test_activation_raises_confidence():
    """Svítící vítěz (uživatel už zaostřil) jistotu vrací nad práh —
    dialogová smyčka tak konverguje."""
    low = assurance(quality=1.0, rivals=1)
    boosted = assurance(quality=1.0, rivals=1, activation=1.0)
    assert boosted > low and boosted >= 0.6


def test_affinity_filters_rivals():
    """Soupeř bez faktu predikátu není skutečná alternativa: „Kde se narodil
    Ježíš?" — narodit-fakt má jen jeden kandidát → volba není hádání
    (rivals prázdné). U identity („Kdo je Čapek?") mají fakt oba → dialog."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Ježíš", "person"),
                                     Participant("loc", "Betlémě", "geo")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Ježíš Martu", "person"),
                                     Participant("obj", "dům", "concept")]))
    a = GraphAnswerer(g, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))
    a._resolve_topic(["Ježíš"], "narodit")
    assert a.turn.resolution["winner"] == "Ježíš"
    assert a.turn.resolution["rivals"] == []          # Martu narodit-fakt nemá
    a._resolve_topic(["Ježíš"])                       # bez predikátu — beze změny
    assert "Ježíš Martu" in a.turn.resolution["rivals"]


def test_resolver_records_resolution_evidence():
    """_resolve_topic po ostrém rozlišení zapíše evidenci (kvalitu, soupeře)
    do turn.resolution — vstup pro assurance; sondy is_node nezapisují."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("pred", "spisovatel", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                 Participant("pred", "malíř", "concept")]))
    a = GraphAnswerer(g, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))
    winner = a._resolve_topic(["Čapek"])
    res = a.turn.resolution
    assert res["winner"] == winner
    rivals = set(res["rivals"])
    assert rivals == {"Karel Čapek", "Josef Čapek"} - {winner}
    # jednoznačné jméno soupeře nemá
    a._resolve_topic(["Karel", "Čapek"])
    assert a.turn.resolution["rivals"] == []
    # sonda (warm=False) evidenci nepřepisuje
    a._span_is_node("Josef")
    assert a.turn.resolution["term"] == "Karel Čapek"
