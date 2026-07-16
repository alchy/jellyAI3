"""Predikát jako preference — druhé patro matchování přes kontextovou asociaci.

Když přesný predikát otázky nemá v grafu fakt, odpoví asociační fakty
„kontext" (dokumentová blízkost entit — role ③ aktivačního pole). Porozumění
z grafu a vah, ne z ručně vytěžených řádků.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _client(q):
    return FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 2, "deprel": "obj"},
    ]]})


def test_predicate_falls_back_to_context_association():
    """„Kdo napsal R.U.R.?" bez napsat-faktu → nejtěžší asociovaná osoba."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "R.U.R.", "person")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Harry Domin", "person"),
                                     Participant("obj", "R.U.R.", "person")]))
    q = "Kdo napsal R.U.R.?"
    a = GraphAnswerer(g, _client(q), ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Karel Čapek"     # váha 3 > 1


def test_identity_hole_never_guesses_from_context():
    """„Kdo je X?" (díra pred) kontextovou asociací NEhádá — bez být-faktu je
    poctivá odpověď „nenašel", ne nejtěžší soused (dialog: Ludvík Němec)."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "Ludvík Němec", "person")]))
    q = "Kdo je Ludvík Němec?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Ludvík", "lemma": "Ludvík", "upos": "PROPN", "head": 0,
         "deprel": "root"},
        {"form": "Němec", "lemma": "Němec", "upos": "PROPN", "head": 3,
         "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert "Karel" not in a.answer(q, []).text


def test_exact_predicate_beats_association():
    """Existuje-li přesný fakt, asociace se nepoužije (predikát = preference)."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Josef Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    for _ in range(9):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "R.U.R.", "dílo")]))
    q = "Kdo napsal R.U.R.?"
    a = GraphAnswerer(g, _client(q), ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Josef Čapek"
