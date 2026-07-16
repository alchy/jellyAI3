"""demo() — první „ono to funguje" bez modelů a stahování.

Postaví malý vestavěný faktový graf a odpoví na pár otázek přes `GraphAnswerer`
s nakonzervovaným rozborem otázek (`FakeUfalClient`). Slouží jako rychlá ukázka pro
začátečníky: `import jellyai; jellyai.demo()`. Druhá otázka nemá vlastní téma —
navazuje přes konverzační těžiště.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient

_Q1 = "kdo napsal Babičku?"
_Q2 = "kdy se narodila?"


def _demo_graph():
    """Malý vestavěný graf o Boženě Němcové."""
    graph = FactGraph()
    graph.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                        Participant("obj", "Babička", "concept")]))
    graph.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                         Participant("num", "1818", "number")]))
    return graph


def _demo_client():
    """Nakonzervovaný rozbor ukázkových otázek (bez modelů)."""
    return FakeUfalClient(parse={
        _Q1: [[
            {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
            {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
            {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
        ]],
        _Q2: [[
            {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
            {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 15},
        ]],
    })


def demo(verbose=True):
    """Postaví mini-graf a odpoví na ukázkové otázky (bez modelů).

    Args:
        verbose (bool): Vypíše otázky a odpovědi na stdout.

    Returns:
        dict[str, str]: otázka → odpověď.
    """
    answerer = GraphAnswerer(_demo_graph(), _demo_client(),
                             ExtractiveAnswerer(AnswererConfig()))
    result = {}
    for question in (_Q1, _Q2):
        answer = answerer.answer(question, [])
        result[question] = answer.text
        if verbose:
            print(f"❓ {question}\n💬 {answer.text}\n")
    return result
