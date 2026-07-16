"""04 — Konverzace a těžiště, bez modelů.

Navazující otázka bez vlastního tématu se svede na „nejteplejší" uzel rozhovoru.
Spusť: python examples/04_conversation.py
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient

graph = FactGraph()
graph.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
graph.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("num", "1818", "number")]))

q1, q2 = "kdo napsal Babičku?", "kdy se narodila?"
client = FakeUfalClient(parse={
    q1: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]],
    q2: [[
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 15},
    ]],
})

answerer = GraphAnswerer(graph, client, ExtractiveAnswerer(AnswererConfig()))
print(q1, "→", answerer.answer(q1, []).text)   # Božena Němcová
print(q2, "→", answerer.answer(q2, []).text)   # 1818 (navázalo přes těžiště)
print("těžiště:", answerer.context.hottest())
