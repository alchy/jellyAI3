"""02 — Faktový graf a dotazy, bez modelů.

Nejrychlejší cesta: `jellyai.demo()` postaví mini-graf a odpoví. Níže i ručně.
Spusť: python examples/02_fact_graph.py
"""

import jellyai
from jellyai.graph.extract import make_fact, Participant

# 1) zero-setup ukázka
jellyai.demo()

# 2) postav vlastní graf z faktů a prohlédni ho
graph = jellyai.FactGraph()
graph.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "concept")]))
print("uzlů:", len(graph.nodes), "faktů:", len(graph.facts))
print("fakty o R.U.R.:", graph.facts_of("R.U.R.", role="obj"))
