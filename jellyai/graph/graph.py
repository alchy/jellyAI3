"""Reifikovaný faktový graf — entitní uzly + faktové uzly + role-hrany.

Faktový uzel je reifikovaná událost (predikát + váha opakování); k němu vedou
role-hrany na účastníky (entitní/hodnotové uzly). Index `_by_node` umožní z uzlu
najít fakty, v nichž vystupuje (a v jaké roli) — to je základ 2-skokového průchodu.
"""

import os
import pickle
from dataclasses import dataclass

from jellyai.graph.extract import extract_facts


@dataclass
class Node:
    """Entitní/hodnotový uzel.

    Atributy:
        id (str): Kanonické id (entita nebo nominativní lemma).
        type (str): Typ (person/geo/time/number/concept/institution).
        weight (int): V kolika faktech-výskytech uzel figuruje.
    """
    id: str
    type: str
    weight: int = 0


@dataclass
class FactNode:
    """Reifikovaný fakt (uzel).

    Atributy:
        id (tuple): Klíč (predicate, participants) — identita faktu.
        predicate (str): Predikát.
        weight (int): Kolikrát se fakt v korpusu opakoval.
        participants (tuple): N-tice Participant.
    """
    id: tuple
    predicate: str
    weight: int
    participants: tuple


class FactGraph:
    """Graf entitních a faktových uzlů propojených role-hranami."""

    def __init__(self):
        """Prázdný graf (`nodes`, `facts`, index `_by_node`)."""
        self.nodes = {}
        self.facts = {}
        self._by_node = {}

    def add_fact(self, fact):
        """Přidá fakt: sloučí podle identity (`váha++`) nebo založí; udrží indexy.

        Args:
            fact (Fact): Reifikovaný fakt z extrakce.
        """
        key = (fact.predicate, fact.participants)
        node = self.facts.get(key)
        if node is None:
            node = FactNode(key, fact.predicate, 0, fact.participants)
            self.facts[key] = node
            for p in fact.participants:
                self._by_node.setdefault(p.node, []).append((key, p.role))
        node.weight += 1
        for p in fact.participants:
            self._touch(p.node, p.type)

    def _touch(self, node_id, node_type):
        """Zajistí uzel a připočte frekvenci."""
        n = self.nodes.get(node_id)
        if n is None:
            self.nodes[node_id] = Node(node_id, node_type, 1)
        else:
            n.weight += 1

    def facts_of(self, node_id, role=None, predicate=None):
        """Fakty, v nichž uzel vystupuje (volitelně filtr role a predikátu).

        Args:
            node_id (str): Id uzlu.
            role (str | None): Požadovaná role uzlu ve faktu.
            predicate (str | None): Požadovaný predikát faktu.

        Returns:
            list[FactNode]: Odpovídající faktové uzly.
        """
        out = []
        for (key, r) in self._by_node.get(node_id, []):
            if role is not None and r != role:
                continue
            fact = self.facts[key]
            if predicate is not None and fact.predicate != predicate:
                continue
            out.append(fact)
        return out

    @staticmethod
    def participants(fact_node, role):
        """Id účastníků faktu dané role."""
        return [p.node for p in fact_node.participants if p.role == role]

    def save(self, path):
        """Uloží graf (pickle). Vrátí cestu."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "facts": self.facts,
                         "by_node": self._by_node}, f)
        return path

    @classmethod
    def load(cls, path):
        """Načte graf z disku."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        g = cls()
        g.nodes = state["nodes"]
        g.facts = state["facts"]
        g._by_node = state["by_node"]
        return g


def build_graph(annotations):
    """Postaví faktový graf ze všech větných anotací.

    Args:
        annotations (dict): (doc_id, index věty) → anotace (viz `annotate_documents`).

    Returns:
        FactGraph: Naplněný graf.
    """
    graph = FactGraph()
    for annotation in annotations.values():
        for fact in extract_facts(annotation):
            graph.add_fact(fact)
    return graph
