"""Reifikovaný faktový graf — entitní uzly + faktové uzly + role-hrany.

Faktový uzel je reifikovaná událost (predikát + váha opakování); k němu vedou
role-hrany na účastníky (entitní/hodnotové uzly). Index `_by_node` umožní z uzlu
najít fakty, v nichž vystupuje (a v jaké roli) — to je základ 2-skokového průchodu.
"""

import os
import pickle
import re
from dataclasses import dataclass

from jellyai.graph.extract import extract_facts, make_fact, Participant, _SUBJ
from jellyai.graph.activation import ActivationField

_MONTHS = {
    "ledna": "leden", "února": "únor", "března": "březen", "dubna": "duben",
    "května": "květen", "června": "červen", "července": "červenec", "srpna": "srpen",
    "září": "září", "října": "říjen", "listopadu": "listopad", "prosince": "prosinec",
}


def parse_date(text):
    """Rozloží české datum na složky, které najde (rok/měsíc/den).

    Robustně regexem — datum bývá „13. ledna 1890", „roku 1890" i jen „1890".

    Args:
        text (str): Text časové entity.

    Returns:
        dict: Podmnožina {„rok": str, „měsíc": str (nominativ), „den": str}.
    """
    out = {}
    year = re.search(r"\b(1\d{3}|20\d{2})\b", text)
    if year:
        out["rok"] = year.group(1)
    day = re.search(r"\b([12]?\d|3[01])\.\s", text)
    if day:
        out["den"] = day.group(1)
    for genitive, nominative in _MONTHS.items():
        if genitive in text:
            out["měsíc"] = nominative
            break
    return out


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


def _canonical_persons(items):
    """Sestaví mapu osobní jméno → nejdelší tvar téhož jména v dokumentu.

    NameTag tvoří překrývající se jména („Karel", „Karel Čapek", „Karel Antonín
    Čapek"). Aby se fakta téže osoby nerozpadla, sjednotíme každý fragment na
    nejdelší jméno, které obsahuje všechna jeho slova.

    Args:
        items (list[tuple[int, dict]]): (index věty, anotace) dokumentu.

    Returns:
        dict: osobní jméno → kanonický (nejdelší) tvar.
    """
    persons = set()
    for _, annotation in items:
        for e in annotation.get("entities", []):
            if e.get("type", "")[:1].lower() == "p":
                persons.add(e["text"])
    # deterministicky (set má náhodné pořadí → jinak nereprodukovatelný graf)
    ordered = sorted(persons)
    canon = {}
    for p in ordered:
        words = set(p.split())
        best = p
        for q in ordered:
            if words <= set(q.split()) and len(q.split()) > len(best.split()):
                best = q
        canon[p] = best
    return canon


def _warm_persons(field, annotation, canon):
    """Rozsvítí osobní entity věty (kanonicky); podmětovou entitu silněji.

    Args:
        field (ActivationField): Aktivační pole dokumentu.
        annotation (dict): Anotace věty (entity + tokeny).
        canon (dict): Kanonizace osobních jmen.
    """
    persons = [e for e in annotation.get("entities", [])
               if e.get("type", "")[:1].lower() == "p"]
    subj_spans = [(t["start"], t["end"])
                  for sent in annotation.get("sentences", []) for t in sent
                  if t.get("deprel") in _SUBJ and t.get("start") is not None]
    for e in persons:
        is_subject = (e.get("start") is not None
                      and any(e["start"] <= s and en <= e["end"] for s, en in subj_spans))
        field.warm((canon.get(e["text"], e["text"]), "person"), 2.0 if is_subject else 1.0)


def build_graph(annotations):
    """Postaví faktový graf ze všech větných anotací (s aktivační koreferencí).

    Anotace se zpracují **po dokumentech v pořadí vět**; aktivační pole drží
    „aktuální subjekt" (naposledy zmíněná osoba, s pohasínáním). Věty s elidovaným
    podmětem (pro-drop) se přiřadí nejteplejší osobě — tím se zachytí biografická
    fakta a správně se ošetří i přesun tématu (odstavec o jiné osobě).

    Args:
        annotations (dict): (doc_id, index věty) → anotace (viz `annotate_documents`).

    Returns:
        FactGraph: Naplněný graf.
    """
    by_doc = {}
    for key, annotation in annotations.items():
        doc_id, idx = key if isinstance(key, tuple) else (key, 0)
        by_doc.setdefault(doc_id, []).append((idx, annotation))
    graph = FactGraph()
    for _, items in by_doc.items():
        items.sort(key=lambda t: t[0])
        canon = _canonical_persons(items)
        field = ActivationField()
        for _, annotation in items:
            subject = field.hottest()          # (id, typ) nejteplejší osoby, nebo None
            for fact in extract_facts(annotation, default_subject=subject, canon=canon):
                graph.add_fact(fact)
            _warm_persons(field, annotation, canon)
            field.step()
    _decompose_dates(graph)
    return graph


def _decompose_dates(graph):
    """Zanoří časové uzly: datum se stane uzlem s vlastními pod-fakty rok/měsíc/den.

    „13. ledna 1890" pak není jen řetězcová hodnota, ale uzel grafu, z něhož se dá
    dojít na rok (1890) — umožní dotaz „v kterém roce". Reifikace o patro níž.

    Args:
        graph (FactGraph): Graf s časovými uzly (upraví se in-place).
    """
    for node in list(graph.nodes.values()):
        if node.type != "time":
            continue
        for part, value in parse_date(node.id).items():
            vtype = "number" if part in ("rok", "den") else "concept"
            graph.add_fact(make_fact(part, [Participant("subj", node.id, "time"),
                                            Participant("val", value, vtype)]))
