# Faktový graf — Implementační plán (reifikované n-ární fakty)

> **Pro agentní workery:** POVINNÁ SUB-SKILL: superpowers:executing-plans. Kroky mají
> checkboxy (`- [ ]`).

**Goal:** Z větných anotací poskládat vážený **reifikovaný n-ární** faktový graf
(faktový uzel = událost s predikátem + role-hrany na účastníky, váha = opakování) a
odpovídat na faktické otázky **2-skokem** (téma → fakt → hodnota); export do viewBase.

**Architecture:** Balíček `jellyai/graph/` (`extract` → `Fact`/`Participant` z anotace,
`graph` → `FactGraph` s entitními i faktovými uzly a indexem `_by_node`,
`viewbase_export`) + `GraphAnswerer` (`mode="graph"`) používající `analyze_question`.

**Tech Stack:** Python 3.11/3.12, stdlib. NetworkX jen líný import při exportu.

## Global Constraints

- Bez nových závislostí jádra (NetworkX líně jen při exportu).
- **Fakt je uzel** (reifikace): `predicate` na uzlu, účastníci přes role-hrany
  (`subj/obj/time/loc/num/pred`). Váha faktu = opakování.
- Uzel entity/hodnoty = NameTag entita (kanonicky) nebo nominativní lemma
  (`selection._clean_lemma`); typ `number` pro NUM, jinak `concept`.
- Identita faktu = `(predicate, seřazená n-tice (role, uzel))`.
- Bohaté české docstringy. Testy hermetické. Spouštění `.venv/bin/python -m pytest`.
- TDD, jeden commit na úkol; message končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Extrakce faktů — podmět–sloveso–předmět

**Files:**
- Create: `jellyai/graph/__init__.py` (prázdný)
- Create: `jellyai/graph/extract.py`
- Test: `tests/test_graph_extract.py`

**Interfaces:**
- Produces: `Participant(role, node, type)`, `Fact(predicate, participants)`
  (frozen), `make_fact(predicate, participants) -> Fact`, `_node_for(token, entities)`,
  `extract_facts(annotation) -> list[Fact]`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_graph_extract.py
from jellyai.graph.extract import extract_facts, make_fact, Fact, Participant


def _ann(sent, entities=None):
    return {"entities": entities or [], "sentences": [sent]}


def test_svo_fact():
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]
    ents = [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")])
    assert expected in facts
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_graph_extract.py -q`
Expected: FAIL (`ModuleNotFoundError: jellyai.graph.extract`).

- [ ] **Step 3: Implementuj balíček + extrakci S-V-O**

Vytvoř prázdný `jellyai/graph/__init__.py` a `jellyai/graph/extract.py`:

```python
"""Extrakce reifikovaných faktů z větné anotace.

Každá slovesná událost se stane jedním **faktem** (Fact) s predikátem a účastníky
v rolích (podmět/předmět/čas/místo). Fakt je pozdější uzel grafu — reifikace vztahu.
Uzel účastníka je pojmenovaná entita (kanonicky) nebo nominativní lemma tokenu.
"""

from dataclasses import dataclass

from jellyai.answerer.selection import _clean_lemma

_SUBJ = {"nsubj", "nsubj:pass"}
_OBJ = {"obj", "iobj"}
_ATTR = {"obl", "nmod"}
_ENTITY_TYPE = {"p": "person", "g": "geo", "t": "time", "i": "institution"}
_ATTR_ROLE = {"time": "time", "geo": "loc", "number": "num"}   # typ cíle → role


@dataclass(frozen=True)
class Participant:
    """Účastník faktu v konkrétní roli.

    Atributy:
        role (str): Role (subj/obj/time/loc/num/pred).
        node (str): Id uzlu účastníka.
        type (str): Typ uzlu (person/geo/time/number/concept/institution).
    """
    role: str
    node: str
    type: str


@dataclass(frozen=True)
class Fact:
    """Reifikovaná událost — pozdější faktový uzel.

    Atributy:
        predicate (str): Predikát (lemma slovesa, nebo „být" u spony).
        participants (tuple): Seřazená n-tice Participant (určuje identitu faktu).
    """
    predicate: str
    participants: tuple


def make_fact(predicate, participants):
    """Sestaví `Fact` s deterministicky seřazenými účastníky (kvůli identitě).

    Args:
        predicate (str): Predikát faktu.
        participants (list[Participant]): Účastníci (libovolné pořadí).

    Returns:
        Fact: Fakt s n-ticí účastníků seřazenou podle (role, node).
    """
    return Fact(predicate, tuple(sorted(participants, key=lambda p: (p.role, p.node))))


def _entity_type(entity):
    """CNEC typ entity (první písmeno) → typ uzlu."""
    return _ENTITY_TYPE.get(entity.get("type", "")[:1].lower(), "concept")


def _node_for(token, entities):
    """Vrátí (id, typ) uzlu pro token: entita (kanonicky) nebo nominativní lemma.

    Args:
        token (dict): Token s start/end/lemma/upos.
        entities (list[dict]): Entity věty s offsety.

    Returns:
        tuple[str, str] | None: (id, typ), nebo None když token nemá lemma.
    """
    start, end = token.get("start"), token.get("end")
    if start is not None and end is not None:
        for e in entities:
            if e.get("start") is not None and e["start"] <= start and end <= e["end"]:
                return e["text"], _entity_type(e)
    lemma = _clean_lemma(token.get("lemma", ""))
    if not lemma:
        return None
    return lemma, ("number" if token.get("upos") == "NUM" else "concept")


def _children(sent, head_id):
    """Tokeny věty s `head` == head_id (1-based)."""
    return [t for t in sent if t.get("head") == head_id]


def _first(tokens, deprels):
    """První token s `deprel` z množiny, nebo None."""
    return next((t for t in tokens if t.get("deprel") in deprels), None)


def extract_facts(annotation):
    """Vytáhne z anotace věty seznam reifikovaných faktů.

    Pro každý sloveso-token s podmětem vznikne jeden n-ární fakt (podmět + předmět +
    atributy). Sponová věta → fakt „být" (podmět + přísudek). Fakt bez dalšího
    účastníka než podmět se zahazuje.

    Args:
        annotation (dict): {"entities": [...], "sentences": [[token,...],...]}.

    Returns:
        list[Fact]: Nalezené fakty (mohou se opakovat — agreguje graf).
    """
    facts = []
    entities = annotation.get("entities", [])
    for sent in annotation.get("sentences", []):
        for head_id in range(1, len(sent) + 1):
            head = sent[head_id - 1]
            children = _children(sent, head_id)
            subj = _first(children, _SUBJ)
            if subj is None:
                continue
            subj_node = _node_for(subj, entities)
            if subj_node is None:
                continue
            if head.get("upos") != "VERB":
                continue
            verb = _clean_lemma(head.get("lemma", ""))
            parts = [Participant("subj", subj_node[0], subj_node[1])]
            obj = _first(children, _OBJ)
            if obj:
                o = _node_for(obj, entities)
                if o:
                    parts.append(Participant("obj", o[0], o[1]))
            if len(parts) > 1:
                facts.append(make_fact(verb, parts))
    return facts
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_graph_extract.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/graph/__init__.py jellyai/graph/extract.py tests/test_graph_extract.py
git commit -m "feat: extract_facts — reifikované fakty podmět-sloveso-předmět"
```

---

### Task 2: Extrakce — spona a n-ární atributy

**Files:**
- Modify: `jellyai/graph/extract.py`
- Test: `tests/test_graph_extract.py`

**Interfaces:**
- Produces: rozšířené `extract_facts` o spony (`být`) a atributy (`time/loc/num`) →
  n-ární fakt (podmět + čas + místo v jednom).

- [ ] **Step 1: Napiš padající testy**

```python
def test_copula_fact():
    sent = [
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 7, "end": 9},
        {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 19},
    ]
    facts = extract_facts(_ann(sent))
    expected = make_fact("být", [Participant("subj", "Rossum", "concept"),
                                 Participant("pred", "vynálezce", "concept")])
    assert expected in facts


def test_nary_fact_place_and_time():
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 13},
        {"form": "Praze", "lemma": "Praha", "upos": "PROPN", "head": 2, "deprel": "obl", "start": 17, "end": 22},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 2, "deprel": "obl", "start": 23, "end": 27},
    ]
    ents = [{"text": "Čapek", "type": "P", "start": 0, "end": 5},
            {"text": "Praha", "type": "G", "start": 17, "end": 22}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                     Participant("loc", "Praha", "geo"),
                                     Participant("num", "1890", "number")])
    assert expected in facts       # jeden n-ární fakt: podmět + místo + čas
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_graph_extract.py -q`
Expected: FAIL (spona a atributy se zatím neextrahují).

- [ ] **Step 3: Rozšiř `extract_facts`**

V `extract_facts`, hned za `subj_node is None` guard (a **před** `if head.get("upos")
!= "VERB": continue`), přidej sponu:

```python
            if _first(children, {"cop"}):
                pred = _node_for(head, entities)
                if pred:
                    facts.append(make_fact("být", [
                        Participant("subj", subj_node[0], subj_node[1]),
                        Participant("pred", pred[0], pred[1]),
                    ]))
                continue
```

A do slovesné větve (za zpracování předmětu, před `if len(parts) > 1`) přidej atributy:

```python
            for attr in children:
                base = (attr.get("deprel") or "").split(":")[0]
                if base not in _ATTR:
                    continue
                a = _node_for(attr, entities)
                if a and a[1] in _ATTR_ROLE:
                    parts.append(Participant(_ATTR_ROLE[a[1]], a[0], a[1]))
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_graph_extract.py -q`
Expected: PASS (3 testy).

- [ ] **Step 5: Commit**

```bash
git add jellyai/graph/extract.py tests/test_graph_extract.py
git commit -m "feat: extrakce spony (být) a n-árních atributů (čas/místo/číslo)"
```

---

### Task 3: `FactGraph` + `build_graph` + váhy

**Files:**
- Create: `jellyai/graph/graph.py`
- Test: `tests/test_fact_graph.py`

**Interfaces:**
- Produces: `Node(id, type, weight)`, `FactNode(id, predicate, weight, participants)`,
  `FactGraph` (`add_fact`, `facts_of(node_id, role, predicate)`,
  `participants(fact_node, role)`, `nodes`, `facts`), `build_graph(annotations)`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_fact_graph.py
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph, build_graph


def _born(year):
    return make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                 Participant("num", year, "number")])


def test_fact_weight_aggregates_repetition():
    g = FactGraph()
    for _ in range(3):
        g.add_fact(_born("1890"))
    g.add_fact(_born("1915"))
    born90 = [f for f in g.facts.values() if ("num", "1890", "number") in
              [(p.role, p.node, p.type) for p in f.participants]][0]
    assert born90.weight == 3
    facts = g.facts_of("Čapek", role="subj", predicate="narodit")
    assert len(facts) == 2
    assert g.participants(born90, "num") == ["1890"]


def test_build_graph_from_annotations():
    ann = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                      "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    g = build_graph(ann)
    assert g.facts_of("Božena Němcová", role="subj", predicate="napsat")
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_fact_graph.py -q`
Expected: FAIL (`ModuleNotFoundError: jellyai.graph.graph`).

- [ ] **Step 3: Implementuj `graph.py`**

```python
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
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_fact_graph.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/graph/graph.py tests/test_fact_graph.py
git commit -m "feat: FactGraph (reifikované fakty) + build_graph + váhy"
```

---

### Task 4: Perzistence grafu (round-trip)

**Files:**
- Test: `tests/test_fact_graph.py`

**Interfaces:** Consumes `FactGraph.save/load` (Task 3).

- [ ] **Step 1: Napiš test**

```python
def test_graph_save_load_roundtrip(tmp_path):
    g = FactGraph()
    g.add_fact(_born("1890"))
    path = str(tmp_path / "graph.pkl")
    g.save(path)
    loaded = FactGraph.load(path)
    assert list(loaded.facts.keys()) == list(g.facts.keys())
    assert loaded.facts_of("Čapek", predicate="narodit")[0].weight == 1
```

- [ ] **Step 2: Spusť — musí projít** (save/load je z Task 3)

Run: `.venv/bin/python -m pytest tests/test_fact_graph.py::test_graph_save_load_roundtrip -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fact_graph.py
git commit -m "test: round-trip perzistence FactGraph"
```

---

### Task 5: `GraphAnswerer` — odpovídání 2-skokem

**Files:**
- Create: `jellyai/answerer/graph_answerer.py`
- Test: `tests/test_graph_answerer.py`

**Interfaces:**
- Produces: `GraphAnswerer(graph, client, fallback)` (podtřída `Answerer`).
- Consumes: `analyze_question`, `FactGraph.facts_of/participants`, `Answer`.

- [ ] **Step 1: Napiš padající testy**

```python
# tests/test_graph_answerer.py
from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _graph():
    g = FactGraph()
    born = make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("loc", "Praha", "geo"),
                                 Participant("num", "1890", "number")])
    for _ in range(3):
        g.add_fact(born)
    g.add_fact(make_fact("žít", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("num", "1915", "number")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    return g


def _client(q, tokens):
    return FakeUfalClient(parse={q: [tokens]})


def test_kdy_picks_most_repeated_fact():
    q = "kdy se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 14},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 15, "end": 20},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 21, "end": 26},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "1890"


def test_kde_uses_same_nary_fact():
    q = "kde se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "kde", "lemma": "kde", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 14},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 15, "end": 20},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 21, "end": 26},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Praha"        # z TÉHOŽ narozovacího faktu


def test_kdo_traverses_object():
    q = "kdo napsal Babičku?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Božena Němcová"


def test_missing_topic_falls_back():
    q = "kdo je Rossum?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 4, "end": 6},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 7, "end": 13},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text != "Rossum"       # není v grafu → fallback
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_graph_answerer.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj `GraphAnswerer`**

```python
"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.question import analyze_question


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback."""

    def __init__(self, graph, client, fallback):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (rozbor otázky).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback

    def _resolve_topic(self, topic_terms):
        """Najde uzel tématu (shoda lemmatu s id uzlu), s nejvyšší vahou."""
        terms = [t.lower() for t in topic_terms if t]
        best = None
        for node in self.graph.nodes.values():
            nid = node.id.lower()
            if any(term == nid or term in nid.split() for term in terms):
                if best is None or node.weight > best.weight:
                    best = node
        return best.id if best else None

    def _pick(self, facts, role):
        """Z faktů (s daným rolem cíle) vrátí účastníka z faktu s nejvyšší vahou."""
        best = None
        for fact in facts:
            values = self.graph.participants(fact, role)
            if not values:
                continue
            if best is None or fact.weight > best[0]:
                best = (fact.weight, values[0])
        return best[1] if best else None

    def answer(self, question, retrieved):
        """Odpoví 2-skokem grafu; při neúspěchu deleguje na fallback.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list): Pasáže (jen pro fallback).

        Returns:
            Answer: Odpověď z grafu (zdroj „graf"), nebo výsledek fallbacku.
        """
        qa = analyze_question(question, self.client)
        topic = self._resolve_topic(qa.topic_terms)
        result = None
        if topic is not None:
            g, verb = self.graph, qa.verb_lemma
            if qa.is_copula or qa.qtype in ("Jaký", "Který"):
                result = self._pick(g.facts_of(topic, role="subj", predicate="být"), "pred")
            elif qa.qtype in ("Kdy", "Kde", "Kolik"):
                facts = (g.facts_of(topic, role="subj", predicate=verb)
                         or g.facts_of(topic, role="subj"))
                if qa.qtype == "Kdy":
                    result = self._pick(facts, "time") or self._pick(facts, "num")
                elif qa.qtype == "Kde":
                    result = self._pick(facts, "loc")
                else:
                    result = self._pick(facts, "num")
            elif qa.qtype in ("Kdo", "Co"):
                result = (self._pick(g.facts_of(topic, role="obj", predicate=verb), "subj")
                          or self._pick(g.facts_of(topic, role="subj", predicate=verb), "obj"))
        if result is None:
            return self.fallback.answer(question, retrieved)
        return Answer(text=result, sources=["graf"], score=1.0)
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_graph_answerer.py -q`
Expected: PASS (4 testy).

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/graph_answerer.py tests/test_graph_answerer.py
git commit -m "feat: GraphAnswerer — 2-skokové odpovídání (n-ární fakty, max váha)"
```

---

### Task 6: Konfigurace + pipeline dispatch

**Files:**
- Modify: `config.py`, `jellyai/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:** `GraphConfig(graph_path="data/graph.pkl")`, `Config.graph`,
`_make_answerer` větev "graph".

- [ ] **Step 1: Napiš padající test**

```python
def test_make_answerer_graph_mode(tmp_path):
    from config import Config, GraphConfig, AnswererConfig
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.pipeline import _make_answerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "A", "concept"),
                                 Participant("pred", "B", "concept")]))
    path = str(tmp_path / "graph.pkl"); g.save(path)
    cfg = Config()
    cfg.graph = GraphConfig(graph_path=path)
    cfg.answerer = AnswererConfig(mode="graph")
    assert isinstance(_make_answerer(cfg), GraphAnswerer)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::test_make_answerer_graph_mode -q`
Expected: FAIL (chybí `GraphConfig`).

- [ ] **Step 3: Přidej `GraphConfig` do `config.py`**

Za `ServicesConfig` (před `class Config`):
```python
@dataclass
class GraphConfig:
    """Nastavení faktového grafu.

    Atributy:
        graph_path (str): Cesta k uloženému grafu.
    """
    graph_path: str = "data/graph.pkl"
```
Do `class Config` (za `services`):
```python
    graph: GraphConfig = field(default_factory=GraphConfig)
```
A do docstringu `AnswererConfig.mode` doplň možnost `"graph"`.

- [ ] **Step 4: Přidej větev do `_make_answerer`**

V `jellyai/pipeline.py`, v `_make_answerer`, před `return ExtractiveAnswerer(...)`:
```python
    if config.answerer.mode == "graph":
        from jellyai.graph.graph import FactGraph
        from jellyai.answerer.graph_answerer import GraphAnswerer
        from jellyai.ufal_client import UfalClient
        graph = FactGraph.load(config.graph.graph_path)
        return GraphAnswerer(graph, UfalClient(config.services),
                             ExtractiveAnswerer(config.answerer))
```

- [ ] **Step 5: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add config.py jellyai/pipeline.py tests/test_pipeline.py
git commit -m "feat: GraphConfig + pipeline dispatch pro mode=graph"
```

---

### Task 7: Export do viewBase (uzly + faktové uzly + role-hrany)

**Files:**
- Create: `jellyai/graph/viewbase_export.py`
- Test: `tests/test_viewbase_export.py`

**Interfaces:** `to_json(graph) -> dict`, `to_networkx(graph) -> nx.DiGraph` (líný import).

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_viewbase_export.py
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.viewbase_export import to_json


def test_to_json_has_fact_nodes_and_role_edges():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Čapek", "person"),
                                    Participant("obj", "R.U.R.", "concept")]))
    data = to_json(g)
    ids = {n["id"] for n in data["nodes"]}
    types = {n["type"] for n in data["nodes"]}
    assert "Čapek" in ids and "R.U.R." in ids
    assert "fact" in types                                    # faktový uzel je taky uzel
    roles = {e["role"] for e in data["edges"]}
    assert {"subj", "obj"} <= roles                           # role-hrany
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_viewbase_export.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj export**

```python
"""Export reifikovaného faktového grafu do viewBase.

viewBase je force-graph (Three.js + d3-force-3d) a umí `Canvas.from_networkx(G)`.
Do grafu jdou **entitní uzly i faktové uzly** (faktový uzel má typ `fact`, popisek =
predikát); hrany jsou role-hrany fakt→účastník s vahou faktu. `to_networkx` je
primární most (líný import), `to_json` je bezzávislostní alternativa.
"""


def _fact_id(fact_node):
    """Krátké čitelné id faktového uzlu (predikát + hash klíče)."""
    return "fact:" + fact_node.predicate + ":" + str(abs(hash(fact_node.id)) % 100000)


def to_json(graph):
    """Serializuje graf do {nodes, edges} (entitní i faktové uzly, role-hrany).

    Args:
        graph (FactGraph): Graf k exportu.

    Returns:
        dict: {"nodes": [{id,type,weight,label}], "edges": [{src,dst,role,weight}]}.
    """
    nodes = [{"id": n.id, "type": n.type, "weight": n.weight, "label": n.id}
             for n in graph.nodes.values()]
    edges = []
    for fact in graph.facts.values():
        fid = _fact_id(fact)
        nodes.append({"id": fid, "type": "fact", "weight": fact.weight,
                      "label": fact.predicate})
        for p in fact.participants:
            edges.append({"src": fid, "dst": p.node, "role": p.role,
                          "weight": fact.weight})
    return {"nodes": nodes, "edges": edges}


def to_networkx(graph):
    """Převede graf na `networkx.DiGraph` (most do viewBase `from_networkx`).

    NetworkX se importuje líně (není závislost jádra). Faktové uzly mají
    `type="fact"` a `label=predikát`; hrany nesou `role` a `weight`.

    Args:
        graph (FactGraph): Graf k exportu.

    Returns:
        networkx.DiGraph: Uzly a role-hrany.
    """
    import networkx as nx
    g = nx.DiGraph()
    for n in graph.nodes.values():
        g.add_node(n.id, type=n.type, weight=n.weight, label=n.id)
    for fact in graph.facts.values():
        fid = _fact_id(fact)
        g.add_node(fid, type="fact", weight=fact.weight, label=fact.predicate)
        for p in fact.participants:
            g.add_edge(fid, p.node, role=p.role, weight=fact.weight)
    return g
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_viewbase_export.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/graph/viewbase_export.py tests/test_viewbase_export.py
git commit -m "feat: export do viewBase (faktové uzly + role-hrany)"
```

---

### Task 8: CLI `graph` + přepínač `--graph`

**Files:**
- Modify: `cli.py`
- Test: `tests/test_cli_graph.py`

**Interfaces:** `cmd_graph(config, view=False) -> int`; CLI `graph [--view]` + `--graph`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_cli_graph.py
import pickle
from config import Config, GraphConfig, ServicesConfig
from cli import cmd_graph


def test_cmd_graph_builds_from_annotations(tmp_path):
    ann_path = str(tmp_path / "ann.pkl")
    graph_path = str(tmp_path / "graph.pkl")
    annotations = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                              "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    with open(ann_path, "wb") as f:
        pickle.dump(annotations, f)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=graph_path)
    n = cmd_graph(cfg)
    assert n >= 2
    from jellyai.graph.graph import FactGraph
    g = FactGraph.load(graph_path)
    assert g.facts_of("Božena Němcová", predicate="napsat")
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_cli_graph.py -q`
Expected: FAIL (`ImportError: cannot import name 'cmd_graph'`).

- [ ] **Step 3: Přidej `cmd_graph` do `cli.py`** (za `cmd_annotate`)

```python
def cmd_graph(config, view=False):
    """Postaví faktový graf z větných anotací a uloží ho.

    Args:
        config (Config): Konfigurace (annotations_path, graph_path).
        view (bool): Zda po postavení exportovat do viewBase.

    Returns:
        int: Počet entitních uzlů grafu.
    """
    from jellyai.annotate import load_annotations
    from jellyai.graph.graph import build_graph
    annotations = load_annotations(config.services.annotations_path)
    graph = build_graph(annotations)
    graph.save(config.graph.graph_path)
    print(f"Faktový graf: {len(graph.nodes)} uzlů, {len(graph.facts)} faktů "
          f"→ {config.graph.graph_path}")
    if view:
        from jellyai.graph.viewbase_export import to_networkx
        try:
            from viewbase import Canvas
            Canvas.from_networkx(to_networkx(graph)).serve()
        except ImportError:
            print("viewBase/networkx není k dispozici — přeskočeno.")
    return len(graph.nodes)
```

- [ ] **Step 4: Zaregistruj příkaz a přepínač v `_build_parser`**

K ostatním `sub.add_parser(...)`:
```python
    p_graph = sub.add_parser("graph", parents=[common], help="postaví faktový graf z anotací")
    p_graph.add_argument("--view", action="store_true", help="export do viewBase")
```
Ke společnému parseru `common` (vedle `--template`):
```python
    common.add_argument("--graph", action="store_true",
                        help="odpovídat přes faktový graf (mode=graph)")
```

- [ ] **Step 5: Dispatch v `main`**

Za `if getattr(args, "template", False): ...`:
```python
    if getattr(args, "graph", False):
        config.answerer.mode = "graph"
```
Do řetězu `elif`:
```python
    elif args.command == "graph":
        cmd_graph(config, view=args.view)
```

- [ ] **Step 6: Spusť celou sadu**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add cli.py tests/test_cli_graph.py
git commit -m "feat: CLI graph (build + --view) a přepínač --graph"
```

---

### Task 9: Živé ověření + dokumentace

**Files:**
- Create: `docs/superpowers/2026-07-16-fact-graph-results.md`
- Modify: `README.md`

- [ ] **Step 1: Postav graf z reálných anotací**

```bash
./jelly annotate     # (pokud nejsou aktuální větné anotace)
./jelly graph        # data/graph.pkl; vypíše počet uzlů a faktů
```

- [ ] **Step 2: Ověř odpovídání přes graf**

```bash
./jelly ask --graph "kdy se narodil Karel Čapek?"
./jelly ask --graph "kde se narodil Karel Čapek?"
./jelly ask --graph "kdo napsal Babičku?"
./jelly ask --graph "kdo je Rossum?"
```
Zapiš skutečné odpovědi. Klíč: hlídej rok narození (opravuje B1 „1915") a že „kde/kdy"
čerpají z téhož faktu.

- [ ] **Step 3: Export do viewBase**

```bash
pip install networkx      # jen pro export
./jelly graph --view
```
Zaznamenej, co funguje / co ve viewBase chybí (kandidát na rozšíření jejího repa —
zejména vizualizace tras dotazu).

- [ ] **Step 4: Results dokument**

`docs/superpowers/2026-07-16-fact-graph-results.md`: co graf umí, tabulka odpovědí
(srovnání s retrievalem/B1), poctivá omezení (řídké fakty, extrakce z reálných
rozborů), a zda se potvrdil směr „když funguje, zredukovat kód".

- [ ] **Step 5: README** — sekce „Faktový graf" + roadmapa.

- [ ] **Step 6: Sada + commit**

```bash
.venv/bin/python -m pytest -q
git add -A
git commit -m "docs: výsledky faktového grafu + README"
```

---

## Navazující (podmíněné) fáze — mimo tento plán

- **Vizualizace trasy dotazu** *(nápad uživatele)*: `GraphAnswerer` už při 2-skoku zná
  trasu (téma → faktový uzel → hodnota). Rozšíření: vracet trasu a předat viewBase
  („trasy") — dotaz proletí sítí přes faktové uzly. Blízký krok po ověření jádra.
- **Force layout jako služba**: rozšířit viewBase o backend počítající fyziku a vracet
  layout jellyAI3 (přispět upstream).
- **Konverzační „těžiště"**: hmotnost témat + posun v rozhovoru (na vrácené fyzice + B2).
- **Konsolidace kódu k grafu** *(nápad uživatele)*: až po ověření hodnoty jádra.
- Subsumpce částečných faktů, koreference, víceskoký průchod, hybrid vážení — později.

## Self-review (proti specu v2)

- §4 extrakce (S-V-O, spona, n-ární atributy, `make_fact`) → Task 1–2. ✓
- §5 struktura (Node, FactNode, `_by_node`, váhy) + perzistence → Task 3–4. ✓
- §6 odpovídání 2-skokem (`facts_of`/`_pick`, n-ární kde/kdy) → Task 5. ✓
- §7 CLI + §9 config/mode → Task 6, 8. ✓
- §8 export (faktové uzly + role-hrany) → Task 7. ✓
- §10 testy → hermetické Task 1–8, živě Task 9. ✓
- §11 hraniční případy → prázdný graf/téma mimo graf (Task 5 fallback), věta bez
  podmětu (Task 1 guard), fakt bez druhého účastníka (Task 1 `len(parts) > 1`). ✓
- §12 mimo rozsah + navazující → sekce výše. ✓

Typová konzistence: `Fact`/`Participant`/`make_fact` (Task 1) → `add_fact` (Task 3) →
`facts_of`/`participants` konzumuje `GraphAnswerer._pick` (Task 5); `FactNode.weight`
(opakování) je klíč pro výběr. ✓
