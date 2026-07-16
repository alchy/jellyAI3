# jellyAI3 výuková knihovna — Fáze 1 — Implementační plán

> **Pro agentní workery:** POVINNÁ SUB-SKILL: superpowers:executing-plans. Kroky mají
> checkboxy (`- [ ]`).

**Goal:** Přerodit jellyAI3 na `pip`-instalovatelnou výukovou knihovnu se skládačkou
**portů** (NN-ready), tenkou fasádou `Jelly` (DI), start/stop korpusem, JSON
perzistencí/konfigurací, teplotou shody + `Composer`em, vysvětlitelnou odpovědí,
logováním a akčními chybami. **Bez přesunů souborů.**

**Architecture:** Nové moduly `jellyai/ports.py` (Protocol rozhraní), `jellyai/errors.py`,
`jellyai/logs.py`, `jellyai/corpus.py` (refaktor `ufal_client` → `CorpusTools`),
`jellyai/answerer/composer.py`, `jellyai/facade.py` (`Jelly`), `jellyai/demo.py`;
naplněné `jellyai/__init__.py`; `pyproject.toml`; `examples/`. Stávající bloky se
nemění (jen aditivně: `temperature`, `Answer.trace`, `Config` JSON).

**Tech Stack:** Python 3.11/3.12, numpy, stdlib (`typing.Protocol`, `logging`, `json`).

## Global Constraints

- **Bez fyzických přesunů souborů** (přeskupení = pozdější fáze).
- **JSON > pickle** pro session a konfiguraci (čitelné).
- Bohaté české docstringy pod každou funkcí. Testy hermetické. `.venv/bin/python -m pytest`.
- Aditivní změny stávajících bloků (`temperature` default 0.0 = dnešní chování).
- Chyby přes `JellyError` s **akční** hláškou. Logging přes `logging`, default ticho.
- TDD, jeden commit na úkol; message končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `pyproject.toml` + veřejné API v `__init__.py`

**Files:**
- Create: `pyproject.toml`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces: `import jellyai` a symboly stávajících bloků na top-level.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_public_api.py
def test_public_api_exposes_blocks():
    import jellyai
    for name in ["load_documents", "chunk", "tokenize", "split_sentences",
                 "Retriever", "build_fact_graph", "FactGraph", "extract_facts",
                 "Fact", "Participant", "ExtractiveAnswerer", "GraphAnswerer",
                 "analyze_question", "Answer", "explain"]:
        assert hasattr(jellyai, name), f"chybí veřejný symbol {name}"
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_public_api.py -q`
Expected: FAIL (prázdné `__init__.py`).

- [ ] **Step 3: Naplň `jellyai/__init__.py`**

```python
"""jellyAI3 — výuková knihovna pro české QA nad texty.

Skládačka malých bloků (portů): retrieval, faktový graf, answerery, korpusové
nástroje. Píšeš program nad `import jellyai` a skládáš granulární funkce; kdo chce
rychlý výsledek, použije fasádu `Jelly`; kdo chce první „ono to funguje", zavolá
`demo()`. Bohaté české docstringy u každého bloku.
"""

from jellyai.loader import load_documents, Document
from jellyai.text import tokenize, split_sentences
from jellyai.chunker import chunk, Passage
from jellyai.retriever import Retriever
from jellyai.graph.graph import build_graph as build_fact_graph, FactGraph
from jellyai.graph.extract import extract_facts, Fact, Participant
from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.answerer.question import analyze_question
from jellyai.explain import explain

__version__ = "0.1.0"

__all__ = [
    "load_documents", "Document", "tokenize", "split_sentences", "chunk", "Passage",
    "Retriever", "build_fact_graph", "FactGraph", "extract_facts", "Fact", "Participant",
    "Answer", "Answerer", "ExtractiveAnswerer", "GraphAnswerer", "analyze_question",
    "explain",
]
```

- [ ] **Step 4: Vytvoř `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "jellyai"
version = "0.1.0"
description = "Výuková knihovna pro české QA nad texty (retrieval + faktový graf)"
requires-python = ">=3.11"
dependencies = ["numpy"]

[tool.setuptools]
packages = ["jellyai", "jellyai.answerer", "jellyai.graph"]
py-modules = ["config"]
```

- [ ] **Step 5: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_public_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml jellyai/__init__.py tests/test_public_api.py
git commit -m "feat: veřejné API v jellyai/__init__ + pyproject (pip install -e .)"
```

---

### Task 2: Porty — `Protocol` rozhraní

**Files:**
- Create: `jellyai/ports.py`
- Modify: `jellyai/__init__.py` (export portů)
- Test: `tests/test_ports.py`

**Interfaces:**
- Produces: `Tokenizer`, `FactExtractor`, `QuestionAnalyzer`, `Composer`, `CorpusPort`
  (`typing.Protocol`, `runtime_checkable`). Answerer/Retriever už existují jako
  rozhraní (ABC/třída).

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_ports.py
def test_ports_are_structural():
    from jellyai.ports import QuestionAnalyzer, Composer
    # jednoduchá třída se strukturálně shoduje s portem
    class MyAnalyzer:
        def analyze(self, question):
            return None
    assert isinstance(MyAnalyzer(), QuestionAnalyzer)

    class MyComposer:
        def compose(self, question, facts):
            return "text"
    assert isinstance(MyComposer(), Composer)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_ports.py -q`
Expected: FAIL (`ModuleNotFoundError: jellyai.ports`).

- [ ] **Step 3: Implementuj `jellyai/ports.py`**

```python
"""Porty knihovny — úzká rozhraní (Protocol), kam sedne i neuronová síť.

Primární abstrakce nejsou fasády, ale malé porty: každá fáze pipeline má úzký
kontrakt. Rozhraní jsou **strukturální** (`typing.Protocol`), takže stávající bloky
je splňují bez dědičnosti — a pozdější NN implementace stačí, když má stejné metody.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Rozdělí text na tokeny."""
    def tokenize(self, text: str) -> list: ...


@runtime_checkable
class QuestionAnalyzer(Protocol):
    """Rozebere otázku (typ, téma, sloveso…)."""
    def analyze(self, question: str): ...


@runtime_checkable
class FactExtractor(Protocol):
    """Z anotace věty vytáhne fakty."""
    def extract(self, annotation: dict) -> list: ...


@runtime_checkable
class Composer(Protocol):
    """Ze sady faktů složí čitelný text (víc než jednoslovná odpověď)."""
    def compose(self, question: str, facts: list) -> str: ...


@runtime_checkable
class CorpusPort(Protocol):
    """České korpusové nástroje (rozbor/entity/morfologie)."""
    def parse(self, text: str) -> list: ...
    def entities(self, text: str) -> list: ...
    def analyze(self, text: str) -> list: ...
    def generate(self, lemma: str, tag: str) -> list: ...
```

- [ ] **Step 4: Exportuj z `__init__.py`**

Přidej k importům a do `__all__`:
```python
from jellyai.ports import Tokenizer, QuestionAnalyzer, FactExtractor, Composer, CorpusPort
```
```python
    "Tokenizer", "QuestionAnalyzer", "FactExtractor", "Composer", "CorpusPort",
```

- [ ] **Step 5: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_ports.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jellyai/ports.py jellyai/__init__.py tests/test_ports.py
git commit -m "feat: porty (Protocol rozhraní) pro skládačku + NN hooky"
```

---

### Task 3: `JellyError` + logování

**Files:**
- Create: `jellyai/errors.py`, `jellyai/logs.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_errors_logs.py`

**Interfaces:**
- Produces: `JellyError(Exception)`, `ModelsMissingError`, `CorpusNotStartedError`;
  `get_logger()`, `set_debug(on)`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_errors_logs.py
import logging


def test_jelly_error_actionable():
    from jellyai.errors import JellyError, ModelsMissingError
    err = ModelsMissingError("data/models")
    assert isinstance(err, JellyError)
    assert "qa-models" in str(err)          # hláška říká, jak opravit


def test_logger_silent_by_default_and_debug_toggles():
    from jellyai.logs import get_logger, set_debug
    log = get_logger()
    assert any(isinstance(h, logging.NullHandler) for h in log.handlers)
    set_debug(True)
    assert log.level == logging.DEBUG
    set_debug(False)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_errors_logs.py -q`
Expected: FAIL (moduly chybí).

- [ ] **Step 3: Implementuj `jellyai/errors.py`**

```python
"""Chyby knihovny — srozumitelné a **akční** (říkají, jak to opravit)."""


class JellyError(Exception):
    """Základ chyb jellyAI3. Hláška má vždy vysvětlit i nápravu."""


class ModelsMissingError(JellyError):
    """Chybí ÚFAL modely."""
    def __init__(self, path):
        super().__init__(f"Modely nenalezeny v „{path}" — stáhni je: ./jelly qa-models")


class CorpusNotStartedError(JellyError):
    """Korpusové služby nejsou nastartované."""
    def __init__(self):
        super().__init__("Korpus není nastartovaný — použij `with jellyai.CorpusTools() "
                         "as t:` nebo zavolej `tools.start()`.")
```

- [ ] **Step 4: Implementuj `jellyai/logs.py`**

```python
"""Logování knihovny — správně: default ticho (NullHandler), debug přidá výstup.

Knihovna nemá konfigurovat globální logging hostitele. Proto logger „jellyai" má
`NullHandler` (mlčí), a `set_debug(True)` mu přidá `StreamHandler` a zapne DEBUG —
uživatel pak vidí, co knihovna dělá.
"""

import logging

_LOGGER_NAME = "jellyai"
_debug_handler = None


def get_logger():
    """Vrátí logger knihovny (s NullHandlerem, default ticho)."""
    log = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in log.handlers):
        log.addHandler(logging.NullHandler())
    return log


def set_debug(on):
    """Zapne/vypne ladicí výpis knihovny.

    Args:
        on (bool): True přidá StreamHandler + DEBUG; False ho odebere.
    """
    global _debug_handler
    log = get_logger()
    if on and _debug_handler is None:
        _debug_handler = logging.StreamHandler()
        _debug_handler.setFormatter(logging.Formatter("jellyai: %(message)s"))
        log.addHandler(_debug_handler)
        log.setLevel(logging.DEBUG)
    elif not on and _debug_handler is not None:
        log.removeHandler(_debug_handler)
        _debug_handler = None
        log.setLevel(logging.WARNING)
```

- [ ] **Step 5: Exportuj + spusť + commit**

V `__init__.py` přidej `from jellyai.errors import JellyError` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_errors_logs.py -q` → PASS.
```bash
git add jellyai/errors.py jellyai/logs.py jellyai/__init__.py tests/test_errors_logs.py
git commit -m "feat: JellyError (akční hlášky) + logování (NullHandler + debug)"
```

---

### Task 4: `CorpusTools` — start/stop + context manager

**Files:**
- Create: `jellyai/corpus.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_corpus_tools.py`

**Interfaces:**
- Produces: `CorpusTools` (podtřída `UfalClient`) s `start(*tools)`, `stop()`,
  `__enter__/__exit__`. `UfalClient` zůstává (zpětná kompatibilita).

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_corpus_tools.py
def test_corpus_tools_lifecycle_no_spawn():
    from config import ServicesConfig
    from jellyai.corpus import CorpusTools
    # bez volání parse/entities se žádná služba nespustí → stop je no-op
    tools = CorpusTools(ServicesConfig())
    tools.stop()                       # bez chyby
    with CorpusTools(ServicesConfig()) as t:
        assert t is not None           # context manager vrací self
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_corpus_tools.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj `jellyai/corpus.py`**

```python
"""Korpusové nástroje jako spravovatelná služba (start/stop).

Přehledné rozhraní nad ÚFAL službami (UDPipe/NameTag/MorphoDiTa): explicitní
životní cyklus (`start`/`stop`, context manager), aby bylo vidět, kdy služby žijí.
Vnitřně je to `UfalClient` (líný spawn + atexit) — jen s čitelnou správou.
"""

from jellyai.ufal_client import UfalClient

_PORTS = {"nametag": ("nametag_port", "nametag_model"),
          "udpipe": ("udpipe_port", "udpipe_model"),
          "morpho": ("morpho_port", "morphodita_model")}


class CorpusTools(UfalClient):
    """ÚFAL nástroje se start/stop a jako context manager."""

    def start(self, *tools):
        """Explicitně nastartuje uvedené nástroje (jinak se spustí líně).

        Args:
            *tools (str): „udpipe" / „nametag" / „morpho"; prázdné = všechny.
        """
        for name in (tools or _PORTS.keys()):
            port_attr, model_attr = _PORTS[name]
            self._ensure(name, getattr(self.config, port_attr),
                         getattr(self.config, model_attr))
        return self

    def stop(self):
        """Složí všechny běžící služby (alias `close`)."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.corpus import CorpusTools` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_corpus_tools.py -q` → PASS.
```bash
git add jellyai/corpus.py jellyai/__init__.py tests/test_corpus_tools.py
git commit -m "feat: CorpusTools — start/stop + context manager nad ÚFAL službami"
```

---

### Task 5: Teplota shody — `Retriever.search` + graf

**Files:**
- Modify: `jellyai/retriever.py`, `jellyai/answerer/graph_answerer.py`,
  `jellyai/answerer/base.py`
- Test: `tests/test_temperature.py`

**Interfaces:**
- Produces: `Retriever.search(query, top_k=None, *, temperature=0.0)`;
  `Answer.alternatives: list`; `GraphAnswerer` vrací primární + alternativy dle teploty.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_temperature.py
def test_retriever_temperature_widens_candidates():
    from config import RetrieverConfig
    from jellyai.chunker import Passage
    from jellyai.retriever import Retriever
    passages = [Passage("d", i, t, i, i + 1) for i, t in enumerate(
        ["robot robot robot", "robot pracuje", "moře je modré"])]
    r = Retriever(RetrieverConfig()).build(passages)
    tight = r.search("robot", temperature=0.0)
    wide = r.search("robot", temperature=1.0)
    assert len(wide) >= len(tight)
    assert len(wide) >= 2                      # široce pustí i slabší shodu


def test_graph_temperature_returns_alternatives():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                         Participant("num", "1890", "number")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                     Participant("num", "1915", "number")]))
    q = "kdy se narodil Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 2, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 11},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 12, "end": 17},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [], temperature=0.7)
    assert ans.text == "1890"
    assert "1915" in ans.alternatives          # alternativa dle teploty
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_temperature.py -q`
Expected: FAIL.

- [ ] **Step 3: Přidej `alternatives` do `Answer`**

V `jellyai/answerer/base.py`, do dataclass `Answer` přidej pole:
```python
    alternatives: list = field(default_factory=list)
```

- [ ] **Step 4: Teplota v `Retriever.search`**

V `jellyai/retriever.py` uprav `search` (za výpočet `scores`, místo ořezu top_k):
```python
        order = np.argsort(-scores)
        results = [(self.passages[i], float(scores[i])) for i in order if scores[i] > 0]
        if not results:
            return []
        if temperature and temperature > 0.0:
            threshold = results[0][1] * (1.0 - temperature)
            return [rs for rs in results if rs[1] >= threshold][:max(top_k, len(results))]
        return results[:top_k]
```
a doplň signaturu: `def search(self, query, top_k=None, *, temperature=0.0):` (+ docstring
řádek o teplotě).

- [ ] **Step 5: Teplota v `GraphAnswerer`**

V `_pick` přidej variantu, která vrátí i kandidáty; nejjednodušší je nový pomocník a
úprava `answer`:
```python
    def _rank(self, facts, role, temperature):
        """Seřazení kandidátů dané role s prahem podle teploty."""
        cands = []
        for fact in facts:
            for v in self.graph.participants(fact, role):
                cands.append((fact.weight, v, fact))
        cands.sort(key=lambda c: -c[0])
        if not cands:
            return None, None, []
        top = cands[0][0]
        keep = [c for c in cands if c[0] >= top * (1.0 - temperature)] if temperature else cands[:1]
        alts = [c[1] for c in keep[1:]]
        return cands[0][1], cands[0][2], alts
```
V `answer(self, question, retrieved, *, temperature=0.0)` (doplň parametr) po získání
`(value, fact)` přes `_traverse`, když `temperature > 0`, dopočítej alternativy tímtéž
prahem z faktů, ze kterých `_traverse` vybíral, a nastav `Answer(..., alternatives=alts)`.
(Pro jednoduchost lze `_traverse` nechat pro primární a alternativy dopočítat z
`facts_of(topic, role="subj", predicate=verb)` odpovídající roli otázky.)

Minimalistická varianta pro test: v `answer`, pro `qtype=="Kdy"`, po nalezení primární
hodnoty:
```python
        alternatives = []
        if temperature and value is not None:
            facts = g.facts_of(topic, role="subj", predicate=verb)
            _, _, alternatives = self._rank(facts, "num", temperature)
```
a vrať `Answer(text=value, sources=["graf"], score=1.0, alternatives=alternatives)`.

- [ ] **Step 6: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_temperature.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add jellyai/retriever.py jellyai/answerer/graph_answerer.py jellyai/answerer/base.py tests/test_temperature.py
git commit -m "feat: teplota shody — Retriever.search + graf vrací primární + alternativy"
```

---

### Task 6: `Composer` — výchozí šablonový kompozitor

**Files:**
- Create: `jellyai/answerer/composer.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_composer.py`

**Interfaces:**
- Produces: `TemplateComposer.compose(question, facts) -> str`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_composer.py
def test_template_composer_joins_facts():
    from jellyai.answerer.composer import TemplateComposer
    from jellyai.graph.extract import make_fact, Participant
    facts = [
        make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                              Participant("num", "1890", "number"),
                              Participant("loc", "Malých Svatoňovicích", "geo")]),
        make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                             Participant("obj", "R.U.R.", "concept")]),
    ]
    text = TemplateComposer().compose("kdo je Karel Čapek?", facts)
    assert "Karel Čapek" in text and "1890" in text and "R.U.R." in text
    assert text.strip().endswith(".")
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_composer.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj `jellyai/answerer/composer.py`**

```python
"""Kompozitor — ze sady faktů složí čitelný text (víc než jednoslovná odpověď).

Port `Composer` je místo, kam později sedne malá generativní NN. Výchozí je
šablonový: fakty seskupí podle podmětu a poskládá jednoduché věty. Bere kandidátní
fakty (třeba z teploty shody) a vrátí souvislý odstavec.
"""

_ROLE_PHRASE = {"num": "roku {}", "time": "{}", "loc": "v {}", "obj": "{}",
                "pred": "{}"}


class TemplateComposer:
    """Výchozí kompozitor — spojí fakty do vět bez modelu."""

    def compose(self, question, facts):
        """Složí čitelný text ze sady faktů.

        Args:
            question (str): Původní otázka (pro budoucí kontext; zde neužito).
            facts (list[Fact]): Kandidátní fakty.

        Returns:
            str: Souvislý text (prázdný, když nejsou fakty).
        """
        sentences = []
        for fact in facts:
            parts = {p.role: p.node for p in fact.participants}
            subj = parts.get("subj", "")
            tail = []
            for role in ("obj", "num", "time", "loc", "pred"):
                if role in parts:
                    tail.append(_ROLE_PHRASE[role].format(parts[role]))
            if subj and tail:
                sentences.append(f"{subj} {fact.predicate} {' '.join(tail)}".strip())
        text = ". ".join(s[0].upper() + s[1:] for s in sentences)
        return (text + ".") if text else ""
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.answerer.composer import TemplateComposer` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_composer.py -q` → PASS.
```bash
git add jellyai/answerer/composer.py jellyai/__init__.py tests/test_composer.py
git commit -m "feat: TemplateComposer — sada faktů → čitelný text (default port)"
```

---

### Task 7: Vysvětlitelná odpověď — `Answer.explain()`

**Files:**
- Modify: `jellyai/answerer/base.py`, `jellyai/answerer/graph_answerer.py`
- Test: `tests/test_explainable.py`

**Interfaces:**
- Produces: `Answer.trace: dict | None`, `Answer.explain() -> str`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_explainable.py
def test_graph_answer_explains_path():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    ans = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig())).answer(q, [])
    ex = ans.explain()
    assert "Babička" in ex and "napsat" in ex and "Božena Němcová" in ex
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_explainable.py -q`
Expected: FAIL (`explain` neexistuje).

- [ ] **Step 3: Přidej `trace` + `explain()` do `Answer`**

V `jellyai/answerer/base.py`, do `Answer`:
```python
    trace: dict = None
```
a metodu:
```python
    def explain(self):
        """Lidský popis, jak odpověď vznikla (trasa grafu), nebo prostý text.

        Returns:
            str: Např. „Babička ← napsat ← Božena Němcová", jinak text odpovědi.
        """
        if not self.trace:
            return self.text
        t = self.trace
        return f"{t.get('topic')} ← {t.get('predicate')} ← {t.get('answer')}"
```

- [ ] **Step 4: `GraphAnswerer` naplní `trace`**

V `answer`, kde se vrací grafová odpověď, předej trasu:
```python
                return Answer(text=value, sources=["graf"], score=1.0,
                              alternatives=alternatives if 'alternatives' in dir() else [],
                              trace=self.last_trace)
```
(Pokud alternativy nejsou v této větvi, vynech je; podstatné je `trace=self.last_trace`.)

- [ ] **Step 5: Spusť — musí projít + celá sada**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jellyai/answerer/base.py jellyai/answerer/graph_answerer.py tests/test_explainable.py
git commit -m "feat: vysvětlitelná odpověď — Answer.trace + explain()"
```

---

### Task 8: Konfigurace jako JSON

**Files:**
- Modify: `config.py`
- Test: `tests/test_config_json.py`

**Interfaces:**
- Produces: `Config.to_json(path) -> str`, `Config.from_json(path) -> Config`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_config_json.py
def test_config_json_roundtrip(tmp_path):
    from config import Config
    cfg = Config()
    cfg.retriever.method = "tfidf"
    path = str(tmp_path / "config.json")
    cfg.to_json(path)
    loaded = Config.from_json(path)
    assert loaded.retriever.method == "tfidf"
    assert loaded.chunker.size == cfg.chunker.size
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_config_json.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementuj `to_json`/`from_json`**

V `config.py`, do třídy `Config` (metody využijí `dataclasses.asdict` + rekonstrukci):
```python
    def to_json(self, path):
        """Uloží konfiguraci jako čitelný JSON (všechny knoflíky na jednom místě)."""
        import json
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_json(cls, path):
        """Načte konfiguraci z JSON (chybějící pole doplní výchozími)."""
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        for block, values in data.items():
            target = getattr(cfg, block, None)
            if target is None or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
        return cfg
```
(Pozn.: n-tice v JSON přijdou jako seznamy; pro Fázi 1 to nevadí — čtou se jako
sekvence. Případné přetypování na tuple je pozdější drobnost.)

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_config_json.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_json.py
git commit -m "feat: Config.to_json/from_json — čitelná konfigurace (JSON)"
```

---

### Task 9: Perzistence session (JSON) — `ActivationField` + save/load

**Files:**
- Modify: `jellyai/graph/activation.py`
- Create: `jellyai/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `ActivationField.to_dict()/from_dict(d)`; `save_session(name, answerer, graph_path, dir)`
  a `load_session(name, answerer, dir)` (JSON `data/sessions/<name>.json`).

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_session.py
def test_activation_field_json_roundtrip():
    from jellyai.graph.activation import ActivationField
    f = ActivationField(); f.warm("A", 1.5); f.warm("B", 0.5)
    g = ActivationField.from_dict(f.to_dict())
    assert g.hottest() == "A"


def test_session_save_load_continues_weights(tmp_path):
    from jellyai.session import save_session, load_session
    from jellyai.graph.activation import ActivationField

    class Dummy:
        def __init__(self):
            self.context = ActivationField(); self.history = []
    a = Dummy(); a.context.warm("Božena Němcová", 2.0)
    a.history.append({"question": "…", "topic": "Božena Němcová", "answer": "1818"})
    save_session("test", a, graph_path="data/graph.pkl", directory=str(tmp_path))

    b = Dummy()
    load_session("test", b, directory=str(tmp_path))
    assert b.context.hottest() == "Božena Němcová"     # pokračuje od vah
    assert b.history[-1]["answer"] == "1818"
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_session.py -q`
Expected: FAIL.

- [ ] **Step 3: `ActivationField` JSON**

V `jellyai/graph/activation.py`, do třídy:
```python
    def to_dict(self):
        """Serializuje váhy do prostého dictu (JSON-friendly)."""
        return {"decay": self.decay, "floor": self.floor, "scores": dict(self.scores)}

    @classmethod
    def from_dict(cls, data):
        """Obnoví pole z dictu (viz `to_dict`)."""
        field = cls(decay=data.get("decay", 0.55), floor=data.get("floor", 1e-3))
        field.scores = dict(data.get("scores", {}))
        return field
```

- [ ] **Step 4: Implementuj `jellyai/session.py`**

```python
"""Perzistence pojmenované konverzace — JSON (čitelné pro uživatele).

Stav rozhovoru = váhy těžiště (`ActivationField`) + historie + odkaz na graf. Uloží
se jako `data/sessions/<name>.json`, takže se dá pojmenovat, načíst a **pokračovat
od posledních vah**. Graf zůstává zvlášť (velký, numpy) — session ho jen referuje.
"""

import json
import os

from jellyai.graph.activation import ActivationField

_DIR = "data/sessions"


def _path(name, directory):
    return os.path.join(directory or _DIR, f"{name}.json")


def save_session(name, answerer, graph_path=None, directory=None):
    """Uloží stav konverzace answereru do JSON.

    Args:
        name (str): Jméno session.
        answerer: Objekt s `.context` (ActivationField) a `.history`.
        graph_path (str | None): Cesta ke grafu, který session používá.
        directory (str | None): Cílový adresář (default data/sessions).

    Returns:
        str: Cesta k uloženému souboru.
    """
    path = _path(name, directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"name": name, "graph_path": graph_path,
            "weights": answerer.context.to_dict(), "history": answerer.history}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_session(name, answerer, directory=None):
    """Načte stav konverzace do answereru (pokračuje od posledních vah).

    Args:
        name (str): Jméno session.
        answerer: Objekt s `.context` a `.history` (přepíší se).
        directory (str | None): Adresář (default data/sessions).

    Returns:
        dict: Načtená data session (vč. `graph_path`).
    """
    with open(_path(name, directory), encoding="utf-8") as f:
        data = json.load(f)
    answerer.context = ActivationField.from_dict(data["weights"])
    answerer.history = data["history"]
    return data
```

- [ ] **Step 5: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_session.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jellyai/graph/activation.py jellyai/session.py tests/test_session.py
git commit -m "feat: perzistence session (JSON) — váhy těžiště + historie, pokračování"
```

---

### Task 10: Fasáda `Jelly` (composition root + DI + korpus + debug)

**Files:**
- Create: `jellyai/facade.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_facade.py`

**Interfaces:**
- Produces: `Jelly(config=None, *, debug=False, retriever=None, answerer=None, corpus=None)`
  s `ask`, `save_session`, `load_session`, `gravity`, `trajectory`, `reset`, `close`,
  `__enter__/__exit__`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_facade.py
def test_jelly_injects_answerer_and_answers():
    from jellyai.facade import Jelly
    from jellyai.answerer.base import Answer

    class FakeAnswerer:
        def __init__(self): self.context = None; self.history = []
        def answer(self, q, retrieved, **kw):
            return Answer(text="Božena Němcová", sources=["graf"], score=1.0)
    j = Jelly(answerer=FakeAnswerer())
    ans = j.ask("kdo napsal Babičku?")
    assert ans.text == "Božena Němcová"


def test_jelly_context_manager_closes_corpus():
    from jellyai.facade import Jelly
    closed = {"v": False}

    class FakeCorpus:
        def stop(self): closed["v"] = True
    with Jelly(corpus=FakeCorpus()):
        pass
    assert closed["v"] is True
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_facade.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementuj `jellyai/facade.py`**

```python
"""Fasáda Jelly — tenký „composition root" nad porty.

Není to monolit: složí **výchozí** bloky (retriever, korpus, answerer) a nechá je
**injektovat/vyměnit** (i za NN). Vlastní životní cyklus korpusu, umí debug a
pojmenované session. Kdo chce vnitřek, jde granulárně přes bloky/porty.
"""

from jellyai.logs import get_logger, set_debug as _set_debug
from jellyai.session import save_session, load_session


class Jelly:
    """Rychlá cesta i composition root pro skládačku portů."""

    def __init__(self, config=None, *, debug=False, retriever=None,
                 answerer=None, corpus=None):
        """Vytvoří fasádu s výchozími nebo injektovanými bloky.

        Args:
            config (Config | None): Konfigurace (default `Config()`).
            debug (bool): Zapne ladicí výpis.
            retriever/answerer/corpus: Injektované porty (None = výchozí, dolní bloky).
        """
        from config import Config
        self.config = config or Config()
        self.log = get_logger()
        self.set_debug(debug)
        self._retriever = retriever
        self._answerer = answerer
        self.corpus = corpus                      # None = líné vytvoření při potřebě
        self._graph = None

    def set_debug(self, on):
        """Zapne/vypne ladicí výpis knihovny."""
        self.debug = on
        _set_debug(on)

    def ask(self, question, *, debug=None, temperature=0.0):
        """Odpoví na dotaz vybraným answererem; loguje, když debug.

        Args:
            question (str): Dotaz.
            debug (bool | None): Přebije globální debug pro tento dotaz.
            temperature (float): Teplota shody (0 = nejlepší, 1 = široce).

        Returns:
            Answer: Odpověď (s trasou a případnými alternativami).
        """
        if debug is not None:
            self.set_debug(debug)
        self.log.debug("otázka: %s (temperature=%s)", question, temperature)
        answerer = self._require_answerer()
        try:
            return answerer.answer(question, [], temperature=temperature)
        except TypeError:
            return answerer.answer(question, [])   # answerer bez temperature

    def _require_answerer(self):
        """Vrátí answerer (injektovaný, nebo srozumitelná chyba)."""
        if self._answerer is None:
            from jellyai.errors import JellyError
            raise JellyError("Answerer není nastavený — předej `Jelly(answerer=…)` "
                             "nebo použij granulární bloky.")
        return self._answerer

    def gravity(self):
        """Aktuální těžiště konverzace (nejteplejší uzel), nebo None."""
        ctx = getattr(self._answerer, "context", None)
        return ctx.hottest() if ctx is not None else None

    def trajectory(self):
        """Historie konverzace (trajektorie těžiště)."""
        return list(getattr(self._answerer, "history", []))

    def reset(self):
        """Začne nový rozhovor (vymaže těžiště a historii answereru)."""
        if hasattr(self._answerer, "reset"):
            self._answerer.reset()

    def save_session(self, name):
        """Uloží pojmenovanou session (váhy + historie)."""
        return save_session(name, self._answerer,
                            graph_path=self.config.graph.graph_path)

    def load_session(self, name):
        """Načte pojmenovanou session (pokračuje od posledních vah)."""
        return load_session(name, self._answerer)

    def close(self):
        """Složí korpusové služby, pokud běží."""
        if self.corpus is not None and hasattr(self.corpus, "stop"):
            self.corpus.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.facade import Jelly` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_facade.py -q` → PASS.
```bash
git add jellyai/facade.py jellyai/__init__.py tests/test_facade.py
git commit -m "feat: fasáda Jelly — composition root + DI + korpus lifecycle + session"
```

---

### Task 11: `demo()` — zero-setup první spuštění

**Files:**
- Create: `jellyai/demo.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Produces: `demo() -> dict[str, str]` (otázka → odpověď), bez modelů.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_demo.py
def test_demo_runs_without_models():
    import jellyai
    result = jellyai.demo(verbose=False)
    assert result["kdo napsal Babičku?"] == "Božena Němcová"
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_demo.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementuj `jellyai/demo.py`**

```python
"""demo() — první „ono to funguje" bez modelů a stahování.

Postaví malý vestavěný faktový graf a odpoví na pár otázek přes `GraphAnswerer`
s nakonzervovaným rozborem otázek (`FakeUfalClient`). Slouží jako rychlá ukázka pro
rookie: `import jellyai; jellyai.demo()`.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient

_Q1 = "kdo napsal Babičku?"
_Q2 = "kdy se narodila?"


def _graph():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("num", "1818", "number")]))
    return g


def _client():
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
    answerer = GraphAnswerer(_graph(), _client(), ExtractiveAnswerer(AnswererConfig()))
    out = {}
    for q in (_Q1, _Q2):
        ans = answerer.answer(q, [])
        out[q] = ans.text
        if verbose:
            print(f"❓ {q}\n💬 {ans.text}\n")
    return out
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.demo import demo` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_demo.py -q` → PASS.
```bash
git add jellyai/demo.py jellyai/__init__.py tests/test_demo.py
git commit -m "feat: demo() — zero-setup ukázka bez modelů"
```

---

### Task 12: examples/ + viewBase odstínění + celá sada + docs

**Files:**
- Create: `examples/01_retrieval.py`, `examples/02_fact_graph.py`,
  `examples/03_corpus_tools.py`, `examples/04_conversation.py`,
  `examples/05_swap_a_block.py`
- Modify: `README.md`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Ověř odstínění viewBase (test)**

```python
# tests/test_examples.py
import sys


def test_import_jellyai_does_not_import_viewbase():
    for mod in list(sys.modules):
        if mod.startswith("viewbase"):
            del sys.modules[mod]
    import jellyai  # noqa: F401
    assert not any(m.startswith("viewbase") for m in sys.modules)
```

- [ ] **Step 2: Spusť — musí projít** (viewBase se importuje jen líně v export adaptéru)

Run: `.venv/bin/python -m pytest tests/test_examples.py -q`
Expected: PASS. (Pokud FAIL, uprav `jellyai/graph/viewbase_export.py`, ať `networkx`/
`viewbase` importuje až uvnitř funkcí — viz stávající stav.)

- [ ] **Step 3: Napiš runnable examples**

`examples/02_fact_graph.py` (bez modelů, přes demo/mini-graf):
```python
"""Postav malý faktový graf a ptej se (bez modelů)."""
import jellyai

result = jellyai.demo()          # mini-graf + ukázkové otázky
print("Hotovo:", result)
```
`examples/05_swap_a_block.py` (injektování vlastního portu do fasády):
```python
"""Vyměň answerer za vlastní implementaci portu (ukázka DI)."""
import jellyai
from jellyai.answerer.base import Answer


class YesAnswerer:
    context = None; history = []
    def answer(self, question, retrieved, **kw):
        return Answer(text="ano", sources=[], score=1.0)


jelly = jellyai.Jelly(answerer=YesAnswerer())
print(jelly.ask("funguje injektování?").text)   # → ano
```
(A analogicky `01_retrieval.py`, `03_corpus_tools.py`, `04_conversation.py` — retrieval
přes `Retriever`, korpus přes `CorpusTools` context manager, konverzace přes
`GraphAnswerer`.)

- [ ] **Step 4: Uprav README**

Přidej sekci „Knihovna" s `import jellyai`, `jellyai.demo()`, `Jelly`, `CorpusTools`,
portami a odkazem na `examples/`.

- [ ] **Step 5: Celá sada + commit**

Run: `.venv/bin/python -m pytest -q` → PASS.
```bash
git add examples/ tests/test_examples.py README.md
git commit -m "docs: examples/ + odstínění viewBase + README (knihovna)"
```

---

## Self-review (proti specu)

- §2/§3 porty → Task 2; teplota + Composer → Task 5, 6. ✓
- §4 fasáda Jelly (DI, korpus, debug, session) → Task 10. ✓
- §5 CorpusTools start/stop → Task 4. ✓
- §6 session JSON → Task 9; §7 config JSON → Task 8. ✓
- §8 vysvětlitelná odpověď → Task 7. ✓
- §9 logging + JellyError → Task 3. ✓
- §10 viewBase odstínění → Task 12. ✓
- §11 demo/examples/pyproject → Task 11, 12, 1. ✓
- §12 veřejné API → Task 1 (+ průběžně). ✓
- §13 testy → hermetické v každém úkolu. ✓

Typová konzistence: `Answer` získává `alternatives` (Task 5) i `trace`/`explain` (Task 7);
`GraphAnswerer.answer` má `temperature` (Task 5) a plní `trace` (Task 7); `Jelly` používá
`session.save/load` (Task 9) a `logs`/`errors` (Task 3). ✓
