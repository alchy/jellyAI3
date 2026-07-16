# Fáze 2 — CLI→knihovna + viewBase propojení — Implementační plán

> **Pro agentní workery:** POVINNÁ SUB-SKILL: superpowers:executing-plans. Kroky mají
> checkboxy (`- [ ]`).

**Goal:** Přesunout orchestraci z `cli.py` do knihovních funkcí a přidat živé propojení
s viewBase přes port `GraphView` + adaptér `ViewBaseView` (líný, vlastní webserver
lifecycle), s reflexí těžiště a trasy. Dvojí režim: terminál / web.

**Architecture:** `jellyai/tasks.py` (orchestrace), rozšíření `jellyai/ports.py`
(`GraphView`), `jellyai/viz/` (`reflect.py`, `viewbase_view.py`). `cli.py` a nový
`./jelly web` volají knihovní funkce. Jádro `import jellyai` zůstává viewBase-free.

**Tech Stack:** Python 3.11/3.12, stdlib, numpy. viewBase jen líný import v adaptéru.

## Global Constraints

- Jádro **nesáhne** na viewBase (líný import jen v `viewbase_view.py`).
- Testy **hermetické** — přes `FakeView` (implementuje `GraphView`), bez viewBase.
- Bohaté české docstringy. `.venv/bin/python -m pytest`. Průběžně `pylint` (nové 10/10).
- Bez přesunů souborů. TDD, commit na úkol; message končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `jellyai/tasks.py` — orchestrace z CLI do knihovny

**Files:**
- Create: `jellyai/tasks.py`
- Modify: `cli.py` (`cmd_graph`, `cmd_annotate` volají tasks), `jellyai/__init__.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `annotate_corpus(config, client=None) -> int`, `build_fact_graph(config) -> FactGraph`,
  `load_fact_graph(config) -> FactGraph`, `make_graph_answerer(config) -> GraphAnswerer`.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_tasks.py
import pickle


def _write_annotations(path):
    ann = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                      "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    with open(path, "wb") as f:
        pickle.dump(ann, f)


def test_build_and_load_fact_graph(tmp_path):
    from config import Config, ServicesConfig, GraphConfig
    from jellyai.tasks import build_fact_graph, load_fact_graph
    ann_path = str(tmp_path / "ann.pkl"); graph_path = str(tmp_path / "graph.pkl")
    _write_annotations(ann_path)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=graph_path)
    graph = build_fact_graph(cfg)
    assert graph.facts_of("Božena Němcová", predicate="napsat")
    assert load_fact_graph(cfg).facts_of("Božena Němcová", predicate="napsat")
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_tasks.py -q`
Expected: FAIL (`ModuleNotFoundError: jellyai.tasks`).

- [ ] **Step 3: Implementuj `jellyai/tasks.py`**

```python
"""Orchestrace pipeline jako knihovní funkce — sdílí je CLI, web i vlastní programy.

Dřív tahle logika žila v tělech CLI příkazů; teď je v knihovně, takže terminál i web
volají totéž (žádná duplikace).
"""

from jellyai.loader import load_documents
from jellyai.annotate import annotate_documents, save_annotations, load_annotations
from jellyai.graph.graph import build_graph, FactGraph


def annotate_corpus(config, client=None):
    """Anotuje dokumenty po větách a uloží (entity + role).

    Args:
        config (Config): Konfigurace (processed_dir, services).
        client: ÚFAL klient; None = vytvoří `UfalClient`.

    Returns:
        int: Počet anotovaných vět.
    """
    documents = load_documents(config.data.processed_dir)
    own = client is None
    if own:
        from jellyai.ufal_client import UfalClient
        client = UfalClient(config.services)
    try:
        annotations = annotate_documents(documents, client)
    finally:
        if own:
            client.close()
    save_annotations(annotations, config.services.annotations_path)
    return len(annotations)


def build_fact_graph(config):
    """Postaví faktový graf z uložených anotací a uloží ho.

    Args:
        config (Config): Konfigurace (annotations_path, graph_path).

    Returns:
        FactGraph: Postavený graf.
    """
    annotations = load_annotations(config.services.annotations_path)
    graph = build_graph(annotations)
    graph.save(config.graph.graph_path)
    return graph


def load_fact_graph(config):
    """Načte dříve uložený faktový graf."""
    return FactGraph.load(config.graph.graph_path)


def make_graph_answerer(config):
    """Sestaví `GraphAnswerer` nad uloženým grafem (graf + klient + fallback).

    Args:
        config (Config): Konfigurace.

    Returns:
        GraphAnswerer: Připravený answerer.
    """
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.ufal_client import UfalClient
    graph = load_fact_graph(config)
    return GraphAnswerer(graph, UfalClient(config.services),
                         ExtractiveAnswerer(config.answerer))
```

- [ ] **Step 4: Přepoj `cli.py`**

V `cmd_graph` nahraď stavbu grafu voláním:
```python
    from jellyai.tasks import build_fact_graph
    graph = build_fact_graph(config)
    print(f"Faktový graf: {len(graph.nodes)} uzlů, {len(graph.facts)} faktů "
          f"→ {config.graph.graph_path}")
    ...
    return len(graph.nodes)
```
V `cmd_annotate` nahraď tělo voláním `annotate_corpus(config, client)` a vypiš počet.

- [ ] **Step 5: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.tasks import (annotate_corpus, build_fact_graph, load_fact_graph, make_graph_answerer)` (+ `__all__`).
Run: `.venv/bin/python -m pytest -q` → PASS. `pylint jellyai/tasks.py` → 10/10.
```bash
git add jellyai/tasks.py cli.py jellyai/__init__.py tests/test_tasks.py
git commit -m "feat: jellyai/tasks — orchestrace z CLI do knihovny (sdílí terminál/web)"
```

---

### Task 2: Port `GraphView`

**Files:**
- Modify: `jellyai/ports.py`, `jellyai/__init__.py`
- Test: `tests/test_ports.py`

- [ ] **Step 1: Napiš padající test**

Přidej do `tests/test_ports.py`:
```python
def test_graphview_port_structural():
    from jellyai.ports import GraphView

    class FakeView:
        def add_node(self, node_id, **meta): pass
        def add_edge(self, src, dst, **meta): pass
        def update_node(self, node_id, **attrs): pass
        def flow(self, path): pass
        def on_prompt(self, callback): pass
        def serve(self, open_browser=True): pass
        def stop(self): pass
    assert isinstance(FakeView(), GraphView)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_ports.py::test_graphview_port_structural -q`
Expected: FAIL.

- [ ] **Step 3: Přidej `GraphView` do `jellyai/ports.py`**

```python
@runtime_checkable
class GraphView(Protocol):
    """Vizualizace grafu — build/modify v kódu + prompt pro interakci.

    Abstrakce nad grafovým UI (výchozí adaptér: viewBase). Jádro ji zná, ale žádný
    konkrétní backend neimportuje.
    """
    def add_node(self, node_id, **meta): ...
    def add_edge(self, src, dst, **meta): ...
    def update_node(self, node_id, **attrs): ...   # barva/velikost/label živě
    def flow(self, path): ...                       # animace po hranách (trasa)
    def on_prompt(self, callback): ...              # prompt(text) → callback
    def serve(self, open_browser=True): ...         # nastartuje webserver
    def stop(self): ...                             # složí webserver
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: přidej `GraphView` k importu z `jellyai.ports` a do `__all__`.
Run: `.venv/bin/python -m pytest tests/test_ports.py -q` → PASS.
```bash
git add jellyai/ports.py jellyai/__init__.py tests/test_ports.py
git commit -m "feat: port GraphView (abstrakce grafové vizualizace, viewBase-free)"
```

---

### Task 3: `reflect` — živé propojení (těžiště + trasa)

**Files:**
- Create: `jellyai/viz/__init__.py`, `jellyai/viz/reflect.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_reflect.py`

**Interfaces:**
- Produces: `reflect(view, answerer)` — promítne aktivaci a trasu do view.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_reflect.py
def test_reflect_pushes_activation_and_flow():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    from jellyai.viz.reflect import reflect

    class FakeView:
        def __init__(self): self.updated = {}; self.flows = []
        def add_node(self, node_id, **meta): pass
        def add_edge(self, src, dst, **meta): pass
        def update_node(self, node_id, **attrs): self.updated[node_id] = attrs
        def flow(self, path): self.flows.append(path)
        def on_prompt(self, callback): pass
        def serve(self, open_browser=True): pass
        def stop(self): pass

    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.answer(q, [])                       # rozsvítí těžiště + nastaví trasu
    view = FakeView()
    reflect(view, a)
    assert "Božena Němcová" in view.updated          # aktivace nodu
    assert view.flows and "Babička" in view.flows[0] and "Božena Němcová" in view.flows[0]
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_reflect.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj `jellyai/viz/reflect.py`** (a prázdný `jellyai/viz/__init__.py`)

```python
"""Živé propojení answereru s grafovou vizualizací (GraphView).

Po každém dotazu promítne stav answereru do view: **aktivaci nodů** (konverzační
těžiště → velikost/barva) a **trasu** dotazu (téma → hodnota) jako `flow` po hranách.
Uživatel tak vidí, jak se graf rozsvěcuje a jak dotaz běží.
"""


def _trace_path(trace):
    """Z trasy poslední odpovědi sestaví cestu uzlů pro `flow` (téma → hodnota)."""
    path = [trace.get("topic"), trace.get("answer")]
    return [node for node in path if node]


def reflect(view, answerer):
    """Promítne stav answereru (těžiště + trasu) do grafové vizualizace.

    Args:
        view (GraphView): Cílová vizualizace.
        answerer: Objekt s `.context` (ActivationField) a `.last_trace`.
    """
    context = getattr(answerer, "context", None)
    if context is not None:
        for node_id, weight in context.scores.items():
            view.update_node(node_id, size=1.0 + weight)
    trace = getattr(answerer, "last_trace", None)
    if trace:
        path = _trace_path(trace)
        if len(path) >= 2:
            view.flow(path)
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.viz.reflect import reflect` (+ `__all__`).
Run: `.venv/bin/python -m pytest tests/test_reflect.py -q` → PASS. `pylint` → 10/10.
```bash
git add jellyai/viz/__init__.py jellyai/viz/reflect.py jellyai/__init__.py tests/test_reflect.py
git commit -m "feat: reflect — živé propojení (aktivace nodů + flow po trase)"
```

---

### Task 4: Adaptér `ViewBaseView` (líný, webserver lifecycle)

**Files:**
- Create: `jellyai/viz/viewbase_view.py`
- Modify: `jellyai/__init__.py`
- Test: `tests/test_viewbase_view.py`

**Interfaces:**
- Produces: `ViewBaseView(title=…)` implementující `GraphView` + `from_graph`, context manager.

- [ ] **Step 1: Napiš padající test**

```python
# tests/test_viewbase_view.py
import importlib.util
import sys


def test_module_import_does_not_import_viewbase():
    for m in list(sys.modules):
        if m.startswith("viewbase"):
            del sys.modules[m]
    import jellyai.viz.viewbase_view  # noqa: F401
    assert not any(m.startswith("viewbase") for m in sys.modules)


def test_missing_viewbase_raises_actionable():
    if importlib.util.find_spec("viewbase") is not None:
        return  # viewBase je nainstalovaný → tenhle hermetický test přeskoč
    from jellyai.viz.viewbase_view import ViewBaseView
    from jellyai.errors import JellyError
    try:
        ViewBaseView()
        assert False, "mělo hodit JellyError"
    except JellyError as err:
        assert "viewbase" in str(err).lower()
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_viewbase_view.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementuj `jellyai/viz/viewbase_view.py`**

```python
"""Adaptér `GraphView` nad viewBase — jediné místo, kde se viewBase importuje.

Líný import (jádro zůstává viewBase-free). Vlastní **životní cyklus webserveru**:
`serve()` nastartuje, `stop()` složí; context manager. Prompt řeší přes viewBase
`ControlWindow` (textové pole + on_submit).
"""

from jellyai.errors import JellyError

_TYPE_COLOR = {"person": "ember", "geo": "green", "time": "gold",
               "number": "gold", "concept": "steel", "fact": "muted"}


class ViewBaseView:
    """viewBase adaptér: build/modify v kódu + prompt + živé aktualizace."""

    def __init__(self, title="jellyAI3"):
        """Vytvoří plátno viewBase (líný import).

        Raises:
            JellyError: Když viewBase není nainstalovaný (akční hláška).
        """
        try:
            import viewbase as vb
        except ImportError as exc:
            raise JellyError(
                "viewBase není nainstalovaný — nainstaluj ho: pip install viewbase "
                "(je volitelný, jádro jellyAI3 ho nepotřebuje).") from exc
        self._vb = vb
        self._canvas = vb.Canvas(title=title)
        self._handle = None

    def from_graph(self, graph):
        """Naplní plátno uzly a hranami faktového grafu (typované barvy).

        Args:
            graph (FactGraph): Zdrojový graf.

        Returns:
            ViewBaseView: self (pro řetězení).
        """
        for node in graph.nodes.values():
            self.add_node(node.id, label=node.id, kind=node.type)
        for (src, _relation, dst), _weight in graph.edges.items() if hasattr(graph, "edges") else []:
            self.add_edge(src, dst)
        return self

    def add_node(self, node_id, **meta):
        """Přidá uzel (barva podle `kind`)."""
        self._canvas.add_node(node_id, **meta)

    def add_edge(self, src, dst, **meta):
        """Přidá hranu."""
        self._canvas.add_edge(src, dst, **meta)

    def update_node(self, node_id, **attrs):
        """Živě změní uzel (velikost/barva/label)."""
        self._canvas.update_node(node_id, **attrs)

    def flow(self, path):
        """Animuje světelné částice po cestě uzlů (trasa dotazu)."""
        if len(path) >= 2:
            self._canvas.flow(path[0], path[-1], path=path)

    def on_prompt(self, callback):
        """Napojí textový prompt (ControlWindow) na callback(text)."""
        window = self._vb.ControlWindow()
        window.string("dotaz")
        self._canvas.open_window(window, on_submit=lambda values: callback(values["dotaz"]))

    def serve(self, open_browser=True):
        """Nastartuje webserver (neblokující); uloží handle."""
        self._handle = self._vb.serve(self._canvas, open_browser=open_browser, block=False)
        return self

    def stop(self):
        """Složí webserver."""
        if self._handle is not None:
            self._handle.stop()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
```

- [ ] **Step 4: Exportuj + spusť + commit**

V `__init__.py`: `from jellyai.viz.viewbase_view import ViewBaseView` (+ `__all__`).
**Pozn.:** import třídy je OK (viewBase se importuje až v `__init__`, ne při importu modulu).
Run: `.venv/bin/python -m pytest tests/test_viewbase_view.py tests/test_examples.py -q` → PASS
(separace drží). `pylint` → 10/10.
```bash
git add jellyai/viz/viewbase_view.py jellyai/__init__.py tests/test_viewbase_view.py
git commit -m "feat: ViewBaseView — adaptér GraphView nad viewBase (líný, webserver lifecycle)"
```

---

### Task 5: Dvojí režim — `./jelly web` + example + README + merge

**Files:**
- Modify: `cli.py` (`cmd_web` + registrace), `jelly` (wrapper), `README.md`
- Create: `examples/06_web.py`
- Test: `tests/test_cli_web.py`

**Interfaces:**
- Produces: `cmd_web(config, view=None) -> None` — sestaví answerer + view, napojí prompt.

- [ ] **Step 1: Napiš padající test** (přes `FakeView`, bez viewBase)

```python
# tests/test_cli_web.py
import pickle


def test_cmd_web_wires_prompt_to_ask(tmp_path, monkeypatch):
    from config import Config, ServicesConfig, GraphConfig
    from cli import cmd_web
    # připrav graf
    graph_path = str(tmp_path / "graph.pkl")
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.save(graph_path)
    cfg = Config()
    cfg.graph = GraphConfig(graph_path=graph_path)

    # FakeView zaznamenává prompt callback a serve
    class FakeView:
        def __init__(self): self.cb = None; self.served = False; self.updated = {}; self.flows = []
        def from_graph(self, graph): return self
        def add_node(self, *a, **k): pass
        def add_edge(self, *a, **k): pass
        def update_node(self, node_id, **attrs): self.updated[node_id] = attrs
        def flow(self, path): self.flows.append(path)
        def on_prompt(self, callback): self.cb = callback
        def serve(self, open_browser=True): self.served = True
        def stop(self): pass

    # klienta answereru zfalšujeme, aby ask fungoval bez modelů
    from jellyai.ufal_client import FakeUfalClient
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    monkeypatch.setattr("jellyai.tasks.UfalClient", lambda services: client, raising=False)

    view = FakeView()
    cmd_web(cfg, view=view)
    assert view.served is True and view.cb is not None
    view.cb(q)                                   # simuluj dotaz z promptu
    assert "Božena Němcová" in view.updated      # ask + reflect proběhly
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_cli_web.py -q`
Expected: FAIL (`cmd_web` neexistuje).

- [ ] **Step 3: Přidej `cmd_web` do `cli.py`**

```python
def cmd_web(config, view=None):
    """Spustí webovou vizualizaci: graf ve viewBase + prompt pro dotazy.

    Terminál i web volají tutéž `jelly.ask`. `view` lze injektovat (testy/vlastní UI);
    None = výchozí `ViewBaseView` nad uloženým grafem.

    Args:
        config (Config): Konfigurace (graf, služby).
        view: Injektovaný GraphView (None = ViewBaseView).
    """
    from jellyai.tasks import make_graph_answerer, load_fact_graph
    from jellyai.viz.reflect import reflect
    answerer = make_graph_answerer(config)
    if view is None:
        from jellyai.viz.viewbase_view import ViewBaseView
        view = ViewBaseView("jellyAI3").from_graph(load_fact_graph(config))

    def on_query(question):
        answer = answerer.answer(question, [])
        reflect(view, answerer)
        print(f"💬 {answer.text}")
        return answer.text

    view.on_prompt(on_query)
    view.serve(open_browser=True)
```

- [ ] **Step 4: Registrace v CLI + wrapper**

V `_build_parser` přidej `sub.add_parser("web", parents=[common], help="webová vizualizace grafu (viewBase)")`.
V `main` dispatch: `elif args.command == "web": cmd_web(config)`.
Ve `jelly` wrapperu přidej `web` do vnějšího i vnitřního `case` (`web) "$PY" "$ROOT/cli.py" web ;;`).

- [ ] **Step 5: Napiš example**

`examples/06_web.py`:
```python
"""06 — Web: graf ve viewBase + prompt (potřebuje pip install viewbase + ./jelly graph)."""
from config import Config
from cli import cmd_web

cmd_web(Config())   # nahoře graf, dole prompt; dotazy rozsvěcují graf
```

- [ ] **Step 6: README + celá sada + commit**

Doplň do README sekce „Knihovna" řádek o `./jelly web` a `ViewBaseView`.
Run: `.venv/bin/python -m pytest -q` → PASS. `pylint jellyai/ cli.py` (nové 10/10).
```bash
git add cli.py jelly examples/06_web.py README.md tests/test_cli_web.py
git commit -m "feat: ./jelly web + example 06 — dvojí režim (terminál/web) nad toutéž ask"
```

---

## Self-review (proti specu)

- §3 CLI→knihovna → Task 1. ✓
- §4 port GraphView → Task 2. ✓
- §5 ViewBaseView (líný, lifecycle) → Task 4. ✓
- §6 reflect (těžiště + trasa) → Task 3. ✓
- §7 dvojí režim (web + example) → Task 5. ✓
- §8 separace (import nesáhne na viewBase) → Task 4 test + stávající. ✓
- §9 testy (FakeView, líný, port) → v každém úkolu. ✓

Typová konzistence: `reflect(view, answerer)` konzumuje `GraphView` (Task 2) a
`answerer.context`/`last_trace`; `cmd_web` používá `make_graph_answerer` (Task 1),
`reflect` (Task 3), `ViewBaseView` (Task 4). ✓
