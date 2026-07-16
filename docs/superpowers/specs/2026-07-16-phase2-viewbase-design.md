# Fáze 2 — CLI→knihovna + živé propojení s viewBase (návrh)

**Datum:** 2026-07-16 · **Větev:** `feature/phase2-viewbase`
**Status:** Schváleno (design), čeká na spec-review

## 1. Cíl a kontext

Cílové propojení **viewBase ↔ jellyAI3**: web s grafem nahoře (viewBase) a promptem
dole; při dotazu se **živě** ukazuje, jak se graf rozsvěcuje (těžiště) a jak dotaz
běží (trasa). Zároveň jde graf **stavět/modifikovat v kódu** — prompt je pro
interakci. Předpoklad: **logika z `cli.py` se přesune do knihovních funkcí**, aby
terminál i web volaly totéž.

Vše zůstává v duchu skládačky: **viewBase je další injektovaný port**, jádro
viewBase-free (líný import, žádná tvrdá závislost). Jelly vlastní životní cykly
(korpus i webserver).

## 2. Klíčová rozhodnutí (schválená)

- **viewBase = port `GraphView`** (úzké rozhraní); adaptér `ViewBaseView` (líný import).
- **Adaptér vlastní webserver lifecycle** — `serve()` nastartuje (`vb.serve(block=False)
  → ServerHandle`), `stop()` složí; context manager; **jelly to řídí** (`jelly.close()`
  složí korpus i view).
- **Prompt přes `ControlWindow`** (string pole + `on_submit`) — první verze bez zásahu
  do viewBase; spodní console lišta = pozdější rozšíření viewBase (abstrakce ho skryje).
- **Živé propojení** — po `ask` pushnout aktivaci nodů (těžiště) přes `update_node` a
  `flow` po trase (`last_trace`).
- **Dvojí režim** z jednoho příkladu: terminálový REPL / web.
- **CLI→knihovna** — orchestrace (graf/anotace) jako knihovní funkce; `cli.py` tenký.
- Bez fyzického přeskupení souborů (to je Fáze 3).

## 3. CLI → knihovní funkce (`jellyai/tasks.py`)

Přesun těla CLI příkazů do knihovny, ať je sdílí terminál i web:
```python
def annotate_corpus(config, client=None) -> int          # dokumenty → větné anotace (uloží)
def build_fact_graph(config) -> FactGraph                 # anotace → graf (uloží)
def load_fact_graph(config) -> FactGraph                  # načte uložený graf
def make_graph_answerer(config) -> GraphAnswerer          # graf + klient + fallback
```
`cli.py` a web app pak jen volají tyto funkce (žádná duplikace logiky).

## 4. Port `GraphView` (`jellyai/ports.py`)

```python
@runtime_checkable
class GraphView(Protocol):
    def add_node(self, node_id, **meta): ...
    def add_edge(self, src, dst, **meta): ...
    def update_node(self, node_id, **attrs): ...   # barva/velikost/label živě
    def flow(self, path): ...                       # animace po hranách (trasa)
    def on_prompt(self, callback): ...              # prompt(text) → callback
    def serve(self, open_browser=True): ...         # nastartuje webserver
    def stop(self): ...                             # složí webserver
```
Rozhraní je viewBase-free — jádro ho zná, ale viewBase neimportuje.

## 5. Adaptér `ViewBaseView` (`jellyai/viz/viewbase_view.py`)

- **Líný import** `viewbase`; chybí-li → `JellyError` s akční hláškou (`pip install viewbase`).
- `__init__(title=…)` vytvoří `vb.Canvas`; `from_graph(fact_graph)` naplní uzly/hrany
  (typované barvy: person/geo/time/fact…) — využije stávající `to_json`/`to_networkx`.
- `update_node` / `flow` / `on_prompt` (ControlWindow `string` + `on_submit`) mapuje na
  viewBase; `serve(block=False) → ServerHandle` (uloží port), `stop()` = `handle.stop()`.
- **Context manager** (`__enter__/__exit__`) → čistý start i úklid.

## 6. Živé propojení (`jellyai/viz/reflect.py`)

```python
def reflect(view, answerer):
    """Promítne stav answereru do view: aktivace nodů (těžiště) + flow po trase."""
    for node_id, weight in answerer.context.scores.items():
        view.update_node(node_id, size=1 + weight, color="hot")
    if answerer.last_trace:
        view.flow(_trace_path(answerer.last_trace))   # téma → fakt → hodnota
```
Volá se po každém `ask`. Uživatel vidí, jak graf reaguje.

## 7. Dvojí režim (`examples/06_web.py` + CLI `web`)

```python
jelly = jellyai.Jelly(answerer=make_graph_answerer(config))
if web:
    with jellyai.ViewBaseView("jellyAI3").from_graph(graph) as view:
        view.on_prompt(lambda q: (reflect(view, jelly._answerer),
                                  print(jelly.ask(q).text)))
        view.serve(open_browser=True)     # nahoře graf, dole ControlWindow prompt
else:
    # terminálový REPL nad toutéž jelly.ask
    ...
```
Terminál i web volají **tutéž** `jelly.ask`. `./jelly web` = webová varianta.

## 8. Separace a životní cykly

- Jádro `import jellyai` **nesáhne** na viewBase/networkx (test to hlídá).
- `ViewBaseView` je jediné místo s viewBase (líný import).
- Jelly vlastní: `CorpusTools` (korpus) **i** `GraphView` (webserver). `jelly.close()`
  složí obojí; oba jako context manager.
- Volitelný flag `mirror_to_viewbase` (později) — operace nad grafem se navíc promítnou.

## 9. Testy (hermetické, bez viewBase)

- **CLI→knihovna:** `build_fact_graph(config)` z testovacích anotací vrátí graf;
  `cli.cmd_graph` volá tuto funkci (mock).
- **reflect:** s `FakeView` (implementuje port) ověř, že po `ask` zavolá `update_node`
  pro horké uzly a `flow` po trase.
- **Separace:** `import jellyai` neimportuje `viewbase` (už máme).
- **ViewBaseView líný:** bez nainstalovaného viewBase → `JellyError` s akční hláškou
  (import se stane až v `__init__`/`serve`, ne při importu modulu).
- **GraphView port:** `FakeView` je `isinstance(..., GraphView)`.

## 10. Rozsah / mimo rozsah

- **Fáze 2 (tento spec):** CLI→knihovna, port `GraphView` + `ViewBaseView` (lifecycle),
  `reflect`, dvojí režim (example + `./jelly web`).
- **Mimo (další kroky):** spodní console lišta jako rozšíření viewBase (upstream);
  backendová fyzika ve viewBase; `docs/guide/` vedený tutoriál (samostatný spec, artefaktový
  styl); flag `mirror_to_viewbase`.
