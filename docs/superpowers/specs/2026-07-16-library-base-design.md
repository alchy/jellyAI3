# jellyAI3 jako výuková knihovna — základ (návrh)

**Datum:** 2026-07-16 · **Větev:** `feature/library-base`
**Status:** Schváleno (design), čeká na spec-review

## 1. Cíl a kontext

Přerod jellyAI3 z „kódu v repu" na **výukovou knihovnu** ve stylu viewBase: píšeš
program nad `import jellyai` a skládáš **malé granulární bloky**. Jen víc komentovanou
(bohaté české docstringy). Cíl je **pevný, ale dál rozvíjitelný základ** — mj.
připravený na pozdější **zapojení neuronových sítí** bez přepisu jádra.

Zásadní posun oproti první úvaze: **primární abstrakce nejsou velké fasády, ale malé
porty** (úzká rozhraní). Fasáda `Jelly` je jen tenký „composition root" s výchozími
bloky a možností je **injektovat/vyměnit**.

## 2. Klíčová rozhodnutí (schválená)

- **Porty + injektování jako primární abstrakce.** Každá fáze = malé rozhraní
  (`typing.Protocol`); bloky se konstruují s **explicitními parametry**. Výměna jednoho
  dílu nechá zbytek stát. Každý port je místo, kam později sedne **NN**.
- **Tenká fasáda `Jelly`** — složí výchozí porty, umožní je nahradit (DI). Není
  monolit.
- **JSON > pickle**, kde to jde: perzistence session (váhy těžiště + historie),
  konfigurace. Čitelné pro uživatele.
- **Životní cyklus korpusu svázaný s knihovnou** — `CorpusTools` se instancuje a
  stopne s `Jelly`; explicitní `start()/stop()`.
- **Logování správně** (`logging` + `NullHandler`), `debug` přidá handler.
- **try/except + akční chybové hlášky** (`JellyError` říká, *jak to opravit*).
- **viewBase maximálně odstíněný a volitelný** — líný import, žádná tvrdá závislost,
  přikrytý wrapperem + flag „vše do viewBase".
- **Vysvětlitelná odpověď** — výsledek nese trasu (`explain()`).
- **`pip install`-ovatelné** (`pyproject.toml`), veřejné API v `__init__.py`.
- **Doku/plány v artefaktovém stylu** (uživatelova preference).
- **Bez fyzických přesunů souborů** v tomto základu (přeskupení = pozdější fáze).

## 3. Architektura — porty (skládačka)

Pipeline jako vyměnitelné porty. Výchozí je „od nuly"; každý port unese NN implementaci
(stejné rozhraní).

| Fáze | Port (`Protocol`) | Výchozí | NN hook |
|---|---|---|---|
| tokenizace | `Tokenizer.tokenize(text) -> list[str]` | regex + stopslova | subword/embeddings |
| retrieval | `Retriever.search(q) -> list[(Passage, float)]` | BM25 | embedding retriever |
| korpus | `CorpusPort.parse/entities/analyze/generate` | UDPipe/NameTag/MorphoDiTa | neuronový NER/parser |
| extrakce | `FactExtractor.extract(anotace) -> list[Fact]` | pravidla | neural relation extractor |
| analýza otázky | `QuestionAnalyzer.analyze(q) -> QuestionAnalysis` | lemma/pravidla | intent/slot klasifikátor |
| téma/koreference | `TopicResolver` nad `ActivationField` | aktivace | neuronová koreference |
| answerer | `Answerer.answer(q, retrieved) -> Answer` | extractive/graph | generativní NN |
| formulace | `Phraser.phrase(fact) -> str` | šablona/MorphoDiTa | malý generátor |

**Poznámka:** rozhraní jsou **strukturální** (`Protocol`), takže **stávající třídy je
už splňují** (Retriever, GraphAnswerer, …) — porty se přidají bez přepisu bloků.

## 4. Fasáda `Jelly` (composition root)

```python
class Jelly:
    def __init__(self, config=None, *, debug=False,
                 retriever=None, extractor=None, analyzer=None,
                 answerer=None, corpus=None):
        # None = výchozí port; jinak injektovaný (i NN)
        ...
    def set_debug(self, on: bool) -> None: ...
    def load(self, directory: str | None = None) -> "Jelly": ...   # docs → chunk → retriever
    def build_graph(self) -> "Jelly": ...                          # anotace (korpus) → FactGraph
    def ask(self, question: str, *, debug: bool | None = None) -> Answer: ...
    def save_session(self, name: str) -> str: ...                  # JSON
    @classmethod
    def load_session(cls, name: str, config=None) -> "Jelly": ...  # pokračuje od vah
    def reset(self) -> None: ...                                   # nová konverzace
    def gravity(self) -> str | None: ...                           # aktuální těžiště
    def trajectory(self) -> list[dict]: ...                        # historie
    def close(self) -> None: ...                                   # stop korpus
    def __enter__/__exit__: ...
```

Fasáda **vlastní** `CorpusTools` (lifecycle) a `GraphAnswerer` (těžiště/historie).
Každý port lze **injektovat** konstruktorem → NN i vlastní implementace.

## 5. `CorpusTools` — start/stop svázaný s knihovnou

```python
class CorpusTools:                # refaktor UfalClient (alias zpětně kompat.)
    def __init__(self, config=None): ...
    def start(self, *tools: str) -> None: ...   # explicitně; jinak lazy při 1. volání
    def stop(self) -> None: ...                 # složí služby
    def __enter__/__exit__: ...
    def parse/entities/analyze/generate(...)    # beze změny signatur
```

Instancuje se v `Jelly.__init__`; `Jelly.close()` volá `corpus.stop()`.

## 6. Perzistence session (JSON)

`data/sessions/<name>.json`:
```json
{"name": "capkovi", "graph_path": "data/graph.pkl",
 "weights": {"Božena Němcová": 1.1, "Babička": 0.55},
 "history": [{"question": "…", "topic": "…", "answer": "…", "gravity": "…"}]}
```
- `ActivationField.to_dict()/from_dict()` — serializace vah.
- `Jelly.save_session/load_session` — pokračuje od posledních vah.
- Graf zůstává (velký, numpy) jako `graph.pkl`; session ho jen referuje.

## 7. Konfigurace (JSON)

`Config.from_json(path)/to_json(path)` — jeden čitelný soubor se všemi knoflíky
(retriever/chunker/services/graph). Bloky ale berou i **explicitní parametry** (ne
nutně celý Config) — kvůli granularitě a injektování.

## 8. Vysvětlitelná odpověď

`Answer` získá volitelnou trasu; `answer.explain() -> str` složí lidský popis:
„Karel Čapek → narodit → 1890 (fakt s vahou 3)". `GraphAnswerer.last_trace` už trasu
má — jen se přenese na výsledek.

## 9. Logování a chyby

- `logging.getLogger("jellyai")` s `NullHandler` (default ticho). `debug=True` přidá
  `StreamHandler` a loguje kroky (načítám…, stavím graf N uzlů…, otázka → typ/téma/fakt).
- `class JellyError(Exception)` — základ; hlášky **říkají, jak to opravit** (např. chybí
  modely → „spusť ./jelly qa-models"). Operace v knihovně obalené `try/except`, které
  přidají srozumitelný kontext.

## 10. viewBase — odstínění

- `jellyai/viz/viewbase.py` (nebo stávající `graph/viewbase_export.py`): **líný import**
  viewBase/networkx, žádná tvrdá závislost. Bez viewBase knihovna plně funguje.
- Volitelný flag/adaptér: když `Jelly(mirror_to_viewbase=True)`, operace nad grafem se
  navíc pošlou do viewBase přes adaptér. Jinak se viewBase ani neimportuje.

## 11. demo(), examples/, pyproject

- `jellyai.demo()` — **zero-setup**: postaví malý vestavěný graf (bez modelů/stahování)
  a odpoví na pár otázek. První „ono to funguje" pro rookie.
- `examples/`: `01_retrieval.py`, `02_fact_graph.py`, `03_corpus_tools.py`,
  `04_conversation.py`, `05_swap_a_block.py` (injektování vlastního portu).
- `pyproject.toml` → `pip install -e .`; `import jellyai` funguje bez „spouštěj z rootu".

## 12. Veřejné API (`jellyai/__init__.py`)

```python
from jellyai import (
    # granulární bloky
    load_documents, chunk, tokenize, split_sentences,
    Retriever, build_fact_graph, FactGraph, extract_facts, Fact, Participant,
    ExtractiveAnswerer, GraphAnswerer, analyze_question, Answer,
    # porty (protokoly) pro vlastní/NN implementace
    Tokenizer, FactExtractor, QuestionAnalyzer, Answerer,
    # korpus, fasáda, výuka
    CorpusTools, Jelly, demo, explain,
    JellyError,
)
```

## 13. Testy (hermetické)

- Veřejné API: `import jellyai` a klíčové symboly existují.
- `demo()` vrátí očekávané odpovědi bez modelů.
- Session JSON round-trip: save → load → pokračuje od vah (těžiště, historie).
- `Config.to_json/from_json` round-trip.
- `CorpusTools` lifecycle (mock/Fake): start/stop, context manager.
- `answer.explain()` obsahuje trasu.
- Injektování: `Jelly(retriever=Fake)` použije vlastní port.
- Chyby: chybějící model → `JellyError` s akční hláškou.

## 14. Rozsah a fáze

- **Fáze 1 (tento spec):** porty + protokoly, tenká fasáda s DI, `CorpusTools` start/stop,
  JSON session + JSON config, logging/debug, `JellyError`, viewBase odstínění, `demo()`,
  `examples/`, `pyproject.toml`, veřejné API. **Bez přesunů**, testy zelené.
- **Fáze 2 (mimo tento spec):** logika z `cli.py` do knihovny; `docs/guide/` vedený
  tutoriál (artefaktový styl).
- **Fáze 3 (volitelně):** fyzické přeskupení do podbalíků + přejmenování.

## 15. Mimo rozsah / roadmapa

- Skutečné NN implementace portů (embeddings, neural extractor…) — porty je jen
  **umožní**; modely samotné jsou pozdější kapitola.
- Bonus: hezké `__repr__` (REPL/Jupyter), CLI 1:1 zrcadlení knihovny v docs.
