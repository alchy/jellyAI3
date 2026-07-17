# pseudo-QL: šablonový dotazovací jazyk — implementační plán

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dovést šablonový parser otázek (pseudo-QL) na paritu s UDPipe cestou (etalon JÁDRO 24/24), publikovat ho jako REST/graphQL službu (`/query`, `/graphql`, `/schema`) a odstranit UDPipe z query strany.

**Architecture:** `Pattern`/`SubQuery` zůstávají AST pseudo-QL (vykonavatel `_pattern_answer`/`_match`/`_solve` se osvědčil — parita je cíl, nový AST je YAGNI). Parser `build_query` nově vrací bohatší **`Query`** (pattern + qtype/rod/spona — vše odvozené šablonami, bez UDPipe), takže po přepnutí query strana UDPipe vůbec nepotřebuje. „GraphQL publikace" se děje na API vrstvě: JSON wire formát Patternu + `/schema`. Postup je řízen benchmarkem: každý task končí měřením (`run_etalon.py --mode templates`).

**Tech Stack:** Python 3.11 (`.venv`), pytest, `http.server` (vzor `services/_common.py`), žádné nové závislosti.

## Global Constraints (ze spec + zásad projektu)

- **Query-side NIKDY nevolá UDPipe** (cílový stav; build grafu UDPipe smí). MorphoDiTa (`client.analyze/generate` pro nominativizaci „Kde" odpovědí) je povolená.
- **Nikdy nevracet chybný Pattern** — když vzor nejde bezpečně sestavit, `build_query` vrací `None` (volající spadne na fallback). Chybný Pattern je horší než None.
- **Funkční slova a tabulky jen z `jellyai/lang/cs.json`** — žádný hardcode jmen, tvarů ani tázacích slov v kódu. Jiný jazyk = jiný JSON.
- **Determinismus** — žádné `set`-pořadí ve výstupech (řadit, `dict.fromkeys`).
- **Interim guardrail:** dokud šablony nemají paritu, `main` drží UDPipe fallback (režim `hybrid`) a etalon nikdy neklesne pod 24/24.
- **Zpětná kompatibilita NENÍ požadavek** — API/tvary se smí měnit, drží se jen funkčnost měřená etalonem.
- **Po každé změně restartovat běžící web/API** — běžící instance (`./jelly web`, query služba) drží starý kód i graf. Před ručním ověřením: `pkill -f "cli.py web" ; pkill -f query_service.py` a spustit znovu. (Poznámka od uživatele — platí pro ruční kontroly; benchmarky startují čerstvý proces samy.)
- **Autonomní režim:** neptej se na další krok, iteruj sám; každou změnu měř čísly (`pytest -q`, `benchmark/run_etalon.py`, `benchmark/run_coverage.py`), každý dořešený gap = nový normativní řádek etalonu.
- Spouštění: vždy `.venv/bin/python …` z kořene repa (`/Users/j/Projects/jellyAI3`).

## Naměřený baseline (2026-07-17, po commitu `95e3b80`)

- UDPipe primární (dnešní default): **JÁDRO 24/24**, GAP 2 opraveno / 3 otevřeno.
- Šablony jako jediná cesta: **JÁDRO 16/24**. Selhává (diagnostikováno na živém grafu):
  1. `Kdo napsal Babičku?` → `_resolve_topic(["Babičku"], "napsat")` vrací zbytkový povrchový uzel **„Babičku"** (exact-hit na skloněný uzel přebije uzel „Babička" s napsat-faktem). Oprava: afinita v rámci téhož kmenového clusteru (Task 4).
  2. `Kdo byl bratr Karla Čapka?` → `_resolve_topic(["Karla","Čapka"])` vrací **„Antonína Čapka"** — jeden case-insensitive povrchový hit („Čapka") přebije DVA kmenové hity uzlu „Karel Antonín Čapek". Oprava: pokrytí termů jako primární patro (Task 4).
  3. `Kde se narodil bratr Karla/Josefa Čapka?` → vztahové pravidlo zahodí sloveso a vrátí `Pattern(bratr,…)` místo `narodit(subj=SubQuery(bratr,…), díra loc)` (Task 8).
  4. `Jakou hru napsal Karel Čapek?` → `_resolve_topic(["hru"])` je None (min_stem=3 blokuje „hru"→„hr"); dva subj místo typového filtru (Task 4 + 7).
  5. `Napsal Karel Čapek Válku s mloky?` → ano/ne nerozpoznáno (Task 6); navíc `_resolve_topic(["Válku","s","mloky"])` vrací **„Hovory s TGM"** — funkční slovo „s" skóruje jako term (Task 4 + 5).
  6. Dialog `Co se stalo s rodinou?` → „stalo" se nespáruje se „stát" (prefix 3 < práh 4) a stane se entitním během (Task 8).
- Ruční scénář (měřeno UDPipe cestou — zdroj etalonových expectů): `Kdo je jezis?` → „David, Kristus, Šimon Petr"; `Kde se narodil Jezis?` → „Betlémě"; `Kdo byl bratr autora, ktery napsal R.U.R.?` (bez diakritiky) → **špatně** („František Xaver Šalda, …") — šablony to po Tasku 9 umí; řádek jde do etalonu jako `gap` a přepnutím zezelená.
- Coverage baseline: před začátkem spusť `.venv/bin/python benchmark/run_coverage.py` a řádek `UZLY (…)` poznamenej do commit message Tasku 1 — finální srovnání ho nesmí zhoršit.

## Struktura souborů

- `jellyai/answerer/query.py` — parser: `Query` dataclass, `build_query(question, predicates, is_node=None) -> Query|None`, greedy dělení, ano/ne, typový filtr, date drill. (Tasky 3, 5–9)
- `jellyai/answerer/pattern.py` — po přepnutí jen `Pattern`/`SubQuery` + `pattern_to_json`/`pattern_from_json`; UDPipe parsing (`question_pattern` a spol.) se maže. (Tasky 11, 14)
- `jellyai/answerer/graph_answerer.py` — `query_mode` gate → nakonec čistě šablony; `_resolve_topic` opravy; `_span_is_node`; `run_pattern`. (Tasky 1, 3, 4, 5, 11, 14)
- `jellyai/lang/cs.json` + `jellyai/lang/__init__.py` — funkční tabulky dotazu. (Task 2)
- `config.py` — `GraphConfig.query_mode`, `ServicesConfig.query_port`. (Tasky 1, 12)
- `services/_common.py` — GET routy; `services/query_service.py` — REST služba. (Task 12)
- `jellyai/query_client.py`, `jellyai/answerer/remote.py`, `jellyai/pipeline.py`, `cli.py` — klient + napojení webu a CLI. (Task 13)
- `benchmark/run_etalon.py` (`--mode`), `benchmark/etalon.jsonl` (nové řádky), `tests/test_query.py`, `tests/test_topic_resolve.py`, `tests/test_query_service.py`, `tests/test_cli_web.py`. (průběžně)

---

### Task 1: Měřicí režim `query_mode` + `--mode` v etalonu

Bez objektivního měření šablon nelze iterovat (§5c). Boolean `use_templates` nahradí tříhodnotový `query_mode`: `"udpipe"` (dnešní default), `"hybrid"` (šablony první, UDPipe fallback), `"templates"` (šablony jediná cesta — měřicí/finální režim).

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py:47-70` (signatura) a `:174-181` (gate v `_pattern_answer`)
- Modify: `config.py` (GraphConfig), `jellyai/tasks.py` (`make_graph_answerer`)
- Modify: `benchmark/run_etalon.py`
- Test: `tests/test_query_mode.py` (nový)

**Interfaces:**
- Produces: `GraphAnswerer(graph, client, fallback, *, …, query_mode="udpipe")`; `GraphConfig.query_mode: str`; `make_graph_answerer(config)` režim čte z configu; `run_etalon.py --mode {udpipe,hybrid,templates}`.

- [ ] **Step 1: Failing test**

```python
# tests/test_query_mode.py
"""Režimy query cesty: templates = šablony JEDINÁ cesta (bez UDPipe fallbacku)."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient

_PARSE = {"Kdo stvořil svět?": [[
    {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
    {"form": "stvořil", "lemma": "stvořit", "upos": "VERB", "head": 0, "deprel": "root"},
    {"form": "svět", "lemma": "svět", "upos": "NOUN", "head": 2, "deprel": "obj"},
]]}


def _graph():
    g = FactGraph()
    g.add_fact(make_fact("stvořit", [Participant("subj", "Bůh", "person"),
                                     Participant("obj", "svět", "concept")]))
    return g


def test_templates_mode_answers_without_fallback():
    a = GraphAnswerer(_graph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert a.answer("Kdo stvořil svět?", []).text == "Bůh"


def test_templates_mode_ignores_udpipe_pattern():
    """Prázdný graf → šablony predikát neznají → None; UDPipe by odpověď
    složil, ale v templates režimu NESMÍ naskočit — poctivé „nenašel"."""
    a = GraphAnswerer(FactGraph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert "nenašel" in a.answer("Kdo stvořil svět?", []).text


def test_hybrid_mode_falls_back_to_udpipe():
    """Hybrid: překlep „sworil" šablony nespárují (prefix 1 znak) → None →
    UDPipe rozbor (fake parse zná správná lemmata) odpověď složí."""
    q = "Kdo sworil svět?"
    parse = {q: _PARSE["Kdo stvořil svět?"]}
    a = GraphAnswerer(_graph(), FakeUfalClient(parse=parse),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="hybrid")
    assert a.answer(q, []).text == "Bůh"
    # tentýž překlep v templates režimu → poctivé nenašel (fallback nesmí naskočit)
    b = GraphAnswerer(_graph(), FakeUfalClient(parse=parse),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert "nenašel" in b.answer(q, []).text
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query_mode.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'query_mode'`.

- [ ] **Step 3: Implementace**

`graph_answerer.py` — v `__init__` nahraď `use_templates=False` za `query_mode="udpipe"` a `self.use_templates = use_templates` za `self.query_mode = query_mode`. V `_pattern_answer` nahraď blok „ŠABLONOVÝ PARSER…" (řádky ~175–181):

```python
        pat = None
        if self.query_mode in ("hybrid", "templates"):
            pat = build_query(question, self._predicates)
        if pat is None and self.query_mode != "templates":
            pat = question_pattern(question, self.client)
        pat = pat or Pattern()
```

a rozšiř import: `from jellyai.answerer.pattern import question_pattern, SubQuery, Pattern`.

`config.py` — do `GraphConfig` přidej atribut (+ řádek do docstringu):

```python
    query_mode: str = "udpipe"   # "udpipe" | "hybrid" | "templates" (pseudo-QL)
```

`jellyai/tasks.py:make_graph_answerer` — do volání konstruktoru přidej `query_mode=config.graph.query_mode`.

`benchmark/run_etalon.py` — `main()` dostane přepínač:

```python
import argparse
...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("udpipe", "hybrid", "templates"),
                        default=None, help="query režim (default: config)")
    args = parser.parse_args()
    with open(ETALON, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    config = Config()
    if args.mode:
        config.graph.query_mode = args.mode
    answerer = make_graph_answerer(config)
    ...
    print(f"\nJÁDRO: {passed}/{core} ({pct} %)   "
          f"GAP: {gap_fixed} opraveno / {gap_open} otevřeno   "
          f"[režim {config.graph.query_mode}]")
```

- [ ] **Step 4: Ověř PASS + benchmarky**

Run: `.venv/bin/python -m pytest tests/test_query_mode.py tests/test_query.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → JÁDRO 24/24 (default udpipe, beze změny).
Run: `.venv/bin/python benchmark/run_etalon.py --mode templates` → JÁDRO 16/24 (startovní čára).
Run: `.venv/bin/python benchmark/run_coverage.py` → poznamenej řádek `UZLY (…)`.

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/graph_answerer.py config.py jellyai/tasks.py benchmark/run_etalon.py tests/test_query_mode.py
git commit -m "feat(query): query_mode gate (udpipe/hybrid/templates) + etalon --mode; baseline šablony 16/24"
```

---

### Task 2: Funkční tabulky dotazu do `cs.json`

`_COPULA`/`_RELATIVE`/`_SKIP`/`_HOLE` jsou dnes hardcode v `query.py` — porušení mantinelu §5. Přesun do jazykových dat + nové tabulky pro Tasky 6–9 (qtype, tvary generických dějových sloves, tvary rok/měsíc/den).

**Files:**
- Modify: `jellyai/lang/cs.json`, `jellyai/lang/__init__.py` (`load_rules`)
- Modify: `jellyai/answerer/query.py` (čte tabulky z `current()`)
- Test: `tests/test_lang.py` (rozšířit); `tests/test_query.py` zůstává zelený (chování beze změny)

**Interfaces:**
- Produces: `current()["interrogatives"]` = dict bezdiakritický tvar → tuple `(hole_role, hole_type|None, qtype)`; `current()["copula_forms"]`, `current()["relative_pronouns"]`, `current()["query_skip_words"]` = frozenset; `current()["event_verb_forms"]`, `current()["date_part_forms"]` = dict tvar→lemma. Klíče tabulek jsou bezdiakritické lowercase (parser normalizuje `_norm`).

- [ ] **Step 1: Failing test**

Do `tests/test_lang.py` přidej:

```python
def test_query_function_tables_loaded():
    """Funkční slova dotazu jsou jazyková data, ne kód (spec §5)."""
    from jellyai.lang import load_rules
    rules = load_rules("cs")
    assert rules["interrogatives"]["kdo"] == ("subj", "person", "Kdo")
    assert rules["interrogatives"]["kterem"] == ("attr", None, "Který")
    assert "byl" in rules["copula_forms"]
    assert "ktery" in rules["relative_pronouns"]
    assert "se" in rules["query_skip_words"]
    assert rules["event_verb_forms"]["stalo"] == "stát"
    assert rules["date_part_forms"]["roce"] == "rok"
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_lang.py -v`
Expected: FAIL s `KeyError: 'interrogatives'`.

- [ ] **Step 3: Implementace**

`jellyai/lang/cs.json` — přidej klíče (JSON nemá tuple — pole; `null` = žádný typ):

```json
  "interrogatives": {
    "kdo": ["subj", "person", "Kdo"], "koho": ["obj", "person", "Kdo"],
    "co": ["obj", null, "Co"], "ceho": ["obj", null, "Co"], "cemu": ["obj", null, "Co"],
    "kde": ["loc", "geo", "Kde"], "kam": ["loc", "geo", "Kde"], "odkud": ["loc", "geo", "Kde"],
    "kdy": ["time", "time", "Kdy"], "kolik": ["num", "number", "Kolik"],
    "jaky": ["attr", null, "Jaký"], "jaka": ["attr", null, "Jaký"],
    "jake": ["attr", null, "Jaký"], "jakou": ["attr", null, "Jaký"],
    "jakeho": ["attr", null, "Jaký"], "jakem": ["attr", null, "Jaký"],
    "ktery": ["attr", null, "Který"], "ktera": ["attr", null, "Který"],
    "ktere": ["attr", null, "Který"], "kterou": ["attr", null, "Který"],
    "kterem": ["attr", null, "Který"], "ci": ["subj", "person", "Čí"]
  },
  "copula_forms": ["je", "byl", "byla", "bylo", "byli", "byly", "jsou", "byt",
                   "jsem", "jsi", "jste", "jsme"],
  "relative_pronouns": ["ktery", "ktera", "ktere", "kteri", "jenz", "jez"],
  "query_skip_words": ["se", "si", "v", "ve", "na", "o", "s", "z", "ze", "do",
                       "u", "k", "ke", "po", "za", "pod", "nad", "pred", "pri",
                       "ten", "ta", "to", "tento", "tato", "toto", "tomto", "teto"],
  "event_verb_forms": {"stalo": "stát", "stala": "stát", "stal": "stát",
                       "staly": "stát", "deje": "dít", "delo": "dít",
                       "udalo": "udát", "prihodilo": "přihodit"},
  "date_part_forms": {"rok": "rok", "roku": "rok", "roce": "rok", "letech": "rok",
                      "mesic": "měsíc", "mesici": "měsíc",
                      "den": "den", "dne": "den", "dni": "den"}
```

`jellyai/lang/__init__.py` — v `load_rules` za blok `predicate_synonyms` přidej:

```python
        rules["interrogatives"] = {k: tuple(v) for k, v
                                   in rules.get("interrogatives", {}).items()}
        for key in ("copula_forms", "relative_pronouns", "query_skip_words"):
            rules[key] = frozenset(rules.get(key, ()))
        for key in ("event_verb_forms", "date_part_forms"):
            rules[key] = dict(rules.get(key, {}))
```

`jellyai/answerer/query.py` — smaž modulové konstanty `_COPULA`, `_RELATIVE`, `_SKIP`, `_HOLE`; na místech užití čti `current()` (na začátku funkce `lang = current()`, pak `lang["copula_forms"]` atd.; díra bere z `lang["interrogatives"]` jen `[:2]` — qtype využije Task 3). `_verb_match` čte `current()["copula_forms"]` přímo.

POZOR: „ktery" je v `interrogatives` i `relative_pronouns` — zachovej dnešní pořadí (uvnitř běhu se testuje relativum, díra se bere z prvního tázacího tokenu věty).

- [ ] **Step 4: Ověř PASS**

Run: `.venv/bin/python -m pytest tests/test_lang.py tests/test_query.py tests/test_query_mode.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → 24/24; `--mode templates` → ≥ 16/24 (nesmí klesnout).

- [ ] **Step 5: Commit**

```bash
git add jellyai/lang/cs.json jellyai/lang/__init__.py jellyai/answerer/query.py tests/test_lang.py
git commit -m "refactor(lang): funkční tabulky dotazu (tázací/spona/skip/relativa/tvary) do cs.json"
```

---

### Task 3: `Query` — šablonová analýza otázky bez UDPipe

`answer()` dnes volá `analyze_question` (= UDPipe parse) VŽDY, i když pattern složí šablony. `build_query` nově vrací `Query`: pattern + signály, které answerer čte z `QuestionAnalysis` (qtype, verb_lemma, is_copula, topic_terms, gender) — odvozené šablonami. V režimu `templates` se UDPipe nedotkne vůbec.

**Files:**
- Modify: `jellyai/answerer/query.py` (dataclass `Query`, návratový typ)
- Modify: `jellyai/answerer/graph_answerer.py` (`answer()` restrukturace, `_pattern_answer` signatura)
- Test: `tests/test_query.py` (migrace na `q.pattern`), `tests/test_query_mode.py` (parse se nesmí volat)

**Interfaces:**
- Produces: `Query(pattern: Pattern, qtype: str|None, verb_lemma: str|None, is_copula: bool, topic_terms: list, gender: str|None)` — duck-type kompatibilní s `QuestionAnalysis` (stejná jména atributů). `build_query(...) -> Query | None`. `GraphAnswerer.last_pattern` = poslední vykonaný `Pattern` (čte Task 12).
- Consumes: `current()["interrogatives"]` z Tasku 2 (třetí prvek = qtype).

- [ ] **Step 1: Failing testy**

Do `tests/test_query_mode.py` přidej:

```python
class _NoParseClient(FakeUfalClient):
    def parse(self, text):
        raise AssertionError("query strana volala UDPipe parse!")


def test_templates_mode_never_calls_parse():
    """Režim templates: celá cesta otázka→odpověď bez jediného parse."""
    a = GraphAnswerer(_graph(), _NoParseClient(),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert a.answer("Kdo stvořil svět?", []).text == "Bůh"
```

`tests/test_query.py` — všech 6 stávajících testů přepiš na nový tvar (`pat` → `q.pattern`), např.:

```python
def test_kdo_predicate_entity():
    q = build_query("Kdo napsal Babičku?", PREDS)
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role == "subj"
    assert ("obj", "Babičku") in q.pattern.known
    assert q.qtype == "Kdo" and q.verb_lemma == "napsat"
    assert q.gender == "Masc"          # „napsal" — l-ové příčestí mužské
```

(analogicky ostatní; `test_non_question_returns_none` beze změny — pořád `is None`). Přidej:

```python
def test_query_gender_from_verb_form():
    q = build_query("Kdy se narodila Božena Němcová?", {"narodit"})
    assert q.gender == "Fem" and q.qtype == "Kdy"


def test_query_copula_flag():
    q = build_query("Kdo je Karel Capek?", PREDS)
    assert q.is_copula is True and q.verb_lemma is None
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query.py tests/test_query_mode.py -v`
Expected: FAIL — `Pattern` nemá atribut `pattern` / `AssertionError: query strana volala UDPipe parse!`.

- [ ] **Step 3: Implementace parseru**

`query.py`:

```python
from dataclasses import dataclass, field


def _verb_gender(form):
    """Rod z l-ového příčestí: „narodila"→Fem, „narodil"→Masc, jinak None."""
    low = form.lower()
    if low.endswith("la"):
        return "Fem"
    if low.endswith("l"):
        return "Masc"
    return None


@dataclass
class Query:
    """Šablonový rozbor otázky: pseudo-QL pattern + signály pro answerer.

    Duck-type náhrada `QuestionAnalysis` (stejná jména atributů) — po přepnutí
    na šablony answerer nepotřebuje UDPipe ani na qtype/rod/sponu."""
    pattern: Pattern = None
    qtype: str = None
    verb_lemma: str = None
    is_copula: bool = False
    topic_terms: list = field(default_factory=list)
    gender: str = None
```

`build_query` dál staví `Pattern`, ale každé `return Pattern(...)` zabal do `Query`:
- `qtype` = třetí prvek záznamu `interrogatives` prvního tázacího tokenu (None bez tázacího slova),
- `verb_lemma` = spárovaný predikát (None u spony),
- `is_copula` = ve větě je tvar spony,
- `gender` = `_verb_gender(tvar)` spárovaného slovesa; u sponové věty z tvaru spony („byla"→Fem),
- `topic_terms` = obsahové tokeny, které neskončily v `known` ani jako predikát/spona/tázací/skip (guard „pojmenoval něco, co vzor nezachytil"; po úspěšném sestavení typicky prázdné). `return None` zůstává `None`.

- [ ] **Step 4: Implementace answereru**

`graph_answerer.py` — v `answer()` nahraď `qa = analyze_question(question, self.client)`; gate z Tasku 1 se přesune z `_pattern_answer` sem:

```python
        qa, pat = None, None
        if self.query_mode in ("hybrid", "templates"):
            query = build_query(question, self._predicates)
            if query is not None:
                qa, pat = query, query.pattern
        if qa is None and self.query_mode != "templates":
            qa = analyze_question(question, self.client)
            pat = question_pattern(question, self.client)
        if qa is None:                    # templates-only a šablony nic → nehádat
            qa, pat = Query(), Pattern()
        self.last_pattern = pat
        topic, values, fact = self._pattern_answer(question, pat, qa)
```

`_pattern_answer(self, question, qa)` → `_pattern_answer(self, question, pat, qa)`; parsing z těla zmizí (začíná se `self.visited = []` a `if pat.known:`); `question` zůstává jen pro `parse_date(question)` v generic-event větvi. `self.last_pattern = None` inicializuj v `__init__` i `reset()`. Import: `from jellyai.answerer.query import build_query, Query`.

- [ ] **Step 5: Ověř PASS + benchmarky**

Run: `.venv/bin/python -m pytest tests/test_query.py tests/test_query_mode.py tests/test_graph_answerer.py tests/test_topic_resolve.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → 24/24; `--mode templates` → ≥ 16/24.

- [ ] **Step 6: Commit**

```bash
git add jellyai/answerer/query.py jellyai/answerer/graph_answerer.py tests/test_query.py tests/test_query_mode.py
git commit -m "feat(query): Query — šablonová analýza (qtype/rod/spona) bez UDPipe na query straně"
```

---

### Task 4: `_resolve_topic` — pokrytí termů, cluster-afinita, volný kmen, skip slova

Čtyři diagnostikované defekty rozlišení (viz baseline 1, 2, 4, 5). Oprava pomáhá oběma cestám, měří ji etalon.

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py` (`_resolve_topic:80-146`, nový helper `_loose`)
- Test: `tests/test_topic_resolve.py` (přidat 4 testy; stávající 4 musí zůstat zelené)

**Interfaces:**
- Produces: `_loose(word) -> str` (modulová funkce v graph_answerer — nejvolnější porovnávací klíč); `_resolve_topic` se skóre `(exact, coverage, ins, stem, da, loose, affinity, len, weight)` + post-pass cluster-afinity. Signatura beze změny.

- [ ] **Step 1: Failing testy**

Do `tests/test_topic_resolve.py` přidej (helper `_answerer` nahoru):

```python
def _answerer(graph):
    return GraphAnswerer(graph, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))


def test_term_coverage_beats_single_surface_hit():
    """Dva kmenové hity („Karla"+„Čapka") přebijí jeden povrchový hit („Čapka"
    v uzlu „Antonína Čapka") — pokrytí termů je primární patro."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Antonína Čapka", "person"),
                                 Participant("pred", "otec", "concept")]))
    g.add_fact(make_fact("bratr", [Participant("subj", "Josef Čapek", "person"),
                                   Participant("obj", "Karel Antonín Čapek", "person")]))
    a = _answerer(g)
    assert a._resolve_topic(["Karla", "Čapka"]) == "Karel Antonín Čapek"


def test_same_cluster_prefers_predicate_affinity():
    """Zbytkový skloněný uzel „Babičku" (exact hit) nesmí přebít variantu
    „Babička", o níž se predikát otázky dá vypovědět — v rámci TÉHOŽ
    kmenového clusteru rozhoduje afinita."""
    g = FactGraph()
    g.add_fact(make_fact("kontext", [Participant("subj", "Babičku", "concept"),
                                     Participant("obj", "kraj", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    a = _answerer(g)
    assert a._resolve_topic(["Babičku"], "napsat") == "Babička"


def test_loose_stem_reaches_short_inflected_concept():
    """„hru"→„hra": min_stem=3 blokuje kmen, nejvolnější patro (oboustranné
    seříznutí koncové samohlásky) dosáhne."""
    g = FactGraph()
    g.add_fact(make_fact("druh", [Participant("subj", "R.U.R.", "dílo"),
                                  Participant("pred", "hra", "concept")]))
    a = _answerer(g)
    assert a._resolve_topic(["hru"]) == "hra"


def test_function_words_do_not_score():
    """„s" nesmí skórovat: „Válku s mloky" ≠ „Hovory s TGM"."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "Hovory s TGM", "dílo")]))
    a = _answerer(g)
    assert a._resolve_topic(["Válku", "s", "mloky"]) is None
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_topic_resolve.py -v`
Expected: 4 nové FAIL (vrací „Antonína Čapka"/„Babičku"/None/„Hovory s TGM"), 4 staré PASS.

- [ ] **Step 3: Implementace**

`graph_answerer.py` — modulový helper (nad třídou, vedle `_synonym_ring`):

```python
def _loose(word):
    """Nejvolnější porovnávací klíč: kmen → bez diakritiky → bez koncové
    samohlásky (floor 2). „hru"≡„hra"≡„hr", „jezis"≡„Ježíš". Nejslabší patro
    rozlišení — nikdy nepřebije přesnější shodu."""
    s = deaccent(_stem(word))
    return s[:-1] if len(s) > 2 and s[-1] in "aeiouy" else s
```

`_resolve_topic` přepiš takto (jádro smyčky; zachovej docstring a homonymní vějíř):

```python
        lang = current()
        terms = [t for t in topic_terms
                 if t and len(t) > 1 and deaccent(t.lower()) not in lang["query_skip_words"]]
        low_terms = [t.lower() for t in terms]
        stems = [_stem(t) for t in terms]
        da_stems = [deaccent(s) for s in stems]
        loose = [_loose(t) for t in terms]
        best_id, best_score = None, None
        candidates = []      # (id, stem_hits, váha, klíč clusteru, afinita)
        ring = _synonym_ring(predicate) if predicate else ()
        for node in self.graph.nodes.values():
            if node.type == "výrok":
                continue
            low_id = node.id.lower()
            low_words = low_id.split()
            node_stems = {_stem(w) for w in low_words}
            da_node = {deaccent(s) for s in node_stems}
            loose_node = {_loose(w) for w in low_words}
            per_term = [(t == low_id or t in low_words, s in node_stems,
                         d in da_node, l in loose_node)
                        for t, s, d, l in zip(low_terms, stems, da_stems, loose)]
            coverage = sum(1 for hit in per_term if any(hit))
            if coverage == 0:
                continue
            ins_hits = sum(1 for hit in per_term if hit[0])
            stem_hits = sum(1 for hit in per_term if hit[1])
            da_hits = sum(1 for hit in per_term if hit[2])
            loose_hits = sum(1 for hit in per_term if hit[3])
            exact_hits = sum(1 for t in terms
                             if any(ch.isupper() for ch in t)
                             and (t == node.id or t in node.id.split()))
            affinity = int(any(self.graph.facts_of(node.id, predicate=pred)
                               for pred, _ in ring))
            score = (exact_hits, coverage, ins_hits, stem_hits, da_hits,
                     loose_hits, affinity, len(low_words), node.weight)
            candidates.append((node.id, stem_hits, node.weight,
                               frozenset(loose_node), affinity))
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
```

Za výběr `best_id` (před homonymní vějíř) vlož **cluster-afinitní post-pass** — varianty TÉHOŽ jména (stejný loose-klíč) arbitruje afinita, jmenné shody jiných uzlů to nikdy nepřebije:

```python
        if best_id is not None and ring:
            best = next(c for c in candidates if c[0] == best_id)
            if not best[4]:            # vítěz bez afinity → zkus variantu téhož jména
                same = [c for c in candidates
                        if c[3] == best[3] and c[4] and c[0] != best_id]
                if same:
                    best_id = max(same, key=lambda c: c[2])[0]
```

Homonymní vějíř: `top_stem = best_score[3]` (index kmenových hitů se posunul) a `fan` filtruje `c[1] == top_stem` jako dnes.

- [ ] **Step 4: Ověř PASS + benchmarky**

Run: `.venv/bin/python -m pytest tests/test_topic_resolve.py tests/test_graph_answerer.py tests/test_answerer.py -q` → PASS (staré i nové).
Run: `.venv/bin/python benchmark/run_etalon.py` → 24/24 (žádná regrese UDPipe cesty!).
Run: `.venv/bin/python benchmark/run_etalon.py --mode templates` → čekej ≥ 19/24 (opraví „Kdo napsal Babičku?", „Kdo byl bratr Karla Čapka?" a pomůže dialogu). Zapiš číslo.
Pokud UDPipe cesta klesne, příčinu najdi systematicky (skóre tuple vypiš pro FAIL řádek) — NEobcházej změnou expectů.

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/graph_answerer.py tests/test_topic_resolve.py
git commit -m "fix(answerer): _resolve_topic — pokrytí termů, cluster-afinita, volné kmenové patro, skip funkčních slov"
```

---

### Task 5: `is_node` + greedy longest-match dělení běhů (spec 4.3)

Slovník entit je graf: `_span_is_node(span)` = rozpětí se rozřeší na uzel A jeho obsahová slova jsou podmnožinou slov uzlu. Běhy se dělí greedy na maximální is_node rozpětí; skip-slova běh nerozbíjejí (titul „Válka s mloky" drží pohromadě). **Vedoucí sirotek** (první obsahové slovo běhu bez žádné shody) → celý dotaz `None` (nikdy chybný Pattern); **koncový sirotek po shodě** (pokračování titulu, „…Válku s mloky" bez uzlu titulu) → zahodit.

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py` (`_span_is_node`, předání do `build_query`)
- Modify: `jellyai/answerer/query.py` (`build_query(question, predicates, is_node=None)`, `_collect_known` + `_split_run`)
- Test: `tests/test_query.py`

**Interfaces:**
- Produces: `build_query(question, predicates, is_node=None)`; `is_node: callable(str) -> bool | None` (None = dnešní chování: celý běh = entita, skip-slova = hranice — drží stávající unit testy). `GraphAnswerer._span_is_node(span) -> bool`.
- Consumes: `_loose`, `_resolve_topic` z Tasku 4; `query_skip_words` z Tasku 2.

- [ ] **Step 1: Failing testy**

Do `tests/test_query.py` přidej:

```python
def _nodes(*spans):
    """Fake slovník grafu: is_node = přesná množina povolených rozpětí."""
    allowed = set(spans)
    return lambda span: span in allowed


def test_greedy_splits_glued_entities():
    """Dvě entity za sebou bez hranice se rozdělí na maximální uzlová rozpětí."""
    is_node = _nodes("Karel Čapek", "Válku s mloky")
    q = build_query("Co napsal Karel Čapek Válku s mloky?", {"napsat"}, is_node)
    spans = [t for _, t in q.pattern.known]
    assert spans == ["Karel Čapek", "Válku s mloky"]


def test_skip_word_bridges_title():
    """Předložka uvnitř titulu běh NErozbíjí: „Válku s mloky" je jedno rozpětí."""
    is_node = _nodes("Válku s mloky")
    q = build_query("Kdo napsal Válku s mloky?", {"napsat"}, is_node)
    assert ("obj", "Válku s mloky") in q.pattern.known


def test_trailing_orphan_after_match_dropped():
    """Graf zná jen „Válku" — „s mloky" je pokračování titulu, zahodí se."""
    is_node = _nodes("Válku")
    q = build_query("Kdo napsal Válku s mloky?", {"napsat"}, is_node)
    assert ("obj", "Válku") in q.pattern.known


def test_leading_orphan_returns_none():
    """„Ludvík" graf nezná → vzor nelze bezpečně sestavit → None (žádné
    hádání přes „Němec", které by trefilo Němcovou)."""
    is_node = _nodes("Němec")
    assert build_query("Kdo je Ludvík Němec?", {"být"}, is_node) is None


def test_without_is_node_keeps_old_behavior():
    q = build_query("Kdo napsal Babičku?", PREDS, None)
    assert ("obj", "Babičku") in q.pattern.known
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query.py -v`
Expected: nové testy FAIL (`TypeError` na třetí argument), staré PASS.

- [ ] **Step 3: Implementace parseru**

`query.py` — `build_query(question, predicates, is_node=None)` protáhne `is_node` do `_collect_known(tokens, predicates, relational, is_node)`. V `_collect_known` skip-slova PŘESTANOU být hranicí běhu (z podmínky `boundary` vypadne `low in _SKIP`, zůstává tázací/spona/sloveso/relativum) — místo toho se každý ukončený běh prožene `_split_run`; když `_split_run` vrátí `None`, celé `_collect_known` (a tím `build_query`) vrací `None`:

```python
def _trim(run):
    """Okrajová skip-slova rozpětí pryč („Válku s" → „Válku")."""
    skip = current()["query_skip_words"]
    while run and _norm(run[0]) in skip:
        run = run[1:]
    while run and _norm(run[-1]) in skip:
        run = run[:-1]
    return run


def _split_run(run, is_node, relational):
    """Greedy longest-match: běh → maximální `is_node` rozpětí (spec 4.3).

    Vedoucí sirotek (první obsahové slovo bez shody) = nebezpečný vzor → None;
    sirotek PO shodě je pokračování titulu → zahodit. Vztahové jméno na začátku
    se přilepí k první entitě (vztahovou šablonu řeší build_query)."""
    run = _trim(run)
    if not run:
        return []
    rel = run[0] if _norm(run[0]) in relational else None
    if rel:
        run = _trim(run[1:])
        if not run:
            return [("obj", rel)]
    if is_node is None:
        parts = [("obj", " ".join(run))]
    else:
        skip = current()["query_skip_words"]
        parts, i, matched = [], 0, False
        while i < len(run):
            if _norm(run[i]) in skip:
                i += 1
                continue
            hit = None
            for j in range(len(run), i, -1):
                span = _trim(run[i:j])
                if span and is_node(" ".join(span)):
                    hit = (j, " ".join(span))
                    break
            if hit is None:
                if not matched:
                    return None          # vedoucí sirotek — nikdy chybný Pattern
                i += 1                   # koncový sirotek po shodě → zahodit
                continue
            parts.append(("obj", hit[1]))
            matched, i = True, hit[0]
        if not parts:
            return None
    if rel:
        parts[0] = ("obj", rel + " " + parts[0][1])
    return parts
```

V `_collect_known` každé `known.append(("obj", " ".join(run)))` nahraď:

```python
            parts = _split_run(run, is_node, relational)
            if parts is None:
                return None
            known.extend(parts)
```

a `build_query` po `known = _collect_known(...)` doplní `if known is None: return None`. Větev relativa (`sub = _subquery(...)`) zůstává (běh před „který" se nahrazuje pod-dotazem jako dnes).

- [ ] **Step 4: Implementace answereru**

`graph_answerer.py`:

```python
    def _span_is_node(self, span):
        """Přísný test rozpětí (spec 4.3): rozřeší se na uzel A jeho obsahová
        slova jsou podmnožinou slov uzlu (kmenově/bezdiakriticky) — slepenec
        dvou entit ani cizí titul neprojde."""
        terms = [t for t in span.split()
                 if len(t) > 1 and deaccent(t.lower()) not in current()["query_skip_words"]]
        if not terms:
            return False
        node_id = self._resolve_topic(terms)
        if node_id is None:
            return False
        node_keys = {_loose(w) for w in node_id.split()}
        return all(_loose(t) in node_keys for t in terms)
```

a v `answer()` (gate z Tasku 3): `build_query(question, self._predicates, self._span_is_node)`.

POZOR: `_resolve_topic` uvnitř `_span_is_node` rozsvěcí homonymní vějíř (`context.warm`) — u zamítnutých rozpětí je to šum. Přidej `_resolve_topic(..., warm=True)` parametr a z `_span_is_node` volej `warm=False` (vějíř přeskoč).

- [ ] **Step 5: Ověř PASS + benchmarky**

Run: `.venv/bin/python -m pytest tests/test_query.py tests/test_query_mode.py tests/test_topic_resolve.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → 24/24; `--mode templates` → zapiš (čekej ≥ 20/24; „Kdo je Ludvík Němec?"/„Kdo byl Svatopluk Machar?" musí zůstat „nenašel").

- [ ] **Step 6: Commit**

```bash
git add jellyai/answerer/query.py jellyai/answerer/graph_answerer.py tests/test_query.py
git commit -m "feat(query): is_node + greedy longest-match dělení běhů (spec 4.3) — slovník entit je graf"
```

---

### Task 6: Zjišťovací (ano/ne) otázky (spec 4.5)

Věta s „?", bez tázacího slova, začínající slovesem spárovaným s predikátem → `Pattern` s `hole_role=None` a `qtype=None`; existující `_existence` v answereru ji vykoná („Ano"/nenašel). Answerer se nemění.

**Files:**
- Modify: `jellyai/answerer/query.py` (`build_query` — větev ano/ne)
- Modify: `benchmark/etalon.jsonl` (negativní řádek)
- Test: `tests/test_query.py`

**Interfaces:**
- Produces: ano/ne `Query`: `pattern.hole_role is None`, `qtype is None`, `known` = všechny entity (první subj, další obj).
- Consumes: `_split_run`/is_node z Tasku 5 (slepený běh „Karel Čapek Válku s mloky" za slovesem).

- [ ] **Step 1: Failing testy**

```python
def test_yes_no_question_builds_existence_pattern():
    """„Napsal Karel Čapek Válku s mloky?" → predikát + všechny entity known,
    díra žádná (existenční test), qtype None."""
    is_node = _nodes("Karel Čapek", "Válku s mloky")
    q = build_query("Napsal Karel Čapek Válku s mloky?", {"napsat"}, is_node)
    assert q is not None and q.qtype is None
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role is None
    assert ("subj", "Karel Čapek") in q.pattern.known
    assert ("obj", "Válku s mloky") in q.pattern.known


def test_yes_no_needs_leading_verb():
    """Bez tázacího slova a bez počátečního slovesa vzor nevznikne (None)."""
    is_node = _nodes("Karel Čapek")
    assert build_query("Karel Čapek napsal?", {"napsat"}, is_node) is None
```

POZOR na guard velkých písmen v `_verb_match` („Napsal" na začátku věty JE sloveso): guard `token[:1].isupper()` musí u PRVNÍHO tokenu věty povolit shodu — uprav `_verb_match(token, predicates, first=False)` a z detekce ano/ne volej s `first=True` (jinde chování beze změny, „Němec" dál neprojde).

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query.py -v` → nové FAIL (vrací None / špatné role).

- [ ] **Step 3: Implementace**

V `build_query` za detekci díry (když `hole_role is None` a žádný tázací token nebyl nalezen):

```python
    if hole_role is None:
        # zjišťovací otázka: věta začíná slovesem spárovaným s predikátem grafu
        verb = _verb_match(tokens[0], predicates, first=True)
        if verb is None:
            return None
        known = _collect_known(tokens[1:], predicates, relational, is_node)
        if not known:                     # None (sirotek) i [] (bez entit) → nehádat
            return None
        known = [("subj" if i == 0 else "obj", term)
                 for i, (_, term) in enumerate(known)]
        return Query(Pattern(verb, known, None, None), qtype=None,
                     verb_lemma=verb, gender=_verb_gender(tokens[0]))
```

(dnešní chování „bez tázacího slova → pokračuj a možná None" tím přechází na explicitní větev; sponové věty bez tázacího slova — „Je X Y?" — nech zatím None, etalon je nevyžaduje).

- [ ] **Step 4: Ověř PASS + benchmarky + etalon**

Run: `.venv/bin/python -m pytest tests/test_query.py -q` → PASS.
Do `benchmark/etalon.jsonl` přidej negativní ano/ne řádek (za stávající ano-ne řádky):

```json
{"q": "Napsal Josef Čapek Babičku?", "expect": ["nenašel"], "reject": ["Ano"], "cat": "ano-ne"}
```

Run: `.venv/bin/python benchmark/run_etalon.py` → 25/25; `--mode templates` → zapiš (řádek „Napsal Karel Čapek Válku s mloky?" musí být PASS).

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/query.py tests/test_query.py benchmark/etalon.jsonl
git commit -m "feat(query): zjišťovací (ano/ne) otázky — existenční Pattern bez díry (spec 4.5)"
```

---

### Task 7: Typový filtr (spec 4.6)

„**Jakou hru** napsal Karel Čapek?" — konceptový termín za attr-tázacím slovem je typový filtr díry, ne druhý subjekt. Parser dá `known=[("obj","hru"),("subj","Karel Čapek")]`, `hole=attr`; join (`_typed_match`: napsat(X,?) ∧ druh/být(?,hra)) v answereru UŽ existuje a spustí se, když `_match` selže a `len(known_set) > 1`.

**Files:**
- Modify: `jellyai/answerer/query.py` (přiřazení rolí ve slovesné větvi)
- Test: `tests/test_query.py`

**Interfaces:**
- Produces: attr-díra + známé entity → první known role `obj` (filtr), ostatní `subj`; jediný known u attr-díry → `obj` (pro-drop: `_fill_subject` doplní osobu z těžiště jen když v known není subj!).

- [ ] **Step 1: Failing testy**

```python
def test_type_filter_roles():
    """„Jakou hru napsal Karel Čapek?" → filtr obj=hru, téma subj=Karel Čapek."""
    is_node = _nodes("hru", "Karel Čapek")
    q = build_query("Jakou hru napsal Karel Čapek?", {"napsat"}, is_node)
    assert q.pattern.hole_role == "attr"
    assert ("obj", "hru") in q.pattern.known
    assert ("subj", "Karel Čapek") in q.pattern.known


def test_type_filter_prodrop_keeps_obj_role():
    """„Jakou hru napsal?" (pro-drop) → jediný known jako obj, aby
    _fill_subject směl doplnit podmět z konverzačního těžiště."""
    is_node = _nodes("hru")
    q = build_query("Jakou hru napsal?", {"napsat"}, is_node)
    assert q.pattern.known == [("obj", "hru")]
    assert q.pattern.hole_role == "attr" and q.gender == "Masc"
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query.py -v` → oba FAIL (dnes všechny knowny dostanou subj).

- [ ] **Step 3: Implementace**

Ve slovesné větvi `build_query` nahraď komplementové pravidlo rolí:

```python
        if hole_role == "attr":
            # výběrová otázka: první known (hned za tázacím slovem) = typový
            # filtr díry (obj), další entity jsou téma (subj) — join řeší answerer
            known = [("obj" if i == 0 else "subj", term)
                     for i, (r, term) in enumerate(known)]
        else:
            role = "obj" if hole_role == "subj" else "subj"
            known = [(role if isinstance(term, str) else r, term)
                     for r, term in known]
```

- [ ] **Step 4: Ověř PASS + benchmarky**

Run: `.venv/bin/python -m pytest tests/test_query.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → 25/25; `--mode templates` → zapiš („Jakou hru napsal Karel Čapek?" a dialog „…Jakou hru napsal?" musí být PASS — dialog potřebuje i Task 4).

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/query.py tests/test_query.py
git commit -m "feat(query): typový filtr díry — konceptový termín u attr otázky je obj filtr (spec 4.6)"
```

---

### Task 8: Vztahy pod slovesem, datový drill, generická dějová slovesa (spec 4.4 + §7)

Tři zbývající parserové gapy: (a) vztahové jméno uvnitř slovesné otázky → `SubQuery` pod slovesem (dnes zahodí sloveso); (b) `date_part` z jazykové tabulky („v kterém roce" → drill rok); (c) tvary generických dějových sloves („stalo"→„stát") z tabulky — prefix na 3 znaky nestačí.

**Files:**
- Modify: `jellyai/answerer/query.py` (pravidlo 1, `_verb_match`, date_part)
- Modify: `benchmark/etalon.jsonl` (řádek date drill)
- Test: `tests/test_query.py`

**Interfaces:**
- Consumes: `event_verb_forms`, `date_part_forms` z Tasku 2.
- Produces: `Pattern(verb, [("subj", SubQuery(rel, [("obj", zbytek)], "subj"))], hole…)` pro „SLOVESO … VZTAH GENITIV"; `Pattern(verb, known, hole, htype, date_part="rok")` pro drill; `_verb_match` zná tvary z `event_verb_forms` (fungují i pro predikáty mimo slovník grafu — `_event_answer` je nepotřebuje mít ve faktech).

- [ ] **Step 1: Failing testy**

```python
def test_relational_under_verb_becomes_subquery():
    """„Kde se narodil bratr Karla Čapka?" → narodit(subj=SubQuery(bratr,
    obj=Karla Čapka), díra loc) — vztah je vnořený dotaz, sloveso vládne."""
    q = build_query("Kde se narodil bratr Karla Čapka?", {"narodit", "bratr"})
    assert q.pattern.predicate == "narodit" and q.pattern.hole_role == "loc"
    role, sub = q.pattern.known[0]
    assert role == "subj" and isinstance(sub, SubQuery)
    assert sub.predicate == "bratr" and ("obj", "Karla Čapka") in sub.known


def test_bare_relational_without_verb_unchanged():
    q = build_query("Kdo byl bratr Karla Capka?", PREDS)
    assert q.pattern.predicate == "bratr"
    assert ("obj", "Karla Capka") in q.pattern.known


def test_date_part_drill():
    """„V kterém roce se narodila BN?" → date_part=rok (2-skokový drill),
    „roce" není účastník."""
    q = build_query("V kterém roce se narodila Božena Němcová?", {"narodit"})
    assert q.pattern.predicate == "narodit" and q.pattern.date_part == "rok"
    assert ("subj", "Božena Němcová") in q.pattern.known
    assert q.gender == "Fem"


def test_generic_event_verb_from_table():
    """„Co se stalo s rodinou?" → predikát stát (tvar z jazykové tabulky,
    prefix nestačí), rodina jako téma."""
    q = build_query("Co se stalo s rodinou?", set())
    assert q.pattern.predicate == "stát"
    assert ("subj", "rodinou") in q.pattern.known


def test_reverse_date_question_stays_none():
    """„Co se stalo v listopadu 1848?" bez uzlového rozpětí → None; reverzní
    lookup (datum→děj) zůstává na answereru (spec §7)."""
    assert build_query("Co se stalo v listopadu 1848?", set(), _nodes()) is None


def test_predicate_synonym_from_language_table():
    """„Kde žili…?" na graf s predikátem „bydlet": tvar se páruje i proti
    lemmatům z predicate_synonyms; vrátí se synonymum („žít") — expanzi na
    bydlet-fakty dělá answererův _synonym_ring (spec 4.2)."""
    q = build_query("Kde žili Čapkovi?", {"bydlet"}, _nodes("Čapkovi"))
    assert q.pattern.predicate == "žít"
    assert q.pattern.hole_role == "loc"
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query.py -v` → 5 nových FAIL (`test_bare_relational…` PASS už dnes).

- [ ] **Step 3: Implementace**

**(c) tvary sloves** — v `_verb_match` před prefixovou smyčku:

```python
    lemma = current()["event_verb_forms"].get(low)
    if lemma:
        return lemma
```

(vrací se i pro predikáty mimo `predicates` — generická dějová větev `_pattern_answer` fakt s predikátem nepotřebuje). Tím se „stalo" stane hranicí běhu a přestane být entitou. A prefixová smyčka páruje i proti synonymům predikátů (spec 4.2 — „žili"→„žít", expanzi na „bydlet" dělá answererův `_synonym_ring`):

```python
    candidates = set(predicates) | set(current()["predicate_synonyms"])
    for pred in candidates:
        ...  # stávající prefixová smyčka beze změny, jen iteruje candidates
```

**(b) date_part** — v `build_query` po tokenizaci:

```python
    date_part = next((current()["date_part_forms"][_norm(t)] for t in tokens
                      if _norm(t) in current()["date_part_forms"]), None)
```

Tokeny z `date_part_forms` jsou hranice běhu (v `_collect_known` je přeskoč jako tázací slova — do `boundary` podmínky přidej `low in current()["date_part_forms"]`) a `date_part` se předá do všech slovesných `Pattern(...)` (`Pattern(verb, known, hole_role, hole_type, date_part)`).

**(a) vztah pod slovesem** — pravidlo 1 v `build_query` přepracuj: sloveso se hledá PŘED vztahovou větví; vztahová šablona s holým predikátem se použije jen bez slovesa:

```python
    verb = next((_verb_match(t, predicates) for t in tokens
                 if _verb_match(t, predicates)), None)

    for k, (role, term) in enumerate(list(known)):
        head = term.split()[0] if isinstance(term, str) and term else None
        if head and _norm(head) in relational:
            rest = " ".join(term.split()[1:])
            if not rest:
                continue
            sub = SubQuery(_norm(head), [("obj", rest)], "subj")
            if verb is not None:
                known[k] = ("subj", sub)       # „Kde se narodil BRATR KARLA…"
            else:
                return Query(Pattern(_norm(head), [("obj", rest)], "subj",
                                     "person"), qtype=qtype, gender=gender)
        if isinstance(term, SubQuery) and verb is None:
            rel = next((t for t in tokens if _norm(t) in relational), None)
            if rel:
                return Query(Pattern(_norm(rel), [("obj", term)], "subj",
                                     "person"), qtype=qtype, gender=gender)
```

(SubQuery členy si ve slovesné větvi drží svou roli — komplementové pravidlo z Tasku 7 přepisuje jen `str` termíny, `isinstance(term, str)` guard už existuje.)

- [ ] **Step 4: Ověř PASS + benchmarky + etalon**

Run: `.venv/bin/python -m pytest tests/test_query.py -q` → PASS.
Do `benchmark/etalon.jsonl` přidej:

```json
{"q": "V kterém roce se narodila Božena Němcová?", "expect": ["1818"], "cat": "čas"}
```

Run: `.venv/bin/python benchmark/run_etalon.py` → 26/26; `--mode templates` → zapiš. Zkontroluj vztahovou symetrii (§7): řádky „Kdo byl bratr Karla/Josefa Čapka?" musí v templates režimu vracet protistranu (Josef/Karel) — pokud vrací téma, změň u holé vztahové šablony `hole_role` na `None` (typ `person` zůstane preferencí) a přeměř.

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/query.py tests/test_query.py benchmark/etalon.jsonl
git commit -m "feat(query): vztah pod slovesem jako SubQuery, date-part drill a tvary dějových sloves z tabulky"
```

---

### Task 9: Diakritika + ruční scénář §6.5 do etalonu

Bezdiakritické dotazy už nesou tabulky (`_norm`, `deaccent`, `_loose`) — tento task je ověří unit testy a ruční scénář přibije do etalonu jako normativní řádky. Rekurzivní bezdiakritický dotaz jde jako `gap` (UDPipe cesta ho dnes kazí — zezelená přepnutím v Tasku 14).

**Files:**
- Modify: `benchmark/etalon.jsonl`, `tests/test_query.py`

**Interfaces:** žádné nové — jen normativní případy.

- [ ] **Step 1: Unit testy (parser, bez grafu)**

```python
def test_diacritic_free_identity():
    q = build_query("Kdo je jezis?", set(), _nodes("jezis"))
    assert q.pattern.predicate == "být"
    assert ("subj", "jezis") in q.pattern.known


def test_diacritic_free_verb_and_entity():
    q = build_query("Kde se narodil Jezis?", {"narodit"}, _nodes("Jezis"))
    assert q.pattern.predicate == "narodit" and q.pattern.hole_role == "loc"
    assert ("subj", "Jezis") in q.pattern.known


def test_diacritic_free_nested_subquery():
    is_node = _nodes("autora", "R.U.R.")
    q = build_query("Kdo byl bratr autora, ktery napsal R.U.R.?",
                    {"bratr", "napsat"}, is_node)
    sub = next(k for _, k in q.pattern.known if isinstance(k, SubQuery))
    assert sub.predicate == "napsat" and ("obj", "R.U.R.") in sub.known
```

Run: `.venv/bin/python -m pytest tests/test_query.py -v` → očekávej PASS (mechanismus už stojí); co FAILne, oprav v `query.py` (bezdiakritické tabulky, `_norm` na správných místech) — to je smysl testů.

- [ ] **Step 2: Etalonové řádky**

Do `benchmark/etalon.jsonl` přidej (expecty pocházejí z měření na živém grafu, viz baseline):

```json
{"q": "Kdo je jezis?", "expect": ["Kristus"], "cat": "diakritika"}
{"q": "Kde se narodil Jezis?", "expect": ["Betlém"], "cat": "diakritika"}
{"q": "Kdo byl bratr autora, ktery napsal R.U.R.?", "expect": ["Josef"], "cat": "diakritika", "gap": "UDPipe cesta bez diakritiky selhává; šablony umí — zezelená přepnutím"}
```

- [ ] **Step 3: Ověř benchmarky**

Run: `.venv/bin/python benchmark/run_etalon.py` → JÁDRO 28/28 (gap řádek se nepočítá do jádra).
Run: `.venv/bin/python benchmark/run_etalon.py --mode templates` → zapiš; bezdiakritické řádky musí být PASS (gap řádek GAP-FIXED). Pokud ne: debug `_span_is_node`/`_resolve_topic` na daném dotazu (skóre vypiš), oprav, přeměř.

- [ ] **Step 4: Commit**

```bash
git add benchmark/etalon.jsonl tests/test_query.py
git commit -m "test(query): diakritika + ruční scénář §6.5 jako normativní etalonové řádky"
```

---

### Task 10: Přepnutí na `hybrid` (guardrail drží 24+/24)

Šablony jako primární cesta s UDPipe fallbackem. Podmínka: parser nikdy nevrací chybný Pattern — všechna „nevím" jsou `None` (fallback naskočí).

**Files:**
- Modify: `config.py` (`GraphConfig.query_mode = "hybrid"`)

- [ ] **Step 1: Změř PŘED přepnutím**

Run: `.venv/bin/python benchmark/run_etalon.py --mode hybrid`
Expected: JÁDRO = plný počet (28/28). Pokud ne, každý FAIL je šablonový Pattern, který je chybný, ale ne None — oprav parser (vrať None dřív), NEPŘEPÍNEJ.

- [ ] **Step 2: Flip defaultu**

V `config.py` změň `query_mode: str = "udpipe"` na `"hybrid"`.

- [ ] **Step 3: Ověř vše**

Run: `.venv/bin/python -m pytest -q` → zelené.
Run: `.venv/bin/python benchmark/run_etalon.py` → JÁDRO plný počet (teď default hybrid).
Run: `.venv/bin/python benchmark/run_etalon.py --mode templates` → zapiš do commit message (cíl ≥ 26/28; zbývající rozdíly vyjmenuj).
Run: `.venv/bin/python benchmark/run_coverage.py` → `UZLY (…)` beze změny proti baseline.

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "feat(query): šablony primární (hybrid) — guardrail UDPipe fallback drží etalon"
```

---

### Task 11: `run_pattern` + JSON wire formát pseudo-QL

Přímé vykonání Patternu (bez parseru) pro `/graphql` endpoint — testovatelnost jazyka samotného. Plus (de)serializace Patternu pro API.

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py` (`run_pattern`)
- Modify: `jellyai/answerer/pattern.py` (`pattern_to_json`, `pattern_from_json`)
- Test: `tests/test_pattern.py` (serializace), `tests/test_graph_answerer.py` (run_pattern)

**Interfaces:**
- Produces: `GraphAnswerer.run_pattern(pat) -> (topic|None, values: list, fact|None)`; `pattern_to_json(pat) -> dict` (`{"predicate", "known": [[role, term|{"subquery": {...}}]], "hole", "hole_type", "date_part"}`); `pattern_from_json(data) -> Pattern` (přijímá `hole` i `hole_role`).

- [ ] **Step 1: Failing testy**

Do `tests/test_pattern.py` přidej:

```python
def test_pattern_json_roundtrip_nested():
    from jellyai.answerer.pattern import (Pattern, SubQuery,
                                          pattern_to_json, pattern_from_json)
    pat = Pattern("bratr", [("obj", SubQuery("napsat", [("obj", "R.U.R.")],
                                             "subj"))], "subj", "person")
    data = pattern_to_json(pat)
    assert data["predicate"] == "bratr" and data["hole"] == "subj"
    assert data["known"][0][1]["subquery"]["predicate"] == "napsat"
    back = pattern_from_json(data)
    assert isinstance(back.known[0][1], SubQuery)
    assert back.known[0][1].known == [("obj", "R.U.R.")]


def test_pattern_from_json_accepts_flat_form():
    from jellyai.answerer.pattern import pattern_from_json
    pat = pattern_from_json({"predicate": "napsat",
                             "known": [["obj", "R.U.R."]], "hole": "subj"})
    assert pat.predicate == "napsat" and pat.hole_role == "subj"
    assert pat.known == [("obj", "R.U.R.")]
```

Do `tests/test_graph_answerer.py` přidej:

```python
def test_run_pattern_executes_direct_ql():
    from jellyai.answerer.pattern import Pattern
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    a = GraphAnswerer(g, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()),
                      query_mode="templates")
    topic, values, fact = a.run_pattern(Pattern("napsat", [("obj", "Babička")],
                                                "subj", "person"))
    assert values == ["Božena Němcová"] and fact.predicate == "napsat"
    # existenční tvar (hole None)
    topic, values, fact = a.run_pattern(Pattern("napsat",
                                                [("subj", "Božena Němcová"),
                                                 ("obj", "Babička")], None, None))
    assert values == ["Ano"]
```

(importy dle hlavičky `tests/test_topic_resolve.py`).

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_pattern.py tests/test_graph_answerer.py -v` → nové FAIL (ImportError/AttributeError).

- [ ] **Step 3: Implementace**

`pattern.py`:

```python
def pattern_to_json(pat):
    """Pseudo-QL jako data (API wire formát): SubQuery → {"subquery": …}."""
    if pat is None:
        return None
    def term(t):
        if isinstance(t, SubQuery):
            return {"subquery": {"predicate": t.predicate,
                                 "known": [[r, term(k)] for r, k in t.known],
                                 "hole": t.hole_role}}
        return t
    return {"predicate": pat.predicate,
            "known": [[r, term(t)] for r, t in pat.known],
            "hole": pat.hole_role, "hole_type": pat.hole_type,
            "date_part": pat.date_part}


def pattern_from_json(data):
    """Opačný směr: dict z API → Pattern (přijímá „hole" i „hole_role")."""
    def term(t):
        if isinstance(t, dict) and "subquery" in t:
            s = t["subquery"]
            return SubQuery(s["predicate"],
                            [(r, term(k)) for r, k in s.get("known", ())],
                            s.get("hole", "subj"))
        return t
    return Pattern(data.get("predicate"),
                   [(r, term(t)) for r, t in data.get("known", ())],
                   data.get("hole") or data.get("hole_role"),
                   data.get("hole_type"), data.get("date_part"))
```

`graph_answerer.py`:

```python
    def run_pattern(self, pat):
        """Vykoná pseudo-QL `Pattern` přímo (API `/graphql` — jazyk je
        testovatelný bez parseru). Sémantika = jádro `_pattern_answer`:
        rozřeš known → díra/existence; bez kontextových pater.

        Returns:
            tuple: (téma | None, list hodnot, fakt | None).
        """
        self.visited = []
        known_set = set()
        for _, known in pat.known:
            node = self._solve(known, pat.predicate)
            if node is None:
                return None, [], None
            known_set.add(node)
        if not known_set:
            return None, [], None
        if pat.hole_role is None and pat.date_part is None:
            return self._existence(pat.predicate, known_set)
        return self._answer_from(pat, known_set)
```

- [ ] **Step 4: Ověř PASS**

Run: `.venv/bin/python -m pytest tests/test_pattern.py tests/test_graph_answerer.py -q` → PASS.
Run: `.venv/bin/python benchmark/run_etalon.py` → beze změny.

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/pattern.py jellyai/answerer/graph_answerer.py tests/test_pattern.py tests/test_graph_answerer.py
git commit -m "feat(ql): run_pattern (přímé vykonání) + JSON wire formát pseudo-QL"
```

---

### Task 12: REST/graphQL služba `services/query_service.py` (spec §5d)

HTTP služba nad grafem + parserem: `POST /query` (přirozený dotaz), `POST /graphql` (přímý pseudo-QL), `GET /schema` (publikovaný slovník jazyka), `GET /health`. Vzor `services/udpipe_service.py`; `--model` = cesta ke `graph.pkl` (graf JE model služby). Stavová (drží konverzační těžiště) — jedna instance, lokální.

**Files:**
- Modify: `services/_common.py` (GET routy)
- Create: `services/query_service.py`
- Modify: `config.py` (`ServicesConfig.query_port: int = 8084`)
- Test: `tests/test_query_service.py` (nový)

**Interfaces:**
- Produces: `serve(host, port, routes, gets=None)`; `query_service.make_routes(answerer) -> (posts: dict, gets: dict)`;
  - `POST /query {"question", "temperature"?}` → `{"answer", "sources", "trace", "alternatives", "pattern", "activation": {"nodes": {id: jas}, "docs": [[doc, jas]…]}}`
  - `POST /graphql {"predicate", "known": [[role, term]…], "hole"?, "hole_type"?, "date_part"?}` → `{"answer", "values", "topic", "fact"}`
  - `GET /schema` → `{"predicates", "roles", "node_types", "holes"}`
- Consumes: `run_pattern`/`pattern_to_json`/`pattern_from_json` (Task 11), `last_pattern` (Task 3), `query_port`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_query_service.py
"""Pseudo-QL jako HTTP služba: /query (přirozený dotaz), /graphql (přímý
pattern), /schema (publikovaný slovník jazyka). Bez ML modelů — graf stačí."""
import json
import sys
import urllib.request

import pytest

sys.path.insert(0, "services")

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _graph():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    return g


def _routes():
    from query_service import make_routes
    a = GraphAnswerer(_graph(), FakeUfalClient(),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    return make_routes(a)


def test_query_route_answers_and_shows_pattern():
    posts, _ = _routes()
    out = posts["/query"]({"question": "Kdo napsal Babičku?"})
    assert "Božena Němcová" in out["answer"]
    assert out["pattern"]["predicate"] == "napsat"
    assert out["trace"]["predicate"] == "napsat"
    assert out["activation"]["nodes"]        # těžiště po tahu svítí


def test_graphql_route_executes_pattern_directly():
    posts, _ = _routes()
    out = posts["/graphql"]({"predicate": "napsat",
                             "known": [["obj", "Babička"]], "hole": "subj"})
    assert out["answer"] == "Božena Němcová"


def test_schema_publishes_graph_vocabulary():
    _, gets = _routes()
    out = gets["/schema"]({})
    assert "napsat" in out["predicates"]
    assert "subj" in out["roles"] and "person" in out["node_types"]
    assert out["holes"]["kdo"] == ["subj", "person"]


def test_service_subprocess_end_to_end(tmp_path):
    """Integrace: služba jako subprocess (--model = graph.pkl), curl přes HTTP."""
    import socket
    import subprocess
    import time
    graph_path = str(tmp_path / "graph.pkl")
    _graph().save(graph_path)
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, "services/query_service.py",
                             "--port", str(port), "--model", graph_path])
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        body = json.dumps({"question": "Kdo napsal Babičku?"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/query", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.load(resp)
        assert "Božena Němcová" in out["answer"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/schema",
                                    timeout=10) as resp:
            assert "napsat" in json.load(resp)["predicates"]
    finally:
        proc.terminate()
```

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query_service.py -v` → FAIL (`ModuleNotFoundError: query_service`).

- [ ] **Step 3: Implementace**

`services/_common.py` — `_make_handler(routes)` → `_make_handler(routes, gets=None)`; v `do_GET`:

```python
        def do_GET(self):
            fn = (gets or {}).get(self.path)
            if fn is not None:
                try:
                    self._send(200, fn({}))
                except Exception as exc:  # noqa: BLE001
                    self._send(500, {"error": str(exc)})
            elif self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})
```

a `serve(host, port, routes, gets=None)` → `_make_handler(routes, gets)`.

`config.py` — do `ServicesConfig` přidej `query_port: int = 8084` (+ docstring řádek).

`services/query_service.py`:

```python
"""Pseudo-QL služba — dotazovací jazyk nad faktovým grafem přes HTTP (spec §5d).

`POST /query` přijme přirozenou otázku (šablonový parser → pseudo-QL → graf),
`POST /graphql` přímý pseudo-QL pattern (obejde parser — testovatelnost jazyka),
`GET /schema` publikuje slovník jazyka: predikáty (= slovník grafu), role, typy
uzlů a tázací slova → díry. Web i CLI se dotazují TUDY — jeden vstupní bod.
`--model` = cesta ke graph.pkl (graf je model služby).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import serve, parse_args   # noqa: E402


def make_routes(answerer):
    """POST a GET routy nad hotovým answererem (injektování usnadňuje testy).

    Returns:
        tuple: (posts: dict cesta→fn, gets: dict cesta→fn).
    """
    from jellyai.answerer.pattern import pattern_to_json, pattern_from_json
    from jellyai.lang import current

    def query(payload):
        answer = answerer.answer(payload["question"], [],
                                 temperature=float(payload.get("temperature", 0.0)))
        docs = sorted(answerer.source_context.scores.items(),
                      key=lambda kv: -kv[1])[:5]
        return {"answer": answer.text, "sources": answer.sources,
                "trace": answer.trace, "alternatives": answer.alternatives,
                "pattern": pattern_to_json(answerer.last_pattern),
                "activation": {"nodes": dict(answerer.context.scores),
                               "docs": [[d, round(v, 4)] for d, v in docs]}}

    def graphql(payload):
        topic, values, fact = answerer.run_pattern(pattern_from_json(payload))
        return {"answer": ", ".join(values), "values": values, "topic": topic,
                "fact": fact and {"predicate": fact.predicate,
                                  "participants": [[p.role, p.node]
                                                   for p in fact.participants]}}

    def schema(_payload):
        graph = answerer.graph
        return {"predicates": sorted({f.predicate for f in graph.facts.values()}),
                "roles": ["subj", "obj", "loc", "time", "num", "pred", "attr",
                          "theme", "val"],
                "node_types": sorted({n.type for n in graph.nodes.values()}),
                "holes": {k: list(v[:2]) for k, v
                          in sorted(current()["interrogatives"].items())}}

    return {"/query": query, "/graphql": graphql}, {"/schema": schema}


def main():
    args = parse_args()
    from config import Config
    from jellyai.tasks import make_graph_answerer
    config = Config()
    config.graph.graph_path = args.model
    posts, gets = make_routes(make_graph_answerer(config))
    serve(args.host, args.port, posts, gets=gets)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ověř PASS + ruční curl**

Run: `.venv/bin/python -m pytest tests/test_query_service.py tests/test_services.py -q` → PASS.
Ruční akceptace (spec §5d) nad ostrým grafem — **nejdřív zabij běžící instance** (`pkill -f query_service.py`):

```bash
.venv/bin/python services/query_service.py --port 8084 --model data/graph.pkl &
sleep 5
curl -s -X POST localhost:8084/query -d '{"question": "Kdo napsal R.U.R.?"}'
curl -s -X POST localhost:8084/graphql -d '{"predicate":"napsat","known":[["obj","R.U.R."]],"hole":"subj"}'
curl -s localhost:8084/schema | head -c 400
kill %1
```

Expected: první dva vrátí odpověď s „Čapek"; schema vypíše predikáty grafu.

- [ ] **Step 5: Commit**

```bash
git add services/_common.py services/query_service.py config.py tests/test_query_service.py
git commit -m "feat(api): pseudo-QL jako REST/graphQL služba — /query, /graphql, /schema (spec §5d)"
```

---

### Task 13: Klient + web a CLI dotazují přes API

Jeden vstupní bod: `QueryServiceClient` (líný start subprocesu, vzor `UfalClient`), `QueryServiceAnswerer` (Answerer nad HTTP), `cmd_web` čte odpověď/trasu/aktivaci z API odpovědi, pipeline mode `graph` jde přes API.

**Files:**
- Create: `jellyai/query_client.py`, `jellyai/answerer/remote.py`
- Modify: `jellyai/pipeline.py:39-46` (`_make_answerer` mode graph), `cli.py:268-359` (`cmd_web`)
- Test: `tests/test_query_client.py` (nový), `tests/test_cli_web.py` (přepis na fake klienta)

**Interfaces:**
- Produces: `QueryServiceClient(services_config, graph_path)` s metodami `query(question, temperature=0.0) -> dict`, `graphql(payload) -> dict`, `schema() -> dict`, `close()`; `QueryServiceAnswerer(client)` (Answerer). `cmd_web(config, view=None, client=None)` — injektovatelný klient pro testy.
- Consumes: `_ServiceHandle`, `_post` z `jellyai/ufal_client.py`; response tvar z Tasku 12.

- [ ] **Step 1: Failing testy**

```python
# tests/test_query_client.py
"""Klient pseudo-QL služby + Answerer nad HTTP (jeden vstupní bod, spec §5d)."""
from config import ServicesConfig
from jellyai.answerer.remote import QueryServiceAnswerer


class _FakeClient:
    def query(self, question, temperature=0.0):
        return {"answer": "Božena Němcová", "sources": ["graf"],
                "trace": {"topic": "Babička", "predicate": "napsat",
                          "fact": "f1", "answer": "Božena Němcová"},
                "alternatives": [], "pattern": {"predicate": "napsat"},
                "activation": {"nodes": {"Babička": 1.4}, "docs": []}}


def test_remote_answerer_wraps_api_response():
    ans = QueryServiceAnswerer(_FakeClient()).answer("Kdo napsal Babičku?", [])
    assert ans.text == "Božena Němcová"
    assert ans.sources == ["graf"] and ans.trace["predicate"] == "napsat"
    assert ans.score == 1.0


def test_client_lazy_start_and_query(tmp_path):
    """Klient službu líně nastartuje (subprocess) a POSTne dotaz."""
    import socket
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.query_client import QueryServiceClient
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    graph_path = str(tmp_path / "graph.pkl")
    g.save(graph_path)
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close()
    client = QueryServiceClient(ServicesConfig(query_port=port), graph_path)
    try:
        out = client.query("Kdo napsal Babičku?")
        assert "Božena Němcová" in out["answer"]
        assert "napsat" in client.schema()["predicates"]
    finally:
        client.close()
```

`tests/test_cli_web.py::test_cmd_web_wires_prompt_to_ask` přepiš: místo monkeypatche `UfalClient` injektuj fake klienta:

```python
    class FakeQueryClient:
        def query(self, question, temperature=0.0):
            return {"answer": "Božena Němcová", "sources": ["graf"],
                    "trace": {"topic": "Babička", "predicate": "napsat",
                              "fact": "f1", "answer": "Božena Němcová"},
                    "alternatives": ["Ratibořice"],
                    "pattern": {"predicate": "napsat"},
                    "activation": {"nodes": {"Babička": 1.4,
                                             "Božena Němcová": 2.0},
                                   "docs": [["kniha_babicka", 2.1]]}}
        def close(self): pass

    view = FakeView()
    cmd_web(cfg, view=view, client=FakeQueryClient())
    view.cb(q)
    assert any("Božena Němcová" in line for line in view.written)
    assert "Božena Němcová" in view.updated
```

(FakeView dostane i `write_docs(ranked)` metodu — přidej `def write_docs(self, ranked): self.docs = ranked`.)

- [ ] **Step 2: Ověř FAIL**

Run: `.venv/bin/python -m pytest tests/test_query_client.py tests/test_cli_web.py -v` → ImportError/TypeError.

- [ ] **Step 3: Implementace**

`jellyai/query_client.py`:

```python
"""Klient pseudo-QL služby — líný start, /health, JSON přes urllib.

Týž životní cyklus jako ÚFAL služby (`_ServiceHandle`): služba se spustí při
první potřebě jako subprocess na localhostu a na konci se složí. Web i CLI
dotazují graf VÝHRADNĚ tudy — jazyk je oddělený a testovatelný přes HTTP.
"""

import json
import urllib.request

from jellyai.ufal_client import _ServiceHandle, _post


class QueryServiceClient:
    """HTTP klient k pseudo-QL službě (`services/query_service.py`)."""

    def __init__(self, services_config, graph_path):
        self.config = services_config
        self.graph_path = graph_path
        self._handle = None

    def _ensure(self):
        if self._handle is None:
            self._handle = _ServiceHandle(
                "services/query_service.py", self.graph_path,
                self.config.host, self.config.query_port,
                self.config.startup_timeout)

    def query(self, question, temperature=0.0):
        """POST /query — přirozený dotaz → odpověď + pattern + trasa + aktivace."""
        self._ensure()
        return _post(self.config.host, self.config.query_port, "/query",
                     {"question": question, "temperature": temperature})

    def graphql(self, payload):
        """POST /graphql — přímý pseudo-QL pattern (obejde parser)."""
        self._ensure()
        return _post(self.config.host, self.config.query_port, "/graphql", payload)

    def schema(self):
        """GET /schema — publikovaný slovník jazyka (predikáty/role/typy/díry)."""
        self._ensure()
        url = f"http://{self.config.host}:{self.config.query_port}/schema"
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.load(resp)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
```

`jellyai/answerer/remote.py`:

```python
"""Answerer nad pseudo-QL API — web i CLI mají jeden vstupní bod (spec §5d)."""

from jellyai.answerer.base import Answer, Answerer


class QueryServiceAnswerer(Answerer):
    """Deleguje odpovídání na HTTP službu (`QueryServiceClient`)."""

    def __init__(self, client):
        self.client = client

    def answer(self, question, retrieved, *, temperature=0.0):
        out = self.client.query(question, temperature=temperature)
        return Answer(text=out["answer"], sources=out.get("sources", []),
                      score=1.0 if out.get("trace") else 0.0,
                      alternatives=out.get("alternatives", []),
                      trace=out.get("trace"))
```

`jellyai/pipeline.py` — větev `mode == "graph"` nahraď:

```python
    if config.answerer.mode == "graph":
        from jellyai.query_client import QueryServiceClient
        from jellyai.answerer.remote import QueryServiceAnswerer
        client = QueryServiceClient(config.services, config.graph.graph_path)
        return QueryServiceAnswerer(client)
```

`cli.py:cmd_web(config, view=None, client=None)` — místo `make_graph_answerer` klient:

```python
    from jellyai.query_client import QueryServiceClient
    if client is None:
        client = QueryServiceClient(config.services, config.graph.graph_path)
```

Nominativizace popisků (`base_form_label`) si drží VLASTNÍ `UfalClient(config.services)` (MorphoDiTa — build-side vizualizace, ne query). `on_query` přepiš na čtení z API odpovědi:

```python
    def on_query(question):
        out = client.query(question, temperature=0.35)
        scores = {k: float(v) for k, v in out["activation"]["nodes"].items()}
        trace = out.get("trace")
        path = [trace["topic"], trace["answer"]] if trace else []
        state = pulse.ignite(scores, path)
        for node_id, bright in state["sizes"].items():
            view.update_node(node_id, size=1.0 + bright,
                             **{"aktivace (attention)": f"{bright:.2f}"})
        for node_id in state["extinguish"]:
            view.update_node(node_id, size=1.0,
                             **{"aktivace (attention)": "0"})
        if state["sizes"] and hasattr(view, "focus"):
            view.focus(max(state["sizes"], key=state["sizes"].get))
        if hasattr(view, "write_docs"):
            view.write_docs([tuple(d) for d in out["activation"]["docs"]])
        reply = f"❓ {question}\n💬 {out['answer']}"
        if out.get("alternatives"):
            reply += f"\n   souvislosti: {', '.join(out['alternatives'][:4])}"
        view.write(reply)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:6]
        trace_line = (f"{trace['topic']} ─[{trace['predicate']}]→ {trace['answer']}"
                      if trace else "žádná (bez odpovědi v grafu → fallback)")
        print(
            f"\n❓ {question}"
            f"\n💬 {out['answer']}   (zdroj: {', '.join(out.get('sources', [])) or '—'})"
            f"\n   trasa: {trace_line}"
            f"\n   souvislosti (fuzzy): {', '.join(out.get('alternatives', [])) or '—'}"
            f"\n   aktivace (kontext): "
            f"{', '.join(f'{n}={v:.2f}' for n, v in top) or '—'}",
            flush=True)
        return out["answer"]
```

(vizualizační smyčka `animate`/`view.every`/`open_terminal`/`serve` zůstává beze změny — mění se jen zdroj dat: API odpověď místo vnitřního answereru.)

- [ ] **Step 4: Ověř PASS + ruční web**

Run: `.venv/bin/python -m pytest tests/test_query_client.py tests/test_cli_web.py tests/test_pipeline.py -q` → PASS.
Run: `.venv/bin/python -m pytest -q` → celé zelené.
Ruční (spec akceptace §5d + poznámka o restartu): `pkill -f "cli.py web" ; pkill -f query_service.py`, pak `./jelly web` → prohlížeč se otevře, polož „Kdo napsal R.U.R.?" a „Kdo byl bratr autora, který napsal R.U.R.?" — odpověď + trasa + rozsvícení grafu + panel dokumentů fungují.

- [ ] **Step 5: Commit**

```bash
git add jellyai/query_client.py jellyai/answerer/remote.py jellyai/pipeline.py cli.py tests/test_query_client.py tests/test_cli_web.py
git commit -m "feat(api): web i CLI dotazují přes pseudo-QL službu — jeden vstupní bod (spec §5d)"
```

---

### Task 13b: Responzivní QL — clarifikace při nejistém rozlišení (návrh uživatele, 2026-07-17)

> **Návrhový princip (uživatel):** QL staví na DIALOGU s uživatelem, ne na „fiklech" — prioritou je aktivace správných uzlů, ne vydolování odpovědi heuristikou. Když je jistota nízká, správná odpověď je otázka (nebo poctivé „nezaostřil jsem"), nikdy další hádací vrstva. Guessing patra answereru (kontext asociace apod.) nesmí přebít clarifikaci.

Pseudo-QL není databázový QL — smí vést dialog. Když je rozlišení subjektu **nejisté** (podspecifikovaný dotaz: holé příjmení „Kdo je Němec?" → víc kmenových clusterů, jen slabá patra bez exact/ins shody), QL místo mlčení/hádání **požádá o upřesnění** a kandidáty **rozsvítí v aktivačním poli** — navazující odpověď uživatele pak téma zaostří (lepší aktivace). Poctivost zůstává: skutečně neznámé („Ludvík Němec" — vedoucí sirotek) dál „nenašel"; clarifikace jen při VÍCE kandidátech s rovnocennou slabou evidencí.

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py` (`_resolve_topic` hlásí nejistotu — např. `self.last_ambiguity = {"term", "candidates"}` když top kandidát vyhrál jen slabými patry a ≥2 kandidáti z RŮZNÝCH loose-clusterů mají rovnocennou jmennou evidenci; `answer()` při nejistotě u identity/entity otázky vrátí clarifikační Answer + warm kandidátů)
- Modify: `jellyai/answerer/base.py` (`Answer.clarify: dict|None = None` — `{"prompt", "candidates"}`)
- Modify: `jellyai/lang/cs.json` (`"clarify_prompt": "Upřesni prosím „{term}" — myslíš: {candidates}?"` — text je jazykové datum, ne kód)
- Modify: `services/query_service.py` (`/query` odpověď + pole `"clarify"`), `cli.py cmd_web` (zobrazení QL otázky v konzoli)
- Test: `tests/test_clarify.py`; `benchmark/etalon.jsonl` řádek kategorie „dialog-upřesnění"

**Interfaces:**
- Produces: `Answer.clarify = {"prompt": str, "candidates": [id…]} | None`; `/query` → `{"clarify": {...}|null, "assurance": float, …}`. Kandidáti clarifikace se zahřejí (`context.warm`, ≥0.5), aby navazující otázka rozřešila proti svítícímu poli.
- **QueryAssurance skóre** (návrh uživatele): číselná jistota zaměření subjektu, 0–1. Výpočet z evidence rozlišení: váhy pater (exact 1.0 > ins 0.8 > stem 0.6 > da 0.4 > loose 0.25) × pokrytí termů, + aktivace vítěze z konverzačního těžiště (`context.scores`), − penalizace za rovnocenné kandidáty z jiných `_loose`-clusterů. Práh `GraphConfig.assurance_threshold` (default ladit etalonem, start 0.5). Pod prahem → clarify; nad prahem → odpověď. **Dialogová smyčka „ptej se, dokud není jistota" vzniká přirozeně**: upřesnění uživatele jde zase přes `/query`, přidané termy + svítící kandidáti skóre zvednou — žádný stavový automat.
- Skóre se vrací v API (`"assurance"`) i bez clarifikace — observabilita, ladění prahu měřením.
- **Rekurzivní ostření** (upřesnění uživatele): QL je hybrid statického odpovědního automatu a dialogu — nejistota může vzniknout i UVNITŘ rekurze (`_solve` vnořeného `SubQuery` řeší svůj term týmž `_resolve_topic`, tedy týmž assurance mechanismem). Clarify pak nese i cestu („v dotazu na bratra autora: kterého autora myslíš — A, B?"); ostří se kterýkoli uzel dotazu, hloubka libovolná.
- **Upřímný terminál** (upřesnění uživatele): když skóre nedosáhne prahu a kandidáty nejde dál ostřit (nebo uživatel neupřesnil), QL NEhádá ani nemlčí — odpoví poctivě: „Nepodařilo se mi zaostřit dotaz dostatečně. Nejbližší možné odpovědi jsou: A, B…" (kandidáti z vějíře, seřazení skóre; zároveň se zahřejí — potrava pro aktivaci). Text = `cs.json` klíč `"assurance_fail"` (šablona s `{candidates}`), vedle `"clarify_prompt"`. Etalon: expect na podřetězec „zaostřit" / kandidáty.
- **Nabídka zaostření při homonymním vějíři** (upřesnění uživatele): třetí spouštěč clarifikace — VÍCE kandidátů s rovnocennou jmennou/aktivační evidencí („Kdo je Čapek?" → Karel i Josef mají exact hit příjmení, různé `_loose`-klíče). QL nabídne témata: „Mám možnost se věnovat tématu: Karel Čapek, nebo Josef Čapek — kam mám zaostřit?" (šablona `"focus_offer"` v cs.json, kandidáti = vějíř seřazený skóre/aktivací, max ~4). Dnešní tiché rozhodnutí vahou se pod prahem jistoty mění na dialog; vybraný směr uživatele = další /query tah, který zaostří aktivaci.

- [ ] Failing test: `_resolve_topic` na grafu se 2 uzly „Božena Němcová"/„Jan Němec" pro term „Němec" → answer("Kdo je Němec?") vrátí `clarify` s oběma kandidáty a text z `clarify_prompt`; kandidáti svítí v `context.scores`. Kontrolní: „Kdo je Božena Němcová?" clarify nemá; „Kdo je Ludvík Němec?" zůstává „nenašel" bez clarify.
- [ ] Implementace (viz Files) + etalon řádek: `{"dialog": ["Kdo je Němec?"], "expect": ["Upřesni"], "cat": "dialog-upřesnění"}` — přesný expect přizpůsob šabloně z cs.json po měření.
- [ ] `pytest -q`, oba benchmarky (žádná regrese — zvlášť poctivost řádky 18/19), commit `feat(ql): responzivní clarifikace — nejisté rozlišení se ptá a rozsvěcí kandidáty`.

**Pozn.:** `/graphql` (přímý pattern) clarifikaci nemá — tam je jazyk deterministický; dialog patří přirozené cestě `/query`.

---

### Task 14: Parita → templates-only, UDPipe pryč z query strany (spec §6.1)

**GATE: nezačínej, dokud `run_etalon.py --mode templates` nedá JÁDRO plný počet.** Pokud po Tasku 13 nedává, iteruj (§5c): každý FAIL řádek → diagnostikuj (vypiš `last_pattern` + `_resolve_topic` skóre pro dotaz), oprav v parseru/rozlišení, přidej unit test, přeměř. Teprve pak:

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py` (odstranit UDPipe cestu + `query_mode`)
- Modify: `jellyai/answerer/pattern.py` (smazat UDPipe parsing)
- Modify: `config.py`, `jellyai/tasks.py`, `benchmark/run_etalon.py` (knob pryč), `jellyai/__init__.py`
- Modify: `tests/test_pattern.py`, `tests/test_query_mode.py`, `tests/test_topic_resolve.py` (FakeUfalClient parse data už netřeba), `tests/test_query_service.py` + `tests/test_query_client.py` (konstruktory bez `query_mode`)
- Modify: `benchmark/etalon.jsonl` (gap řádek z Tasku 9 se stane jádrem — smaž jeho klíč `gap`)

**Interfaces:**
- Produces: `GraphAnswerer(graph, client, fallback, *, context_decay=…, spread_depth=…, spread_falloff=…)` — bez `query_mode`; `client` zůstává jen pro MorphoDiTa nominativizaci. `pattern.py` = `Pattern`, `SubQuery`, `pattern_to_json`, `pattern_from_json` — nic víc.

- [ ] **Step 1: Gate měření**

Run: `.venv/bin/python benchmark/run_etalon.py --mode templates`
Expected: JÁDRO plný počet (28/28 po přidaných řádcích). Číslo zapiš.

- [ ] **Step 2: Odstranění UDPipe z query strany**

`graph_answerer.py`:
- smaž importy `analyze_question` a `question_pattern` (Pattern/SubQuery zůstávají),
- `answer()`: rozbor jen šablonami —

```python
        query = build_query(question, self._predicates, self._span_is_node)
        qa = query if query is not None else Query()
        pat = qa.pattern if query is not None else Pattern()
        self.last_pattern = pat
        topic, values, fact = self._pattern_answer(question, pat, qa)
```

- smaž `query_mode` z `__init__` (a všechna čtení),
- docstring modulu uprav (analýza = šablony, ne `analyze_question`).

`pattern.py` — smaž `question_pattern`, `_parse_sent`, `_known_of`, `_genitive_child`, `_rel_clause`, `_entity_term`, `_HOLE`, `_DEPREL_ROLE`, `_CONTENT`, `_DATE_PARTS` a importy `_clean_lemma`/`current`. Zůstává: docstring (uprav), `Pattern`, `SubQuery`, `pattern_to_json`, `pattern_from_json`.

`config.py` — smaž `GraphConfig.query_mode`; `jellyai/tasks.py` — smaž předávání; `benchmark/run_etalon.py` — smaž `--mode` (jediná cesta); `jellyai/__init__.py` — `analyze_question` v exportech nech (TemplateAnswerer V3 ho dál používá), ale ověř, že nic z `pattern.py` smazaného se neexportuje.

`tests/test_pattern.py` — smaž testy volající `question_pattern` (sémantiku — vnořený SubQuery, genitiv, flat jména — už normují šablonové testy v `test_query.py`; pokud některý případ chybí, přepiš ho na `build_query` ekvivalent, NEmaž bez náhrady). `tests/test_query_mode.py` — smaž hybrid/udpipe testy, nech templates (bez `query_mode` argumentu). `tests/test_topic_resolve.py` — `FakeUfalClient(parse=…)` data smí zůstat (nevolají se), ale otázky musí projít šablonami — spusť a oprav, co spadne.

`benchmark/etalon.jsonl` — u řádku „Kdo byl bratr autora, ktery napsal R.U.R.?" smaž klíč `gap` (teď je to jádro).

- [ ] **Step 3: Ověř vše**

Run: `.venv/bin/python -m pytest -q` → celé zelené.
Run: `.venv/bin/python benchmark/run_etalon.py` → JÁDRO plný počet (jediná cesta = šablony).
Run: `.venv/bin/python benchmark/run_coverage.py` → `UZLY (…)` beze změny proti baseline z Tasku 1.
Ověř, že UDPipe se u otázek nespouští: `grep -rn "question_pattern\|analyze_question" jellyai/answerer/graph_answerer.py` → nic.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(query)!: pseudo-QL jediná query cesta — UDPipe z query strany odstraněn (parita 28/28)"
```

---

### Task 15: Finální verifikace, dokumentace, památka

- [ ] **Step 1: Kompletní měření (verification-before-completion)**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python benchmark/run_etalon.py
.venv/bin/python benchmark/run_coverage.py
```

Expected: pytest zelený; JÁDRO plný počet; coverage/UZLY beze zhoršení. Výstupy VLOŽ do závěrečného shrnutí (čísla, ne tvrzení).

- [ ] **Step 2: Ruční scénář §6.5 v prohlížeči**

`pkill -f "cli.py web" ; pkill -f query_service.py` (poznámka o restartu!), pak `./jelly web`; polož postupně: „Kdo je jezis?", „Kde se narodil Jezis?", „Kdo byl bratr autora, který napsal R.U.R.?" (s diakritikou i bez). Odpovědi věcně správné, trasa + aktivace + panel dokumentů živé. Paralelně curl na `:8084/query`, `/graphql`, `/schema` (viz Task 12 Step 4).

- [ ] **Step 3: Dokumentace**

- Tento plán: odškrtej checkboxy, doplň finální čísla k baseline sekci.
- `docs/superpowers/specs/2026-07-17-pseudo-ql.md`: na konec přidej řádek `> HOTOVO <datum>: JÁDRO X/X šablonami, UDPipe z query strany odstraněn, API :8084.`
- Aktualizuj paměť projektu (`memory/jellyai3-fact-graph.md`): pseudo-QL primární, query bez UDPipe, API endpointy, etalon rozšířen na 28 řádků jádra.

- [ ] **Step 4: Závěrečný commit**

```bash
git add -A
git commit -m "docs: pseudo-QL uzavřen — parita etalonu šablonami, API publikováno, dialog v prohlížeči"
```

---

## Poznámky pro exekutora

- **Pořadí měření po KAŽDÉM tasku:** `pytest -q` → `run_etalon.py` (default režim) → `run_etalon.py --mode templates` (dokud knob existuje) → při dotyku extrakce/grafu i `run_coverage.py`. Čísla piš do commit message.
- **Když etalon spadne pod plný počet v default režimu:** parser vrátil chybný Pattern místo None. Najdi dotaz, vypiš `build_query(q, predicates, is_node)` a oprav podmínku, ať vrací None — fallback (dokud existuje) se postará.
- **Ladění na živém grafu:** rychlá sonda vzoru + rozlišení =
  `.venv/bin/python -c "from config import Config; from jellyai.tasks import make_graph_answerer; from jellyai.answerer.query import build_query; a = make_graph_answerer(Config()); print(build_query('OTÁZKA?', a._predicates, a._span_is_node))"`.
- Očekávaná čísla `--mode templates` po tascích jsou ODHAD trajektorie (16 → ~19 → ~20 → … → plný počet); skutečné číslo vždy přeměř a zapiš. Rozdíl oproti odhadu není chyba — chyba je nezměřit.
- Etalonové expecty nikdy neměň tak, aby prošel špatný výsledek; expect se mění jen, když je věcně správnější odpověď (a pak to zdůvodni v commitu).
