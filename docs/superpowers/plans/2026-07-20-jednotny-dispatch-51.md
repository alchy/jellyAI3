# Jednotný dispatch #51 — Implementation Plan (s postřehy z review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realizovat spec `2026-07-20-jednotny-dispatch-51.md` (jeden
kompilát, rodiny vyrok+prikaz, P1 shadow-first po rodinách) a vetkat
do cesty relevantní postřehy z `docs/architektura-web/postrehy-refaktor.md`
(1.2 jeden deck, 1.3+1.4 recognize vrací výsledek, 4.1 instance_lit ke
grafu, 4.5 harness bez literálů, 3.1 rozklad _turn — vyplyne závěrem).

**Architecture:** Fáze 0 = úklidové postřehy sloužící dispatch (malé,
mechanické, každý s paritou). Fáze 1 = rysy tahu + rodina `vyrok`
v kompilátu + shadow rovina výroků. Fáze 2 = přepnutí výroků. Fáze 3 =
rysové příkazové karty + shadow + přepnutí po druzích. Fáze 4 = zbytek
(recall, focus-shift), smazání ručního pořadí, rozklad `_turn` na fáze.

**VĚDOMĚ ODLOŽENO:** postřeh 2.1 (TurnResult místo 9 side-channel
atributů) — největší zásah do answereru; koncepčně je to zárodek
zaparkovaného ODPOVĚDNÍHO grafu (efemérní reifikace tahu), patří tam,
ne do dispatch přepínání. Zapsat do BACKLOGu k odpovědnímu grafu.

**KOREKCE SPECU (zapsat při uzávěrce):** příkazové karty v1 RYSOVÉ
(rys `cmd:*` počítaný z dnešních frázových tabulek — zachová
substringovou sémantiku, parita konstrukcí); literálové vzory `%{…}`
by sémantiku měnily (tokeny × substring) a jsou až pozdější měřené
zpřesnění.

## Global Constraints

Stejné jako plán `2026-07-20-otazkovy-graf-e1-e2.md`: větev
`jednotny-dispatch`, heredoc commity, české uvozovky v JSON, celá sada
před každým commitem (pytest + 5 benchmarků + run_qgraph tiers
i weights), žádný pokles; dialog benchmark = bitový parity gate
přepínacích kroků; žádný interpret v JSONu; vakuová logika (past 2)
u množinových podmínek.

---

### Task 0a: instance_lit ke grafu (postřeh 4.1)

**Files:** Modify `jellyai/graph/graph.py` (přesun funkce),
`jellyai/iris/qgraph.py` (smazat), `jellyai/answerer/graph_answerer.py`
(2 importy), `tests/test_qgraph.py` (import v testu).

**Interfaces:** Produces `jellyai.graph.graph.instance_lit(predicate,
hole_role, roles_of)` — beze změny signatury/sémantiky; qgraph
re-export NEZAVÁDĚT (zákon 6).

- [ ] Přesuň funkci `instance_lit` z qgraph.py do graph.py (hned za
  `FactGraph.predicate_roles` jako modulovou funkci, docstring beze
  změny); uprav `from jellyai.iris.qgraph import instance_lit` →
  `from jellyai.graph.graph import instance_lit` v graph_answerer.py
  (2 místa — `_empty_role_answer` a kontextový guard; přesuň import
  na hlavičku modulu, cyklus už nehrozí) a v tests/test_qgraph.py.
- [ ] Celá sada. Expected: beze změny čísel. Commit
  (`refactor(graph): instance_lit patří ke grafu (postřeh 4.1)`).

### Task 0b: jeden deck za proces (postřeh 1.2)

**Files:** Modify `jellyai/iris/patterns.py` (+`shared_deck`),
`jellyai/answerer/query.py` (`_query_deck` deleguje),
`jellyai/iris/automaton.py` (`__init__` bere shared_deck).

**Interfaces:** Produces `shared_deck(language="cs") -> PatternDeck`
— modulový cache v patterns.py; obě dnešní instance ho čtou.

- [ ] Do patterns.py přidej:

```python
_SHARED = {}


def shared_deck(language="cs"):
    """JEDEN deck za proces (postřeh 1.2) — obě cesty (automat, dotazy)
    čtou tytéž karty; test smí cache vyprázdnit (`_SHARED.clear()`)."""
    if language not in _SHARED:
        deck = PatternDeck.for_language(language)
        deck.load()
        _SHARED[language] = deck
    return _SHARED[language]
```

  V query.py `_query_deck()` vrací `shared_deck("cs")` (globál
  `_QUERY_DECK` smaž); v automatonu `if deck is None: deck =
  shared_deck(language)` místo konstrukce+load (parametr `deck`
  v testech funguje dál).
- [ ] Celá sada; commit (`refactor(iris): jeden sdílený deck (postřeh 1.2)`).

### Task 0c: recognize vrací výsledek (postřehy 1.3+1.4)

**Files:** Modify `jellyai/iris/claims.py`, `jellyai/iris/qgraph.py`
(illuminate — truthy test), `jellyai/iris/automaton.py`
(`_expert_turn(worker, text, found)` + handlery berou found),
`tests/test_qgraph.py` (přizpůsobit asserty recognize).

**Interfaces:** `ExpertClaim.recognize(text, now) -> object | None`
(payload: metron `(expression, result)`, chronos `str` odpověď,
meta-focus `True`); illuminate: `if claim.recognize(...)` zůstává
(truthy). Automaton: `_metron_query(text, found=None)` a
`_clock_response(text, found=None)` — found předané z dispatch smyčky
(druhý výpočet mizí); bez found si spočtou samy (přímá volání testů).

- [ ] Uprav claims.py: `_metron` vrací `compute(text)`, `_chronos`
  vrací `clock_answer(text, now)`, `_meta_focus` beze změny (bool).
  V _turn smyčce: `found = claim.recognize(text, self.clock())` /
  `if not found: continue` / `handled = self._expert_turn(claim.worker,
  text, found)`. `_expert_turn` předá found handlerům; `_metron_query`
  použije found místo `compute(text)` (fallback `found or compute(text)`),
  `_clock_response` použije `found or clock_answer(...)`.
- [ ] Celá sada (dispatch parity!); commit
  (`refactor(qgraph): recognize claimu vrací výsledek — konec dvojího výpočtu`).

### Task 0d: harness bez literálů uzlů (postřeh 4.5)

**Files:** Modify `benchmark/run_qgraph.py` (`_actual_route`).

- [ ] Mapu komponent → jméno uzlu odvoď z registru:

```python
from jellyai.iris.claims import default_claims
_WORKER_NODES = {c.worker: c.name for c in default_claims()}
```

  a v `_actual_route` nahraď literály `_WORKER_NODES["metron"]`,
  `_WORKER_NODES["iris"]`, `_WORKER_NODES["chronos"]`.
- [ ] run_qgraph obě varianty beze změny; commit
  (`refactor(benchmark): jména worker uzlů z registru claimů`).

---

### Task 1a: rysy tahu (otaznik + výrokové rysy jednou funkcí)

**Files:** Modify `jellyai/iris/subsystems/mnemos.py`
(`utterance_features` beze změny), Create funkci `turn_features`
v `jellyai/iris/qgraph.py`; Test `tests/test_qgraph.py`.

**Interfaces:** `turn_features(text, tagged=None) -> frozenset[str]`
= `utterance_features(tagged)` ∪ {"otaznik" pokud „?" in text} ∪
`command_features(text)` (Task 3a doplní; teď vrací jen prvé dvě).

- [ ] Červený test:

```python
def test_turn_features_nese_otaznik_i_vyrokove_rysy():
    from jellyai.iris.qgraph import turn_features
    assert "otaznik" in turn_features("Prší?")
    features = turn_features("Venku prší.")
    assert "otaznik" not in features
    assert "l_verb" not in features          # prézens, ne l-tvar
    assert "first_person" in turn_features("Dnes jsem měl knedlíky.")
```

- [ ] Implementace v qgraph.py:

```python
def turn_features(text, tagged=None):
    """Povrchové + výrokové rysy tahu (#51) — jedna funkce pro
    osvětlení rodin i karty (requires/forbids čtou TYTÉŽ rysy,
    kterými dnes vybírá parse_statement)."""
    from jellyai.iris.subsystems.mnemos import utterance_features
    if tagged is None:
        tagged = classify(text, is_node=None)
    features = set(utterance_features(tagged))
    if "?" in text:
        features.add("otaznik")
    return frozenset(features)
```

  (Ověř přesnou signaturu `utterance_features` — bere tagged tokeny;
  pokud bere i text, předej ho.)
- [ ] pytest + commit (`feat(qgraph): #51 rysy tahu (turn_features)`).

### Task 1b: výrokové karty zakážou otazník (data)

**Files:** Modify všechny `jellyai/iris/patterns/cs/statement-*.json
+ karty s event utterance.statement (vypiš:
`grep -l "utterance.statement" jellyai/iris/patterns/cs/*.json`).

- [ ] Každé výrokové kartě přidej do trigger `"forbids": ["otaznik"]`
  (má-li už forbids, rozšiř seznam). Dnešní výběr to NEZMĚNÍ
  (parse_statement rys otaznik nepočítá → forbids nikdy neblokuje);
  význam dostane rys při osvětlení rodin (1c). POZOR na české
  uvozovky v teach — soubory needituj ručně přes sed, použij
  json.load/dump s ensure_ascii=False + ručně zkontroluj diff.
- [ ] Celá sada (parita — chování beze změny); commit
  (`data(karty): #51 výrokové karty zakazují otaznik (rys rodiny)`).

### Task 1c: kompilace rodiny vyrok + osvětlení rysy

**Files:** Modify `jellyai/iris/qgraph.py` (compile_qgraph +
illuminate), `jellyai/iris/patterns.py` (extrakce sdíleného jádra
těsnosti — NEduplikovat výběr, postřeh 1.1); Test tests/test_qgraph.py.

**Interfaces:** QNode kind `"vyrok"`, worker `"brana-e"`; rysový uzel
nese `trigger` (requires/forbids) místo pattern. patterns.py Produces
`trigger_specificity(trigger, features) -> int | None` — čisté jádro
rysové těsnosti (bez event/pattern checků), `PatternDeck._specificity`
ho volá (jedna pravda), illuminate též.

- [ ] Červený test:

```python
def test_kompilace_a_osvetleni_rodiny_vyrok():
    qg = _graph()
    vyroky = [n for n in qg.nodes.values() if n.kind == "vyrok"]
    assert vyroky                              # karty statement-* jsou uzly
    lit = illuminate("Venku prší.", qg)
    assert lit and lit[0].kind == "vyrok"      # výrok svítí bez otazníku
    lit = illuminate("Prší?", qg)
    assert not lit or lit[0].kind != "vyrok"   # otaznik výrok zhasíná
```

- [ ] patterns.py: vytáhni z `_specificity` rysové jádro:

```python
def trigger_specificity(trigger, features):
    """Těsnost rysového triggeru nad rysy tahu; None = nesedí.
    JEDNA pravda výběru (postřeh 1.1) — deck i osvětlení grafu."""
    score = 0
    requires = set(trigger.get("requires", ()))
    if not requires <= set(features):
        return None
    forbids = set(trigger.get("forbids", ()))
    if set(features) & forbids:
        return None
    return score + len(requires) + len(forbids)
```

  a `_specificity` po event/pattern/assurance/candidates checkech
  deleguje rysovou část na `trigger_specificity` (chování bitově —
  ověř testy decku).
- [ ] qgraph.py compile: karty `utterance.statement` → uzly
  `QNode(kind="vyrok", worker="brana-e", pattern=trigger.get("pattern"),
  card=…, priority=…)`; RYSOVÉ (bez pattern) nesou celý trigger
  (nový atribut `QNode.trigger = None`). illuminate: spočti
  `features = turn_features(text, tagged)` jednou; vyrok uzly svítí
  (tier 2) — vzorové matcherem jako otázky, rysové přes
  `trigger_specificity(node.trigger, features)`; klíč
  `(2, priorita, těsnost/délka)`. Otázka × výrok: výrokové karty mají
  forbids otaznik (1b), dotazové vzory se na výrocích bez otazníku
  utkají klíčem — SHADOW to změří, nic se nepřepíná.
- [ ] Celá sada (dispatch parity — dnešní roviny se nesmí hnout;
  vyrok uzly zatím jen SVÍTÍ v shadow). Commit
  (`feat(qgraph): #51 rodina vyrok v kompilátu + rysové osvětlení`).

### Task 1d: shadow rovina VÝROKY

**Files:** Modify `jellyai/iris/automaton.py` (`_memorize` přidá
vítěznou výrokovou kartu do `used_patterns`), `benchmark/run_qgraph.py`
(rovina výroků na dnešních mimo-rozsah tazích).

**Interfaces:** `response.used["patterns"]` u zápisového tahu obsahuje
jméno vítězné výrokové karty (z parse výsledku — `statement["kind"]`
nese druh; ověř skutečný klíč:
`grep -n '"kind"' jellyai/iris/subsystems/mnemos.py | head`).

- [ ] `_memorize` doplň used_patterns o kartu každého uloženého
  výroku. Harness: tah, jehož skutečná cesta je zápis (kind
  „memorized"/statement karta v patterns), se měří proti vítězi
  osvětlení rodiny vyrok: `vyrok_agree/vyrok_total`. Výstup přidej do
  QGRAPH SHADOW řádku (`výroky V/V (P %)`).
- [ ] Spusť harness; výsledek je NÁLEZ (ne gate) — neshody vypiš
  a rozhodni: karta×uzel mapování vs. skutečné krádeže. Zapiš do
  spec §Empirie-1. Gate pro Task 2 = 100 % po vyladění (bez
  přepnutí!). Commit
  (`feat(benchmark): #51 shadow rovina výroků (X/Y)` — čísla doplň).

---

### Task 2: PŘEPNUTÍ výroků (gate: shadow 100 % + bitová parita)

**Files:** Modify `jellyai/iris/automaton.py` (_turn).

- [ ] V `_turn` nahraď podmínku vstupu do výrokové větve (dnes
  `elif "?" not in text:` blok — POZOR, kryje i příkazy: přepni JEN
  finální „KONSTATOVÁNÍ" část — `parse_clauses` volání) testem
  „vítěz osvětlení je vyrok uzel": před blokem spočti
  `lit = illuminate(text, self.qgraph, now=self.clock(),
  is_node=self.answerer._span_is_node)` (jednou, ulož pro tah)
  a konstatování prováděj, když `lit and lit[0].kind == "vyrok"`.
  Příkazové/reminder větve zůstávají v „?" bloku beze změny (fáze 3).
- [ ] Celá sada: dialog benchmark BITOVĚ (žádná změna odpovědí),
  vše 100 %. Pokud cokoli klesne → NESHODA výpis, vyšetřit (krádež
  mezi rodinami), neobcházet. Commit
  (`feat(iris): #51 dispatch výroků osvětlením (parity bitová)`).

---

### Task 3a: command_features + příkazové karty (rysové)

**Files:** Modify `jellyai/iris/qgraph.py` (turn_features ∪
command_features), Create `jellyai/iris/patterns/cs/cmd-*.json`
(memorize, forget, reminder, plan, send, focus-shift, recall);
Test tests/test_qgraph.py.

**Interfaces:** `command_features(text) -> set[str]` — rysy
`cmd:memorize|forget|reminder|plan|send|focus|recall` z DNEŠNÍCH
tabulek (memorize_phrases, forget fráze — ověř zdroj
`_forget_command`, reminder_phrases, plan_cancel_words ∪
plan_move_words, send fráze z #54, focus_shift fráze, memory-recall
fráze) — substring po deaccent, PŘESNĚ jako dnešní větve. Karty:
`{"name": "cmd-memorize", "trigger": {"event": "utterance.command",
"priority": 6, "requires": ["cmd:memorize"], "forbids": ["otaznik"]},
"action": {"worker": "mnemos"}, "teach": "…"}` (worker atribut
v action; uzly kind `prikaz`).

- [ ] Červený test: `turn_features("Zapamatuj si, že Roník je pes.")`
  obsahuje `cmd:memorize`; kompilace vyrobí uzly kind `prikaz`;
  osvětlení „Připomeň mi zítra oběd." vybere `cmd-reminder`.
- [ ] Implementace + kompilace `utterance.command` karet na uzly
  (worker z action.worker); illuminate je svítí rysy (jako vyrok).
  Priorita příkazů NAD výroky (příkaz „zapamatuj si…" je zároveň
  konstatováním — dnešní pořadí větví dává příkazům přednost; dej
  kartám vyšší prioritu než výrokovým a SHADOW ověří).
- [ ] Celá sada (nic se nepřepíná); commit.

### Task 3b: shadow rovina PŘÍKAZY + přepnutí po druzích

**Files:** Modify `benchmark/run_qgraph.py`, `jellyai/iris/automaton.py`.

- [ ] Harness: skutečná cesta příkazových tahů (karty
  reminder-set/memory-recall/… v used.patterns + kind) vs. vítězný
  prikaz uzel; rovina `příkazy X/Y`. Vylaď mapování na 100 %
  (nálezy do Empirie).
- [ ] Přepínej PO DRUZÍCH v pořadí: memorize → forget → send →
  plan/reminder (nejsložitější stavy nakonec) → focus-shift → recall.
  Druh = v `_turn` nahraď frázový test druhu podmínkou „vítěz
  osvětlení je cmd-<druh>"; handler (`_memorize_command`,
  `_forget_command`…) volej beze změny; PENDING reminder logika
  (takeover) zůstává PŘED dispatch (stavový tah — pozice, spec §3).
  Po každém druhu: celá sada bitově, commit zvlášť.

### Task 4: zbytek + smazání pořadí + rozklad _turn (postřeh 3.1)

**Files:** Modify `jellyai/iris/automaton.py`; docs (spec Empirie,
BACKLOG #51, ARCHITEKTURA kap. 7.7).

- [ ] Až jsou všechny rodiny přepnuté: smaž ruční pořadí větví
  a rozděl `_turn` na fáze-metody: `_fired()` (hodiny/připomínky),
  `_positional(text)` (stavové kroky: identity/pick/pending),
  `_dispatch(text)` (osvětlení → worker/karta/brána E),
  `_answer_query(text)` (dotazová cesta). Bez změny chování —
  bitová parita VŠEHO.
- [ ] Uzávěrka: Empirie do spec (vč. KOREKCE: rysové příkazové karty
  místo literálových vzorů + proč), BACKLOG #51 → ✓, ARCHITEKTURA
  kap. 7.7 tabulka událostí (přibude utterance.command) a věta
  o jediném dispatchi; aktualizace paměti. Finální commit + merge dle
  volby uživatele.
