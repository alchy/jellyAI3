# Iris (QL stavový automat) — implementační plán

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **STAV (2026-07-17, revize dokumentace):** Fáze 0 a 1 KOMPLETNÍ, Fáze 3
> HOTOVÁ (web tři okna přes REST — commity `0799091`, `b10420e`). Z Fáze 2
> hotové: Mnemos + perzistence deníku (`da07043`, `004fc5a`), Chronos —
> primitiva, intervaly, hodinové odpovědi, rozsvěcení časových uzlů
> (`4452bd7`, `57b45b2`, `58f88b6`), focus-shift (`8a244df`), kartový audit
> (`58f88b6`), dialogový benchmark `run_dialog.py` 9/9 (`bfa6975`). Merge
> fáze: `b0ff114`. Otevřené zbytky Fází 2/4/5/6 sleduje `docs/BACKLOG.md`
> (benefit-výběr karet, karty data-overflow/clarify-*, sharpener, čistý řez,
> hybridní aktivace). Metriky při merge `b0ff114`: etalon 28/28, focus 12/12,
> dialog 9/9 (vše 100 %).

**Goal:** Postavit Iris — modulární, jazykově agnostický stavový automat pro zaostření aktivace uzlů grafu (dialogem s uživatelem), publikovaný přes REST API s metadaty, řízený JSON pattern-kartami.

**Architecture:** Malé OOP jádro (`jellyai/iris/`), chování v pattern-kartách (1 JSON = 1 vzor: trigger→dialog→akce→teach), dotazovací moduly jako pluginy (první = hotový pseudo-QL parser, parita 28/28). QueryAssurance skóre řídí přechody: nad prahem odpověz, pod prahem dialog (clarifikace/nabídka zaostření), terminálně poctivé „nezaostřil jsem + nejbližší kandidáti". Viz spec `2026-07-17-ql-automat.md`.

**Tech Stack:** Python 3.11, pytest, `http.server` (vzor `services/_common.py`), JSON konfigurace. Žádné nové závislosti.

## Global Constraints

- Etalon (`benchmark/run_etalon.py`) NIKDY pod 100 % jader — guardrail každého tasku.
- Dialog > figly ([[ql-dialog-first]]): pod prahem jistoty se ptát, ne hádat; guessing patra nesmí přebít clarifikaci.
- Jazyk/dialogy jen v JSON (`jellyai/lang/`, `jellyai/iris/patterns/<lang>/`) — kód bez češtiny.
- Výukové docstringy („polopatě") u každé třídy/metody — konvence projektu.
- Determinismus; API localhost; zpětná kompatibilita se NEdrží (graf i API jsou prototyp).
- Po každé změně restart běžícího webu/API (`pkill -f "cli.py web"; pkill -f query_service`), běžící instance drží starý kód.
- Každý task končí: `pytest -q` + `run_etalon.py` (+ `run_coverage.py` při dotyku extrakce, + `run_focus.py` od Fáze 4). Čísla do commit message.

## Výchozí stav (změřeno 2026-07-17)

- main: etalon **28/28** (hybrid: šablony primární, UDPipe fallback), 339 testů, coverage UZLY 466/6265/968/299.
- Hotovo z minulého arcu: `Query` analýza bez UDPipe, `query_mode` gate, is_node/greedy dělení, ano/ne, typový filtr, date drill, entity-first, `_resolve_topic` evidence patra.
- Nevyřešeno (vstřebáno sem): REST API, web přes API, clarifikace (13b), odchod UDPipe, `run_pattern`/wire formát.
- Nález „souvislost": viz-popisek hran `kontext` (záměr) + šumové uzly z theme extrakce (oprava ve Fázi 0).

---

## FÁZE 0 — hygiena dat (malá, samostatně mergovatelná)

### Task 0.1: Stop-list funkčních substantiv v theme extrakci

Uzly `souvislost`, `přesvědčení` apod. vznikají z frází „v souvislosti s…" jako theme účastníci; větné slepence („nutné pojímat v souvislosti s dobovým stavem kritických") jako obj z přímé řeči.

**Files:** Modify: `jellyai/lang/cs.json` (nový klíč `"function_nouns"`), `jellyai/lang/__init__.py` (frozenset), `jellyai/graph/extract.py` (guard u theme/obl účastníků + délkový guard obj ≤ 5 slov); Test: `tests/test_graph_extract.py`.

- [x] Failing test: věta s obl „o souvislosti" nevytvoří účastníka `souvislost`; obj delší než 5 slov se zahodí (větný slepenec).
- [x] cs.json: `"function_nouns": ["souvislost", "přesvědčení", "ohled", "rámec", "smysl", "důsledek", "základ", "závislost"]` (doplňovat měřením); extract.py: theme/obl účastník s lemmatem ve `function_nouns` se přeskočí; obj span > 5 slov se přeskočí. *(Realita: stop-list implementován; samostatný délkový guard obj > 5 slov se neukázal potřebný — místo něj později guardy dle auditu úniků, commit `f0e15ad`.)*
- [x] Přegenerovat graf (`./jelly graph`), spustit `run_etalon.py` (28/28 drží!) + `run_coverage.py` (UZLY zapsat — izolovaných smí ubýt), ověřit `[n.id for n in g.nodes.values() if "souvislost" in n.id.lower()]` → jen legitimní.
- [x] Commit `66d7166` `fix(extract): stop-list funkčních substantiv — theme šum ('souvislost') pryč z grafu`.

### Task 0.2: Baseline metrika zaostření

**Files:** Create: `benchmark/run_focus.py`; Create: `benchmark/focus.jsonl`.

- [x] `focus.jsonl`: `{"q": …, "expect_nodes": [id…], "k": 5}` — po dotazu mají očekávané uzly být v top-K `context.scores`. Nasadit ~10 řádků z etalonu (téma+odpověď). *(Reálně 12 řádků.)*
- [x] `run_focus.py`: projede řádky, spočítá hit-rate top-K; vypíše `FOCUS: x/y (z %)`. Baseline 12/12 (100 %).
- [x] Commit `9d98412` `feat(benchmark): run_focus — normativní měření zaostření aktivace (baseline 12/12, top-K guardrail)`.

---

## FÁZE 1 — skelet knihovny Iris + pattern-karty + REST API

Cíl: existuje knihovna `jellyai/iris/` (automaton/state/patterns/assurance/presenter), pseudo-QL parser je její první plugin, REST služba vrací odpověď + metadata (použité komponenty/karty, assurance, aktivační okno). Web zatím beze změny.

### Task 1.1: Pattern-karty — loader a registr

**Files:** Create: `jellyai/iris/__init__.py`, `jellyai/iris/patterns.py`, `jellyai/iris/patterns/cs/focus-offer-homonym.json` (+ 2 další karty: `resolve-miss.json`, `assurance-fail.json`); Test: `tests/test_iris_patterns.py`.

**Interfaces:** `PatternDeck(lang_dir)` — `.load()`, `.match(event: str, context: dict) -> PatternCard|None` (první karta, jejíž trigger sedí; deterministické pořadí dle `priority` pak jména). `PatternCard` dataclass: `name, trigger: dict, dialog: str, action: dict, teach: str`. Trigger klíče: `event` (přesná shoda), `assurance_below` (float), `min_candidates` (int).

- [x] Failing testy: load adresáře; match `resolve.ambiguous` s 2 kandidáty a assurance 0.4 → `focus-offer-homonym`; match s assurance 0.9 → None; neznámý event → None.
- [x] Implementace + 3 karty (dialog texty česky V KARTÁCH, ne v kódu; `{candidates}` placeholder). *(Balíček od té doby narostl na 8 karet — přibyly focus-shift, memory-stored a statement-* karty Mnemos.)*
- [x] Commit `05f156a`.

### Task 1.2: QueryAssurance skóre

**Files:** Create: `jellyai/iris/assurance.py`; Modify: `jellyai/answerer/graph_answerer.py` (`_resolve_topic` navíc vrací/ukládá evidenci — `self.last_resolution = {"term", "evidence": score_tuple, "candidates": [(id, score)…]}`); Test: `tests/test_iris_assurance.py`.

**Interfaces:** `assurance(evidence, candidates, activation) -> float` — váhy pater (exact 1.0, ins .8, stem .6, da .4, loose .25) × pokrytí / počet termů, + aktivace vítěze (normovaná), − penalizace rovnocenných cizích clusterů. `GraphConfig.assurance_threshold: float = 0.5`.

- [x] Failing testy: plné exact pokrytí → ≥0.9; jediný loose hit se 2 rovnocennými kandidáty → <0.5; aktivace zvedá skóre nad práh (dialogová smyčka konverguje).
- [x] Implementace; `_resolve_topic` plní `last_resolution` (bez změny návratové hodnoty).
- [x] Commit `3a33865`.

### Task 1.3: Automat + FocusState (jádro)

**Files:** Create: `jellyai/iris/automaton.py`, `jellyai/iris/state.py`, `jellyai/iris/presenter.py`; Test: `tests/test_iris_automaton.py`.

**Interfaces:**
- `FocusState(activation_field)` — drží pole, historii tahů, `pending` (očekávaná volba uživatele z minulé karty).
- `IrisAutomaton(answerer, deck, threshold)` — `.turn(text: str) -> IrisResponse`. Uvnitř: (1) je-li `pending` a text vypadá jako volba → zaostři (warm) a přehraj původní dotaz; (2) jinak zavolej plugin (answerer), spočti assurance; (3) `focus.ok` → odpověď; jinak vyber kartu (`deck.match`) → dialogová odpověď + akce (warm kandidátů).
- `IrisResponse` dataclass: `text, kind ("answer"|"dialog"), assurance, activation_window: [(node, jas)…], used: {"components": […], "patterns": […]}, trace, sources`.
- `presenter.py`: seřazené aktivační okno + metadata (žádná forma odpovědi!).

- [x] Failing testy (syntetický graf, FakeUfalClient): jistý dotaz → `kind=answer` + okno seřazené sestupně; homonymní dotaz („Kdo je Čapek?" se 2 bratry) → `kind=dialog`, text z karty, kandidáti zahřáti; následná volba („Karel Čapek") → odpověď (smyčka konverguje); nezaostřitelné → karta `assurance-fail` s kandidáty.
- [x] Implementace; answerer se volá přes tenký adapter (plugin) `jellyai/iris/plugins/pseudo_ql.py` — obaluje `GraphAnswerer.answer` + `last_pattern`/`last_resolution`. *(Odchylka od plánu: adresář `plugins/` nevznikl — automat volá answerer přímo (`automaton.turn`); rozpad na pluginy zůstává na Fázi 5.)*
- [x] Commit `007b3a0`.

### Task 1.4: `run_pattern` + wire formát (převzato z minulého plánu, Task 11)

**Files:** Modify: `jellyai/answerer/graph_answerer.py` (`run_pattern`), `jellyai/answerer/pattern.py` (`pattern_to_json`, `pattern_from_json`); Test: `tests/test_pattern.py`, `tests/test_graph_answerer.py`. (Kód viz plán `2026-07-17-query-sablony.md` Task 11 — platí beze změny.)

- [x] Testy (roundtrip vnořeného SubQuery; run_pattern díra + existence) → implementace → commit `7ccec13`.

### Task 1.5: REST služba Iris

**Files:** Modify: `services/_common.py` (GET routy); Create: `services/iris_service.py`; Modify: `config.py` (`ServicesConfig.iris_port: int = 8084`); Test: `tests/test_iris_service.py`.

**Interfaces:**
- `POST /query {"question", "temperature"?}` → `{"answer", "kind", "assurance", "clarify"|null, "pattern", "trace", "sources", "alternatives", "activation": {"nodes": [(id, jas)…seřazené], "docs": [(doc, jas)…]}, "used": {"components", "patterns"}}` — metadata = které komponenty a karty se použily (požadavek §5 spec).
- `POST /graphql {pattern JSON}` → přímé vykonání (`run_pattern`) — testovatelnost jazyka.
- `GET /schema` → predikáty grafu, role, typy uzlů, tázací tabulka, **seznam pattern-karet** (jméno + teach).
- Start: `--port`, `--model` (= graph.pkl). In-process route testy + 1 subprocess integrační (tmp graf, bez ML modelů).

- [x] Testy → implementace (vzor `udpipe_service.py`; `sys.path` bootstrap na kořen) → curl akceptace na ostrém grafu (restart!) → commit `c9d33ba`. *(Navíc oproti plánu: `POST /reset` — nový rozhovor.)*

### Task 1.6: Fáze-gate měření

- [x] `pytest -q` zelené; `run_etalon.py` 28/28; `run_coverage.py` beze změny; `run_focus.py` ≥ baseline. Ruční: 3 curl příklady ze spec §5 + dialogová ukázka „Kdo je Čapek?" přes `POST /query` (dialog → volba → odpověď).
- [x] Commit + merge fáze do main (`b0ff114`).

---

## FÁZE 2 — dialogové zaostřování + Chronos (ČÁSTEČNĚ HOTOVÁ — viz stav nahoře)

> Hotové navíc oproti obrysu (přišlo ze spec §3d/§3e po sepsání plánu):
> **Mnemos** — časově vázaná paměť uživatele v grafu, karty statement-*
> (`da07043`) + perzistence deníku `data/memory.jsonl` s replayem při startu
> (`004fc5a`); **focus-shift** — „v kontextu Bible" posvítí na doménu
> a přehraje otázku (`8a244df`).

- [ ] Karty: `clarify-person`, `clarify-relation`, `clarify-period` („v jakém roce/století?"), `data-overflow` („Co řekl Ježíš?" → nabídka oblastí aktivace), `data-empty` + rekurzivní ostření (nejistota v SubQuery nese cestu). *(Otevřené — BACKLOG 5.)*
- [x] **Kartový audit (ZÁKON §2.4b)**: sweep VŠECH natvrdo udělaných logických stavů v kódu Iris — větvení `turn()` (kdy answer/dialog/terminál), prahy, potvrzovací texty, výběr akcí — a migrace rozhodnutí do karet; v kódu smí zbýt jen mechanismy (rysy, aritmetika, matching). Iris je stavový automat ŘÍZENÝ json kartami. *(Hotovo — `58f88b6`: prahy jen v kartách, kód bez rozhodnutí.)*
- [ ] **Výběr karty benefitem (spec §2.6b)**: `PatternDeck.best(event, context)` — kandidátky se skórují (specificita: kolik trigger podmínek sedí a jak těsně; priorita; případně simulovaný zisk zaostření) a vyhrává největší benefit; first-match zůstává fallback pro jednoduché balíčky. Testy: karta s těsnějším triggerem přebije obecnou i s nižší prioritou. *(Otevřené — BACKLOG 4.)*
- [ ] Datová hygiena (nález z /schema): šum v predikátech grafu („Chvalte", „Izaiáš" — velká písmena/jména jako predikáty) → guard v extrakci, měřit coverage. *(Otevřené — BACKLOG 2.)*
- [x] Dialogové scénáře ze spec §1.1 do etalonu jako `dialog` řádky (runner umí); expecty po změření. *(Hotovo jinak: samostatný benchmark `run_dialog.py` + `dialog.jsonl` — 5 scénářů / 9 tahů, fixní hodiny, 100 % (`bfa6975`); etalon má navíc 2 dialogové řádky.)*
- [ ] `_pattern_answer` guessing patra (kontext asociace) se ZAŘADÍ ZA clarifikaci (dialog > figly) — měřit, že poctivost řádky drží.
- [x] **F2.chronos — Iris orientován v čase (spec §3b)** *(Hotovo — `4452bd7` primitiva a intervaly, `57b45b2` časová kotva + hodinové odpovědi + injektovaný clock, `58f88b6` interval rozsvěcí časové uzly. Interval jako tvrdý filtr odpovědi zůstává otevřený — BACKLOG 10.)*:
  - `jellyai/iris/chronos.py`: `TimeInterval(start, end, granularity)` (půlotevřený, datetime přesnost) + `resolve_temporal(tokens, now) -> TimeInterval|None` — primitiva („dnes/vcera/zitra"), směrovky („pred/za" ± offset, „tento" = aktuální interval jednotky), jednotky (hodina/den/týden/měsíc/rok), číslovky („dvema"→2) — VŠE z `cs.json` klíče `temporal`; `now` vždy parametr (testy fixují `datetime(2026, 7, 17, 12, 0)`, API bere hodiny).
  - Napojení na graf: `interval.contains_date(parsed)` nad `parse_date` časových uzlů; sharpener krok „rozsviť časové uzly v intervalu + účastníky jejich faktů" (hrany); `_reverse_lookup` umí interval místo přesného data („Co se stalo tento měsíc?" na bázi s aktuálními daty).
  - Testy: čistá aritmetika s fixním now (dnes/včera/za hodinu/před dvěma hodinami/před týdnem/tento týden/tento měsíc — hranice intervalů!); syntetický graf s časovými uzly → interval rozsvítí správné uzly a fakty; determinismus (dvojí běh = týž výsledek).
  - POZOR: aktuální korpus (1818/1890/bible) relativní dotazy netrefí — akceptace na syntetickém grafu s dnešními daty; etalon beze změny.
- Akceptace: oba scénáře §1.1 projdou přes `/query` sekvenci; etalon 100 %; chronos testy zelené. *(Dialogové chování normativně drží `run_dialog.py`.)*

## FÁZE 3 — web tři okna (HOTOVÁ)

- [x] `cmd_web` přes Iris REST (klient `jellyai/iris/client.py`, vzor `_ServiceHandle`); okno dialogu (stávající konzole, jen dialog), NOVÉ aktivační okno uzlů (seřazený seznam z `activation.nodes`), Aktivní dokumenty (existuje). `tests/test_cli_web.py` přepis na fake klienta. *(Hotovo — `0799091`; panely nezavíratelné, bez vstupního řádku — `b10420e`.)*
- [x] Akceptace: `./jelly web` — tři okna, dialogová ukázka §1.1 v prohlížeči.

## FÁZE 4 — sharpener (obrys)

- `jellyai/iris/sharpener.py`: aktivační funkce jako kompozice — trasová (dnešní `_spread`), cross-distribuce (kontext bez tras), vyzařování focusu po hranách; váhy v configu.
- `run_focus.py` rozšířit (více scénářů, K-křivka); ladit váhy měřením; etalon nesmí klesnout.

## FÁZE 5 — čistý řez (obrys; absorbuje Task 14 minulého plánu)

- Gate: etalon 100 % šablonami (`--mode templates` — dnes splněno). Odstranit `question_pattern`/`analyze_question` z query cesty + `query_mode` knob; UDPipe parsing z `pattern.py` pryč.
- Monolit `graph_answerer.py` rozdělit: match/solve jádro vs. pluginy (existence, drill, reverse-date, event) — do `jellyai/iris/plugins/`.
- Pohrobci → `conserved_` (inventura: co nejde směrem automat+graf; kandidáti se URČÍ měřením použití, ne odhadem).

## FÁZE 6 — experimenty (odloženo)

- Hybridní aktivace uzel×hrana (hrana vyzařuje s menším ziskem na navázané hrany). Nejdřív návrh metriky, pak prototyp za flagem, měřit `run_focus.py`.

---

## Poznámky pro exekutora

- Fáze = samostatně mergovatelné celky; detailní tasky Fází 2–6 se rozepíší po uzavření předchozí fáze (týmž writing-plans postupem, s aktuálními čísly).
- Při nejasnosti sáhni do spec (`2026-07-17-ql-automat.md`) — body 1–20 jsou normativní přepis brainstormingu uživatele.
- Terminologie („Iris") je návrh — pokud uživatel vetuje, přejmenování je jeden sed (adresář + třídy), neodkládat kvůli tomu start.
