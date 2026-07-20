# Otázkový graf E1+E2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realizovat experimenty E1 (rodinné karty s dimenzemi) a E2
(claim() workerů) ze spec `docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md`,
každý s měřením a kritériem přijetí parita + ≥1 nový řádek zelený.

**Architecture:** E1 přidá do `PatternDeck.load()` čistě mechanické
rozvinutí RODINNÉ karty (kostra × enumerativní dimenze → konkrétní
karty se jmény dnešních klonů + chybějící kombinace). E2 přesune ruční
výčet přímých expertů z `compile_qgraph`/`illuminate` do formálního
registru claimů (`jellyai/iris/claims.py`) — zárodek #26 na úrovni
grafu. Matcher se NEMĚNÍ, žádný interpret v JSONu.

**Tech Stack:** Python 3 (`.venv/bin/python`), pytest, JSON karty
(`jellyai/iris/patterns/cs/`), benchmarky `benchmark/run_*.py`,
shadow harness `benchmark/run_qgraph.py`.

## Global Constraints

- Větev `otazkovy-graf`; commity heredocem `git commit -F - <<'MSG'`
  (české uvozovky v inline `-m` rozbíjejí bash — HANDOVER past 7).
- V JSON řetězcích (teach/gap/dialog) NIKDY ASCII uvozovky `"` —
  vždy české „…“ (pasti 3+13; rozbitá karta = desítky testů).
- Před KAŽDÝM commitem celá sada, nikdy `pytest | tail` v řetězu
  (maska exit kódu — past 4):
  `.venv/bin/python -m pytest -q` → N passed, 0 failed;
  `.venv/bin/python benchmark/run_etalon.py` → JÁDRO 100 %;
  `.venv/bin/python benchmark/run_focus.py` → 100 %;
  `.venv/bin/python benchmark/run_dialog.py` → 100 % (gap řádky zvlášť);
  `.venv/bin/python benchmark/run_mnemos.py` → ZÁPIS 100 % (bez --nom
  — nechceme závislost na běžících ÚFAL službách);
  `.venv/bin/python benchmark/run_qgraph.py` → všechny roviny 100 %
  (a jednou i `--variant weights`).
- Kritérium spec §3: žádné číslo neklesne; klesne-li, je to NÁLEZ —
  vyšetřit (pasti 9–11: krádež otázky jiného smyslu), neobcházet.
- Neměnit `jellyai/lang/matcher.py`; dimenze jsou enumerativní osy
  (dosazení do slotu), žádné podmínky/výrazy (zákaz ATN, spec §3).
- Nové testy psát NAPŘED (červené), pak implementace.

---

### Task 1: E1 — mechanismus rozvinutí rodinné karty

**Files:**
- Modify: `jellyai/iris/patterns.py` (funkce `_expand_family` + hook v `load()`)
- Test: `tests/test_iris_patterns.py` (append)

**Interfaces:**
- Produces: `_expand_family(data: dict) -> list[dict]` — modulová funkce
  v `jellyai/iris/patterns.py`; vstup = JSON dict rodinné karty
  (trigger má `pattern` se sloty `<jmeno>` a `dimensions`), výstup =
  seznam dictů konkrétních karet (bez `dimensions`). `PatternDeck.load()`
  rozvinutí volá automaticky, konzumenti decku (query.py, qgraph.py)
  vidí jen konkrétní karty.
- Formát rodiny: `trigger.dimensions` = seznam os
  `{"slot": "<jmeno>", "values": [{"suffix": str, "element": str|null,
  "priority"?: int}, …]}`. Jméno varianty = jméno rodiny + přípony
  v pořadí os. `element: null` = prvek zmizí (jen POSLEDNÍ prvek vzoru
  — jinak by se posouvaly odkazy `$N`, disciplína pasti 14); reference
  na zmizelý slot v `known` se vypustí, v `hole`/`predicate` je chybou.

- [ ] **Step 1: Zaznamenej baseline**

Run: celá sada z Global Constraints (pytest + 5 benchmarků + harness
`tiers` i `weights`). Zapiš si aktuální čísla (počet testů, dialog
scénáře, harness roviny, počet uzlů z `./jelly qgraph | head -3`).
Expected: vše 100 % (stav větve po c731a04). Pokud ne, STOP — základ
není zelený, nejdřív vyšetřit.

- [ ] **Step 2: Napiš červené testy rozvinutí**

Do `tests/test_iris_patterns.py` přidej (import `pytest` nahoru, pokud
tam není):

```python
def _familia():
    """Zkušební rodinná karta (kostra × dimenze čas a elipse)."""
    return {
        "name": "q-pokus",
        "trigger": {
            "event": "utterance.query",
            "pattern": ["%{TAZACI}", "<sloveso>", "<ucastnik>"],
            "dimensions": [
                {"slot": "<sloveso>", "values": [
                    {"suffix": "-minuly", "element": "%{SLOVESO_MINULE}"},
                    {"suffix": "-prezens", "element": "%{SLOVESO}"}]},
                {"slot": "<ucastnik>", "values": [
                    {"suffix": "", "element": "%{ENTITA}", "priority": 4},
                    {"suffix": "-prodrop", "element": None, "priority": 3}]},
            ],
        },
        "action": {"query": {"hole": "$1", "predicate": "$2",
                             "known": ["$3"]}},
        "teach": "Zkušební rodina.",
    }


def test_expand_family_rozvine_kostru_krat_dimenze():
    from jellyai.iris.patterns import _expand_family
    cards = _expand_family(_familia())
    by_name = {card["name"]: card for card in cards}
    assert set(by_name) == {"q-pokus-minuly", "q-pokus-prezens",
                            "q-pokus-minuly-prodrop",
                            "q-pokus-prezens-prodrop"}
    plna = by_name["q-pokus-minuly"]
    assert plna["trigger"]["pattern"] == [
        "%{TAZACI}", "%{SLOVESO_MINULE}", "%{ENTITA}"]
    assert plna["trigger"]["priority"] == 4
    assert plna["action"]["query"]["known"] == ["$3"]
    assert "dimensions" not in plna["trigger"]
    prodrop = by_name["q-pokus-prezens-prodrop"]
    assert prodrop["trigger"]["pattern"] == ["%{TAZACI}", "%{SLOVESO}"]
    assert prodrop["trigger"]["priority"] == 3
    assert "known" not in prodrop["action"]["query"]


def test_expand_family_prazdny_slot_jen_na_konci():
    from jellyai.iris.patterns import _expand_family
    data = _familia()
    data["trigger"]["pattern"] = ["<ucastnik>", "%{TAZACI}", "<sloveso>"]
    with pytest.raises(ValueError):
        _expand_family(data)


def test_expand_family_dira_nesmi_mirit_na_prazdny_slot():
    from jellyai.iris.patterns import _expand_family
    data = _familia()
    data["action"]["query"]["predicate"] = "$3"
    with pytest.raises(ValueError):
        _expand_family(data)
```

- [ ] **Step 3: Ověř, že testy padají**

Run: `.venv/bin/python -m pytest tests/test_iris_patterns.py -q`
Expected: FAIL — `ImportError: cannot import name '_expand_family'`.

- [ ] **Step 4: Implementuj rozvinutí v patterns.py**

Do `jellyai/iris/patterns.py` přidej pod importy (`import copy` a
`import itertools` k horním importům):

```python
def _expand_family(data):
    """Rozvine RODINNOU kartu (kostra × dimenze) na konkrétní karty (#57 E1).

    Dimenze jsou ENUMERATIVNÍ osy, žádné podmínky (zákaz ATN): hodnota
    osy jen dosadí prvek do slotu kostry, přidá příponu jména a smí
    přepsat prioritu (definuje-li ji víc os, platí poslední). Prvek
    `null` = slot zmizí — smí být jen POSLEDNÍM prvkem vzoru, aby se
    neposouvaly odkazy $N (disciplína pasti 14); odkaz na zmizelý slot
    se z `known` vypustí a v `hole`/`predicate` spadne nahlas.
    """
    trigger = data["trigger"]
    skeleton = trigger["pattern"]
    dimensions = trigger["dimensions"]
    cards = []
    for combo in itertools.product(*(d["values"] for d in dimensions)):
        chosen = {d["slot"]: v for d, v in zip(dimensions, combo)}
        pattern, dropped = [], None
        for index, element in enumerate(skeleton, start=1):
            value = chosen.get(element)
            if value is None:                    # obyčejný prvek kostry
                pattern.append(element)
            elif value["element"] is None:
                if index != len(skeleton):
                    raise ValueError(
                        f"rodina {data['name']}: prázdný slot {element}"
                        f" musí být posledním prvkem vzoru")
                dropped = f"${index}"
            else:
                pattern.append(value["element"])
        priority = trigger.get("priority", 0)
        for value in combo:
            if "priority" in value:
                priority = value["priority"]
        action = copy.deepcopy(data.get("action", {}))
        query = action.get("query", {})
        for key in ("hole", "predicate"):
            if dropped is not None and query.get(key) == dropped:
                raise ValueError(f"rodina {data['name']}: {key}"
                                 f" míří na prázdný slot")
        if dropped is not None and "known" in query:
            kept = [k for k in query["known"]
                    if (k[-1] if isinstance(k, list) else k) != dropped]
            if kept:
                query["known"] = kept
            else:
                del query["known"]
        new_trigger = {k: v for k, v in trigger.items()
                       if k not in ("dimensions", "pattern")}
        new_trigger["pattern"] = pattern
        new_trigger["priority"] = priority
        cards.append({"name": data["name"]
                      + "".join(v["suffix"] for v in combo),
                      "trigger": new_trigger, "action": action,
                      "teach": data.get("teach", "")})
    return cards
```

V `PatternDeck.load()` nahraď blok vytvářející `PatternCard` tímto
(rozvinutí rodin je pro konzumenty neviditelné):

```python
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            variants = (_expand_family(data)
                        if "dimensions" in data.get("trigger", {})
                        else [data])
            for item in variants:
                cards.append(PatternCard(
                    name=item.get("name", name[:-5]),
                    trigger=item.get("trigger", {}),
                    dialog=item.get("dialog", ""),
                    action=item.get("action", {}),
                    teach=item.get("teach", "")))
```

- [ ] **Step 5: Ověř zelené testy + celou sadu**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (žádná rodinná karta zatím neexistuje — chování decku
beze změny).

- [ ] **Step 6: Commit**

```bash
git add jellyai/iris/patterns.py tests/test_iris_patterns.py
git commit -F - <<'MSG'
feat(qgraph): #57 E1 — mechanismus rozvinutí rodinné karty (kostra × dimenze)

Enumerativní osy bez podmínek (zákaz ATN); prázdný slot jen na konci
vzoru (odkazy $N se neposouvají — disciplína pasti 14). Deck rozvine
rodinu při načtení, konzumenti vidí jen konkrétní karty.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 2: E1 — červený gap scénář „Kde bydlí?" (prézens-prodrop)

**Files:**
- Modify: `benchmark/dialog.jsonl` (append 1 řádek)

**Interfaces:**
- Consumes: formát scénáře `{"name", "gap"?, "turns": [{"u", "expect",
  "reject"?}]}` (viz scénář `kdo-bydli-v-miste` tamtéž).
- Produces: gap scénář `kde-bydli-prodrop-prezens`, který Task 3
  přepne na GAP-FIXED.

- [ ] **Step 1: Přidej gap scénář (červený NAPŘED)**

Na konec `benchmark/dialog.jsonl` přidej JEDEN řádek (pozor: uvnitř
JSON řetězců jen české uvozovky „…“):

```json
{"name": "kde-bydli-prodrop-prezens", "gap": "#57 E1: kombinace prézens×prodrop v ručně psaných kartách chybí — navazující „Kde bydlí?“ po zápisu nemá kartu; rozvinutí rodiny q-otaz ji vygeneruje", "turns": [{"u": "Marcela bydlí v Petrovicích.", "expect": ["Zapamatováno"]}, {"u": "Kde bydlí?", "expect": ["Petrovic"]}]}
```

- [ ] **Step 2: Ověř, že scénář je červený (gap open)**

Run: `.venv/bin/python benchmark/run_dialog.py`
Expected: jádrové scénáře 100 %, nový scénář hlášen jako OTEVŘENÝ gap
(ne GAP-FIXED). POKUD projde už teď (chytá ho poziční šablona):
STOP — je to nález; zapiš ho do spec §4 (Empirie E1) a jako měřený
zisk E1 zůstane parita + úbytek karet; scénář pak přidej BEZ pole
`gap` jako regresní.

- [ ] **Step 3: Commit**

```bash
git add benchmark/dialog.jsonl
git commit -F - <<'MSG'
test(qgraph): #57 E1 — gap scénář kde-bydli-prodrop-prezens (červený napřed)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 3: E1 — rodina q-otaz (3 soubory → 1 + nová kombinace)

**Files:**
- Create: `jellyai/iris/patterns/cs/q-otaz.json`
- Delete: `jellyai/iris/patterns/cs/q-otaz-minuly.json`,
  `jellyai/iris/patterns/cs/q-otaz-prezens.json`,
  `jellyai/iris/patterns/cs/q-otaz-minuly-prodrop.json`
- Test: `tests/test_iris_patterns.py` (append)

**Interfaces:**
- Consumes: `_expand_family` z Task 1 (přes `PatternDeck.load()`).
- Produces: karty `q-otaz-minuly`, `q-otaz-prezens`,
  `q-otaz-minuly-prodrop` (bitově shodné s dnešními) +
  **`q-otaz-prezens-prodrop`** (nová kombinace — zavírá gap z Task 2).
  Jména beze změny → testy, telemetrie i harness referencují dál.

- [ ] **Step 1: Napiš červený deck-level test**

Do `tests/test_iris_patterns.py`:

```python
def test_deck_rozvine_rodinu_q_otaz():
    deck = PatternDeck.for_language("cs")
    deck.load()
    cards = {card.name: card for card in deck.cards}
    assert cards["q-otaz-minuly"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO_MINULE}", "%{ENTITA}"]
    assert cards["q-otaz-minuly"].trigger["priority"] == 4
    assert cards["q-otaz-minuly-prodrop"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO_MINULE}"]
    assert cards["q-otaz-minuly-prodrop"].trigger["priority"] == 3
    assert cards["q-otaz-minuly-prodrop"].action["query"] == {
        "hole": "$1", "predicate": "$3"}
    # NOVÁ kombinace z rozvinutí — ručně nikdy nenapsaná:
    assert cards["q-otaz-prezens-prodrop"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO}"]
    assert cards["q-otaz-prezens-prodrop"].trigger["priority"] == 3
```

(Pokud soubor `PatternDeck` neimportuje, přidej
`from jellyai.iris.patterns import PatternDeck` k importům.)

Run: `.venv/bin/python -m pytest tests/test_iris_patterns.py -q`
Expected: FAIL na `q-otaz-prezens-prodrop` (KeyError — neexistuje).

- [ ] **Step 2: Vytvoř rodinnou kartu a smaž klony**

`jellyai/iris/patterns/cs/q-otaz.json` (celý obsah; české uvozovky!):

```json
{
  "name": "q-otaz",
  "trigger": {
    "event": "utterance.query",
    "pattern": ["%{TAZACI}", "?%{SE}", "<sloveso>", "<ucastnik>"],
    "dimensions": [
      {"slot": "<sloveso>", "values": [
        {"suffix": "-minuly", "element": "%{SLOVESO_MINULE}"},
        {"suffix": "-prezens", "element": "%{SLOVESO}"}]},
      {"slot": "<ucastnik>", "values": [
        {"suffix": "", "element": "%{ENTITA}", "priority": 4},
        {"suffix": "-prodrop", "element": null, "priority": 3}]}
    ]
  },
  "action": {
    "query": {"hole": "$1", "predicate": "$3", "known": ["$4"]}
  },
  "teach": "RODINNÁ karta tázacích otázek s dírou (#57 E1): kostra tázací slovo + ?se + sloveso + účastník, dimenze ČAS (minulý/prézens) × ELIPSE (plná/prodrop) — rozvinutí při načtení decku vyrobí q-otaz-minuly („Kdo napsal R.U.R.?“), q-otaz-prezens („Kde bydlí Marcela?“), q-otaz-minuly-prodrop („Kdy se narodil?“) a q-otaz-prezens-prodrop („Kde bydlí?“). Díru určuje tázací slovo (interrogatives), predikát je sloveso ve slotu, entita span potvrzený grafem. Prodrop varianty (nižší priorita, ukotvení na konec věty) nechají podmět doplnit z konverzačního těžiště: l-tvar nese rodovou shodu, prézens rod nenese a bere nejžhavější téma."
}
```

```bash
git rm jellyai/iris/patterns/cs/q-otaz-minuly.json \
       jellyai/iris/patterns/cs/q-otaz-prezens.json \
       jellyai/iris/patterns/cs/q-otaz-minuly-prodrop.json
```

- [ ] **Step 3: Ověř zelený test + CELOU sadu (parita + gap)**

Run: `.venv/bin/python -m pytest -q`, pak všech 5 benchmarků + harness
(obě varianty) dle Global Constraints.
Expected: testy PASS; etalon/focus/mnemos beze změny; **dialog hlásí
GAP-FIXED kde-bydli-prodrop-prezens**; run_qgraph všechny roviny 100 %
(uzlů o 1 víc — nová varianta). Pokles kdekoli = krádež otázky novou
kartou (pasti 9–11) — vyšetři NESHODA výpis harnessu, neobcházej.
Pokud gap NEzezelená (karta vznikla, ale scénář dál padá): ověř, že
lexer dává „bydlí“ třídu sloveso_fin (`.venv/bin/python -c "from
jellyai.lang.lexer import classify; print(classify('Kde bydlí?',
is_node=None))"`), a nález zapiš do Empirie E1 — kritérium pak stojí
na jiné chybějící kombinaci, nebo se E1 nepřijme (rámec §3).

- [ ] **Step 4: Commit**

```bash
git add -A jellyai/iris/patterns/cs/ tests/test_iris_patterns.py
git commit -F - <<'MSG'
feat(qgraph): #57 E1 — rodina q-otaz (3 karty → 1, + q-otaz-prezens-prodrop)

Rozvinutí kostra × čas × elipse reprodukuje klony bitově (jména drží)
a generuje chybějící kombinaci — gap kde-bydli-prodrop-prezens FIXED.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 4: E1 — rodina q-zjistovaci (2 soubory → 1)

**Files:**
- Create: `jellyai/iris/patterns/cs/q-zjistovaci.json`
- Delete: `jellyai/iris/patterns/cs/q-zjistovaci-minuly.json`,
  `jellyai/iris/patterns/cs/q-zjistovaci-prezens.json`
- Test: `tests/test_iris_patterns.py` (append)

**Interfaces:**
- Consumes: `_expand_family` z Task 1.
- Produces: karty `q-zjistovaci-minuly` a `q-zjistovaci-prezens`
  bitově shodné s dnešními (osa čas nemá novou kombinaci — zisk je
  čistě DRY zdrojů).

- [ ] **Step 1: Napiš červený test**

```python
def test_deck_rozvine_rodinu_q_zjistovaci():
    deck = PatternDeck.for_language("cs")
    deck.load()
    cards = {card.name: card for card in deck.cards}
    for jmeno, sloveso in (("q-zjistovaci-minuly", "%{SLOVESO_MINULE}"),
                           ("q-zjistovaci-prezens", "%{SLOVESO}")):
        assert cards[jmeno].trigger["pattern"] == [
            sloveso, "?%{ENTITA}", "?%{ENTITA}"]
        assert cards[jmeno].trigger["priority"] == 3
        assert cards[jmeno].action["query"]["known"] == [
            ["subj", "$2"], ["obj", "$3"]]
    assert not os.path.exists(os.path.join(
        PatternDeck.for_language("cs").directory,
        "q-zjistovaci-minuly.json"))
```

(Přidej `import os` k importům testu, pokud chybí.)

Run: `.venv/bin/python -m pytest tests/test_iris_patterns.py -q`
Expected: FAIL na `os.path.exists` (soubor klonu ještě existuje).

- [ ] **Step 2: Vytvoř rodinu a smaž klony**

`jellyai/iris/patterns/cs/q-zjistovaci.json`:

```json
{
  "name": "q-zjistovaci",
  "trigger": {
    "event": "utterance.query",
    "priority": 3,
    "pattern": ["<sloveso>", "?%{ENTITA}", "?%{ENTITA}"],
    "dimensions": [
      {"slot": "<sloveso>", "values": [
        {"suffix": "-minuly", "element": "%{SLOVESO_MINULE}"},
        {"suffix": "-prezens", "element": "%{SLOVESO}"}]}
    ]
  },
  "action": {
    "query": {"predicate": "$1",
              "known": [["subj", "$2"], ["obj", "$3"]]}
  },
  "teach": "RODINNÁ karta zjišťovacích otázek (#57 E1): predikát na začátku věty (dimenze ČAS: l-ové příčestí „Napsal Karel Čapek Válku s mloky?“ / prézens „Prší?“, „Bydlí Marcel v Petrovicích?“), volitelní účastníci jsou spany potvrzené grafem s rolemi subj/obj, díra žádná → answerer zkoumá EXISTENCI faktu (Ano / poctivé nenašel / „Ne, od T ne-…“ při negované evidenci). Bez účastníků holá existence predikátu. Nerozřešený span = karta nesedí a otázka jde poziční šablonou (bezpečný fallback)."
}
```

```bash
git rm jellyai/iris/patterns/cs/q-zjistovaci-minuly.json \
       jellyai/iris/patterns/cs/q-zjistovaci-prezens.json
```

- [ ] **Step 3: Ověř zelené testy + celou sadu**

Run: `.venv/bin/python -m pytest -q` + 5 benchmarků + harness.
Expected: vše 100 %, čísla beze změny (čistá parita — žádná nová
kombinace).

- [ ] **Step 4: Commit**

```bash
git add -A jellyai/iris/patterns/cs/ tests/test_iris_patterns.py
git commit -F - <<'MSG'
feat(qgraph): #57 E1 — rodina q-zjistovaci (2 karty → 1, čistá parita)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 5: E1 — uzávěrka: Empirie do spec + BACKLOG

**Files:**
- Modify: `docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md`
  (nová sekce za §4 E1)
- Modify: `docs/BACKLOG.md` (řádek #57)

- [ ] **Step 1: Zapiš Empirii E1 do spec**

Za blok E1 v §4 přidej pododdíl `**Empirie E1 (vyplněno po měření):**`
s NAMĚŘENÝMI hodnotami z Tasků 2–4: počet zdrojových karet před/po
(`ls jellyai/iris/patterns/cs/q-*.json | wc -l` před = 19), stav gap
scénáře (GAP-FIXED / nález o poziční šabloně z Task 2), čísla
benchmarků a harnessu, počet uzlů grafu. Výslovně zapiš dva korekční
nálezy z přípravy plánu: (a) `q-vyberova-prodrop` NENÍ klon
q-vyberova-minuly — má jinou sémantiku (implicitní spona „být“
literálem), do rodiny nepatří; (b) svinutí dekoračních karet
(q-rekl-adresatovi, q-kdo-sloveso-misto, q-prvni-osoba-minuly) se
odkládá — jejich `action` se liší strukturně (role theme,
user_subject) a osa s přepisem akce by byla program v datech (riziko
ATN); patří do E3/E4 úvahy, ne do E1.

- [ ] **Step 2: Aktualizuj BACKLOG řádek #57**

Do buňky #57 doplň větu: „E1 HOTOVO (rodiny q-otaz + q-zjistovaci,
karet 19→16, nová kombinace q-otaz-prezens-prodrop = GAP-FIXED
kde-bydli-prodrop-prezens; vyberova-prodrop není klon — jiná
sémantika)“ — čísla uprav podle skutečného měření.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md docs/BACKLOG.md
git commit -F - <<'MSG'
docs(qgraph): #57 E1 — empirie (rodiny karet, parita + GAP-FIXED)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 6: E2 — registr claimů (`jellyai/iris/claims.py`)

**Files:**
- Create: `jellyai/iris/claims.py`
- Test: `tests/test_qgraph.py` (append)

**Interfaces:**
- Produces: `ExpertClaim(name: str, worker: str, priority: int,
  recognize: callable)` — frozen dataclass; `recognize(text, now) ->
  bool`. `default_claims() -> tuple[ExpertClaim, ...]` — nároky tří
  dnešních přímých expertů v pořadí přednosti (metron 2 > chronos 1 >
  meta-focus 0; zrcadlí dnešní tier klíče v `illuminate`).
- Task 7 předá claims do `compile_qgraph`/`illuminate`.

- [ ] **Step 1: Zjisti přesnou meta-frázi**

Run: `.venv/bin/python -c "import json; print(json.load(open('jellyai/lang/cs.json'))['focus_query_phrases'])"`
Expected: seznam frází (např. deakcentované „o kom mluvime“ apod.).
V testu Step 2 použij PRVNÍ frázi z výstupu jako text meta-otázky
(s otazníkem a přirozenou kapitalizací).

- [ ] **Step 2: Napiš červený test**

Do `tests/test_qgraph.py` (NOW už v souboru je):

```python
def test_default_claims_rozpoznavaji_prime_brany():
    """E2 (#26): nároky přímých expertů jako DATA registru — kompilace
    a osvětlení je čtou jednotně, ruční výčet v kódu grafu mizí."""
    from jellyai.iris.claims import default_claims
    claims = {claim.name: claim for claim in default_claims()}
    assert set(claims) == {"metron-vypocet", "chronos-hodiny",
                           "meta-focus"}
    assert claims["metron-vypocet"].recognize("Kolik je 1 plus 1?", NOW)
    assert not claims["metron-vypocet"].recognize(
        "Kolik měla dětí Božena Němcová?", NOW)
    assert claims["chronos-hodiny"].recognize("Kolik je hodin?", NOW)
    assert claims["meta-focus"].recognize("O kom mluvíme?", NOW)
    assert not claims["meta-focus"].recognize("Kdo napsal R.U.R.?", NOW)
    assert (claims["metron-vypocet"].priority
            > claims["chronos-hodiny"].priority
            > claims["meta-focus"].priority)
```

(Text „O kom mluvíme?“ nahraď frází ze Step 1, pokud se liší.)

Run: `.venv/bin/python -m pytest tests/test_qgraph.py -q`
Expected: FAIL — `ModuleNotFoundError: jellyai.iris.claims`.

- [ ] **Step 3: Implementuj claims.py**

Celý obsah `jellyai/iris/claims.py`:

```python
"""Formální claim() přímých expertů (#57 E2 — zárodek #26 na úrovni grafu).

Expert deklaruje NÁROK na celý tah (přímá brána: výraz Metronu,
hodinová otázka Chronosu, meta-otázka na těžiště) jako záznam registru:
jméno worker uzlu + rozpoznávač. Kompilace otázkového grafu z registru
staví worker uzly a osvětlení volá rozpoznávače jednotně — ruční výčet
expertů v kódu grafu mizí; nový expert = nový claim, žádný zásah do
compile/illuminate/turn.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertClaim:
    """Nárok přímého experta na tah.

    Atributy:
        name (str): Jméno worker uzlu grafu (např. „metron-vypocet“).
        worker (str): Subsystém, který tah odbaví.
        priority (int): Přednost mezi přímými experty (vyšší dřív) —
            zrcadlí dnešní pořadí bran v turn().
        recognize: callable(text, now) -> bool — hlásí se expert o tah?
    """
    name: str
    worker: str
    priority: int
    recognize: object


def _metron(text, now):
    from jellyai.iris.subsystems.metron import compute
    return compute(text) is not None


def _chronos(text, now):
    from jellyai.iris.subsystems.chronos import clock_answer
    return clock_answer(text, now) is not None


def _meta_focus(text, now):
    from jellyai.graph.canon import deaccent
    from jellyai.lang import current
    low = deaccent(text.lower())
    return any(p in low for p in current().get("focus_query_phrases", ()))


def default_claims():
    """Nároky dnešních tří přímých expertů (pořadí = přednost bran)."""
    return (
        ExpertClaim("metron-vypocet", "metron", 2, _metron),
        ExpertClaim("chronos-hodiny", "chronos", 1, _chronos),
        ExpertClaim("meta-focus", "iris", 0, _meta_focus),
    )
```

- [ ] **Step 4: Ověř zelený test**

Run: `.venv/bin/python -m pytest tests/test_qgraph.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/iris/claims.py tests/test_qgraph.py
git commit -F - <<'MSG'
feat(qgraph): #57 E2 — registr claimů přímých expertů (zárodek #26)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 7: E2 — kompilace a osvětlení z claimů

**Files:**
- Modify: `jellyai/iris/qgraph.py` (`QGraph`, `compile_qgraph`,
  `illuminate`)

**Interfaces:**
- Consumes: `ExpertClaim`, `default_claims()` z Task 6.
- Produces: `QGraph.claims: tuple`; `compile_qgraph(deck, predicates,
  telemetry_rows, claims=None)` — None = `default_claims()`;
  `illuminate` čte `qgraph.claims`, žádné přímé importy expertů.

- [ ] **Step 1: Uprav QGraph a compile_qgraph**

V `jellyai/iris/qgraph.py` rozšiř dataclass:

```python
@dataclass
class QGraph:
    nodes: dict
    predicates: frozenset          # schéma datového grafu (líné instance)
    claims: tuple = ()             # nároky přímých expertů (#57 E2)
```

V `compile_qgraph` přidej parametr `claims=None` a nahraď blok
`# PŘÍMÍ EXPERTI — brány, které dnes drží ruční pořadí větví v turn()`
(tři natvrdo psané `nodes[...] = QNode(...)`) tímto:

```python
    # PŘÍMÍ EXPERTI — worker uzly z registru claimů (#57 E2, zárodek #26)
    if claims is None:
        from jellyai.iris.claims import default_claims
        claims = default_claims()
    for claim in claims:
        nodes[claim.name] = QNode(claim.name, "worker", claim.worker)
```

a návratovou hodnotu změň na
`return QGraph(nodes=nodes, predicates=frozenset(predicates), claims=tuple(claims))`.

- [ ] **Step 2: Uprav illuminate**

V `illuminate` smaž importy `clock_answer`/`compute` a tři natvrdo
psané bloky tier 3 (`if compute(text)…`, `if clock_answer…`,
`low = deaccent…` včetně proměnné `low`) a nahraď je:

```python
    moment = now or datetime.now()
    for claim in qgraph.claims:
        if claim.recognize(text, moment):
            lit.append(((3, claim.priority, 0), qgraph.nodes[claim.name]))
```

(`from jellyai.graph.canon import deaccent` v hlavičce modulu smaž,
pokud ho už nic jiného v souboru nepoužívá; `lang = current()` zůstává
kvůli `pattern_aliases`. Ověř úklid:
`grep -n "clock_answer\|compute(\|deaccent" jellyai/iris/qgraph.py`
— žádný z nich nesmí v illuminate zbýt.)

- [ ] **Step 3: Ověř celou sadu (parita)**

Run: `.venv/bin/python -m pytest -q` + 5 benchmarků +
`benchmark/run_qgraph.py` (tiers i weights).
Expected: vše 100 %, čísla beze změny — dispatch z claimů je bitově
týž (stejná jména uzlů, stejné tier klíče 3/2/1/0).

- [ ] **Step 4: Commit**

```bash
git add jellyai/iris/qgraph.py
git commit -F - <<'MSG'
refactor(qgraph): #57 E2 — kompilace a osvětlení čtou registr claimů

Ruční výčet přímých expertů v compile_qgraph/illuminate nahrazen
default_claims(); tier klíče beze změny — parita všech rovin drží.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task 8: E2 — test rozšiřitelnosti + uzávěrka

**Files:**
- Test: `tests/test_qgraph.py` (append)
- Modify: `docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md`,
  `docs/BACKLOG.md`

**Interfaces:**
- Consumes: `compile_qgraph(…, claims=…)`, `illuminate` z Task 7.

- [ ] **Step 1: Napiš test rozšiřitelnosti (kritérium E2)**

```python
def test_novy_expert_claimem_bez_zasahu_do_kodu_grafu():
    """Kritérium E2: nový přímý expert = nový claim v registru —
    kompilace mu postaví worker uzel a osvětlení ho zvedne; do
    compile_qgraph/illuminate/turn se NEsahá."""
    from jellyai.iris.claims import ExpertClaim, default_claims
    deck = PatternDeck.for_language("cs")
    deck.load()
    fake = ExpertClaim("pokus-expert", "pokus", 9,
                       lambda text, now: "pokusný nárok" in text)
    qg = compile_qgraph(deck, claims=default_claims() + (fake,))
    assert qg.nodes["pokus-expert"].worker == "pokus"
    lit = illuminate("Tohle je pokusný nárok na tah.", qg, now=NOW)
    assert lit and lit[0].name == "pokus-expert"
```

Run: `.venv/bin/python -m pytest tests/test_qgraph.py -q`
Expected: PASS rovnou (mechanismus z Task 7 to už umí — test je
DOKLAD kritéria, ne driver; kdyby padal, Task 7 je děravý — vrať se).

- [ ] **Step 2: Empirie E2 do spec + BACKLOG**

Za blok E2 v §4 spec přidej `**Empirie E2 (vyplněno po měření):**` —
parita všech rovin (čísla z Task 7 Step 3), test rozšiřitelnosti
zelený, počet řádků smazaného ručního výčtu. Do BACKLOG #57 doplň:
„E2 HOTOVO (registr claimů, parita, rozšiřitelnost testem — brána
přepnutí dispatch otevřena)“. Připomeň v obou, že přepnutí dispatch
je SAMOSTATNÉ rozhodnutí (spec §6).

- [ ] **Step 3: Finální sada + commit**

Run: celá sada z Global Constraints naposled.
Expected: vše 100 %.

```bash
git add tests/test_qgraph.py docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md docs/BACKLOG.md
git commit -F - <<'MSG'
feat(qgraph): #57 E2 — test rozšiřitelnosti + empirie (kritérium splněno)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```
