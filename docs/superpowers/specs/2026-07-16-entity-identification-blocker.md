# Blocker: jednoznačná identifikace entity (genitiv ↔ nominativ)

> ✅ **VYŘEŠENO (2026-07-16, větev `entity-resolution`)** cestou **A** — post-build
> `resolve_entities(graph)` + query-side kmenový fallback v `_resolve_topic`.
> Odchylka od návrhu níže: kanonický tvar clusteru **není** `argmax(subj_count,…)`
> — sonda ukázala, že pro-drop koreference rozdává podměty i genitivním uzlům
> (`Karla Čapka` subj=14, `Van Tocha` subj=7), takže subj_count vybírá genitiv.
> Místo toho **lexikograficky nejmenší člen clusteru**: pádové koncovky nominativ
> prodlužují, minimum = nominativ (ověřeno na všech 15 vícečlenných clusterech
> grafu). Stemmer doplněn o dativ `-ovi`; pravidla kmenování přesunuta do
> `jellyai/lang/cs.json` (jazyk = zásuvný modul). Výsledek: etalon **12/13 (92 %)**
> — obě strany bratr-symetrie i obě rekurze („Kde se narodil bratr X?") v jádru.
> Plán: `docs/superpowers/plans/2026-07-16-entity-resolution.md`.

> **Handoff pro novou instanci.** Refaktor odpovídače je hotový a commitnutý
> (`b9aee99`), etalon JÁDRO 9/11, 206 testů prochází. Tenhle dokument popisuje
> **blocker, který drží další rozvoj**, s tvrdými daty z grafu a s **cestami ven**.
> Není to hotový plán — je to analýza + varianty k rozhodnutí. Po přečtení běž
> přes brainstorming → plán → implementaci, s etalonem jako strážcem kvality.

## TL;DR

Čeština skloňuje vlastní jména. Jedna reálná entita se v grafu **tříští na mnoho
uzlů podle pádu** („Karel Čapek" / „Karla Čapka" / „Karlem Čapkem" / …). Fakty se
tím rozpadají mezi neslučitelné uzly a **průchod přes víc faktů (rekurze, relace,
navazující dotazy) selhává** — dotaz se rozřeší na jeden pádový uzel, kdežto
hledaný fakt visí na jiném. Potřebujeme **jeden kanonický uzel na entitu**,
konzistentně na straně build (id uzlů) i dotazu (rozřešení tématu).

Toto je **kořenový blocker**: bez něj nefunguje rekurze (`graf(subgraph(subgraph))`),
symetrické vztahy ani část etalonu — a každá vrstva nad tím dědí to samé tříštění.

## Symptom (dokázaný na datech)

Etalon má výmluvnou **asymetrii**:

```
Kdo byl bratr Karla Čapka?   → Josefa Čapka          ✓ PASS
Kdo byl bratr Josefa Čapka?  → V textu jsem nenašel   ✗ FAIL
```

Stejná otázka, jen prohozená jména — a jednou to projde, jednou ne. Proč:

### Skutečný stav grafu (`data/graph.pkl`, probe)

Jedna osoba **Karel Čapek** je roztříštěná nejmíň do 8 uzlů:

| uzel (id)               | typ    | váha | pád |
|-------------------------|--------|------|-----|
| `Karel Antonín Čapek`   | person | 38   | nom (3-slovný) |
| `Karel Čapek`           | person | 3    | nom (2-slovný) |
| `Karla Čapka`           | person | 15   | **gen** |
| `Karlem Čapkem`         | person | 5    | instr |
| `Karlu Čapkovi`         | person | 1    | dat |
| `Čapek`                 | person | 4    | holé příjmení |
| `Čapkova`, `Čapků`      | person | 1+1  | fragmenty |

Totéž **Josef Čapek**: `Josef Čapek` (nom, w=40), `Josefa Čapka` (gen, w=4),
`Josefem Čapkem` (instr, w=2), `Josefu Čapkovi` (dat, w=5).

Jediný „bratr" fakt v grafu:

```
bratr(subj="Karel Antonín Čapek",  obj="Josefa Čapka")
```

Biografické „narodit" fakty:

```
narodit(subj="Josef Čapek",          loc="Hronově")
narodit(subj="Karel Antonín Čapek",  loc="Malých Svatoňovicích", obj="Antonína Čapka")
```

### Mechanismus asymetrie (proč jednou PASS, jednou FAIL)

Dotaz projde `_resolve_topic` (querу-side, `graph_answerer.py:59`), který matchuje
**lemmata otázky proti `node.id`** (case-insensitive), preferuje víc pokrytých
témat → delší entitu → vyšší váhu. **Nenormalizuje pád** — spoléhá, že lemma
dotazu = povrch uzlu.

* **„bratr Karla Čapka":** lemma → „Karel Čapek". Nejlepší shoda = `Karel Antonín
  Čapek` (obsahuje slova Karel+Čapek, váha 38). To je **náhodou `subj` bratr-faktu**
  → match → vrátí druhého účastníka `Josefa Čapka`. Obsahuje „Josef" → **PASS**.
  (Pozn.: vrací se **genitivní** hodnota — projde jen díky substring-shodě.)
* **„bratr Josefa Čapka":** lemma → „Josef Čapek". Nejlepší shoda = `Josef Čapek`
  (nominativ, váha 40). Ale bratr-fakt drží `obj="Josefa Čapka"` (**genitiv**, jiný
  uzel!). `facts_of("Josef Čapek", "bratr")` = [] → **FAIL**.

Genitivní uzel `Josefa Čapka` (w=4) je **nedostupný** nominativním lemmatem — a
právě on nese vztah. Náhodná shoda u Karla („Karel Antonín Čapek" je subj) maskuje,
že celý mechanismus stojí na štěstí.

### Stejný kořen maskuje i „úspěch" rekurze

„Kde se narodil bratr Karla Čapka?" **vypadalo**, že rekurze funguje (vrátilo
místo), ale ve skutečnosti: SubQuery `bratr(Karel)` → vrátí `obj="Josefa Čapka"`
(genitiv) → `narodit(subj="Josefa Čapka")` → **[] žádný fakt** (narodit visí na
nominativním `Josef Čapek`) → pattern vrátí None → odpověď přišla z **fallbacku**
(Karlovo rodiště, špatně). **Jakmile sloučíme `Josefa Čapka` → `Josef Čapek`,
rekurze poběží doopravdy:** bratr(Karel)=Josef Čapek → narodit(Josef Čapek)=Hronově.

## Proč je to **blocker** (ne kosmetika)

Náš univerzální princip je *otázka → neúplný fakt → match → díra*, a **rekurze**
(`graf(subgraph(subgraph))`) na něm staví: mezivýsledek jednoho skoku je vstupem
dalšího. Když mezivýsledek přistane na pádovém uzlu, který nemá navazující fakty,
**každý víceskokový dotaz selže** nebo tiše spadne do fallbacku. Tj. blokuje:

* rekurzi (vnořené dotazy, „kdo byl bratr autora, který napsal R.U.R.?"),
* symetrické/relační vztahy (bratr, spolupracovník…),
* navazující (kontextové) dotazy skákající mezi fakty,
* a kazí i jednoskokové (vrací genitivní hodnoty, nedostupné uzly).

## Co už je postavené (nástroje k dispozici)

1. **`jellyai/graph/canon.py`** — `build_entity_canon(surface_freq)`, `cluster_key`,
   `_stem` (odstraní pádovou koncovku + **epentezi** eC→C: Karel→karl, Čapek→čapk,
   Božena→božn). **Otestované** (`tests/test_canon.py`), ale **PARKOVANÉ** — prosté
   zapojení do buildu dělá whack-a-mole (viz níže). Kmenový klíč = n-tice kmenů slov.
   ⚠️ **Mezery stemmeru na reálných datech:** dativ `-ovi` neřeší
   (`Čapkovi`→„čapkov" ≠ genitiv „čapk"), takže `Karlu Čapkovi` se **neshlukne**
   s `Karla Čapka`. Instrumentál `-em` funguje (Čapkem→čapk). Stemmer chce doladit
   nebo doplnit jiný shlukovací signál.
2. **`graph.py::_canonical_persons`** (build-time, per dokument) — sjednotí **jen
   délkové** fragmenty (Karel ⊂ Karel Čapek), **ne pády**. Deterministické (sorted).
3. **`jellyai/graph/spread.py`** — neighbor-spreading; `entity_candidates` recovne
   chybějící (role ②). Kmenový `_stem` tu je taky.

## Proč naivní opravy selhaly (poučení — nešlapat do stejného)

* **Build-time zapojení clusteringu = whack-a-mole.** Přejmenování uzlů na kanonický
  tvar rozhýbe **pro-drop koreferenci** v `build_graph` (`_warm_persons` svítí osoby
  podle kanonického jména; nejteplejší osoba dědí elidované podměty). Změna jmen →
  jiná „nejteplejší" → **fakty se přestěhují**. Etalon zůstal 9/11, jen s **jinou
  rozbitou dvojicí** (opravil Josefa-bratra, rozbil Karel-identitu; subject-weighting
  opravil Karla, rozbil Josef-místo…). Coreference a entity-resolution jsou **spřažené**.
* **Morfologie je nespolehlivá.** UDPipe lemmatizuje skloňovaná ženská jména špatně
  („Boženu"→„Božený"); MorphoDiTa `_to_nominative` je nekonzistentní a **maže
  diakritiku** („Boženu"→„Bozena"). Proto canon.py **neodvozuje nominativ**, ale
  shlukuje varianty a bere nejčastější tvar. Cesta přes externí morfologii = slepá.
* **Nebezpečí over-merge (z dat!).** `Antonína Čapka` (w=19) je **otec** Antonín
  Čapek, ne syn Karel Antonín Čapek — ale kmeny se překrývají. Holé `Čapek` (w=4)
  je subset kohokoli z Čapků. Hladové subset-slučování by **spletlo otce se synem**
  a natáhlo holé příjmení k náhodnému nositeli. Slučování musí být **konzervativní**.

## Invarianty / mantinely (každá cesta je musí držet)

1. **Etalon ≥ 9/11** po celou dobu; cíl je **10/11** (projde „bratr Josefa Čapka").
   Měř `.venv/bin/python benchmark/run_etalon.py` po každém kroku.
2. **Determinismus** buildu (žádné `set`-pořadí; tie-break lexikograficky).
3. **Slučuj jen v rámci typu** (person s person); časové/číselné uzly **nešlukovat**
   (datum ≠ jméno). Začni **jen na `person`** (tam je blocker).
4. **Konzervativně proti over-merge:** neslučuj holé fragmenty (1 slovo) do víc
   různých delších clusterů; ambiguózní (Antonín otec vs. syn) nech radši rozdělené.
5. **Query-side i build-side musí použít TÝŽ mechanismus**, jinak se rozejdou.

## Cesty ven (seřazeno dle doporučení)

### ⭐ A) Post-build resolver — `resolve_entities(graph)` (DOPORUČENO)

**Myšlenka:** oddělit entity-resolution od koreference. Build necháme **beze změny**
(deterministický, pro-drop nerušený) a přidáme **samostatný pass po buildu**, který
sloučí pádové varianty a přepíše graf. Tím zmizí whack-a-mole (nehýbeme tím, co
pro-drop za běhu vidí; sjednocujeme až na konci).

**Algoritmus:**
1. Pro každý `person` uzel spočítej z `_by_node`: `subj_count` (v kolika faktech je
   v roli `subj`) a `weight`. Subjekt je v češtině nominativ → `subj_count` je proxy
   pro „nominativnost".
2. Shlukni person-uzly podle `cluster_key` (canon.py). Volitelně **konzervativní
   subset-merge** délkových fragmentů (jen jednoznačné, viz mantinel 4).
3. **Kanonický tvar clusteru** = `argmax (subj_count, weight, -lex)` → vybere
   nominativ (`Josef Čapek`, ne `Josefa Čapka`).
4. Postav `node_map: old_id → canonical_id`. **Přepiš graf:** znovu složené fakty
   s přemapovanými účastníky (kolize → agreguj váhu), přepočítej `nodes` (součet vah,
   typ kanonického), přestav index `_by_node`. Ulož zpět.
5. **Query-side:** navěš na graf `stem_key → canonical_id` (a `variant → canonical`).
   V `_resolve_topic`/`_solve` termín dotazu nejdřív normalizuj přes tuhle mapu
   (spočti `cluster_key` termínu → kanonický uzel), pak matchuj.

**Proč nejlíp:** dvě čisté, testovatelné fáze (raw graf → kanonizace); build
nerušený → mizí whack-a-mole; `subj_count` řeší volbu nominativu; jeden mechanismus
sdílený build+query. Cílený efekt: `Josefa Čapka`→`Josef Čapek` opraví bratra i
rekurzi naráz. **Nejdřív dolaď stemmer** (dativ `-ovi`), ať se sloučí i `Čapkovi`.

### B) Build-time globální `EntityIndex`

Nahradit per-dok `_canonical_persons` globálním indexem (varianta→id) použitým v
`_node_for`/`_warm_persons`/query. **Riziko:** přesně ten build-time coupling, co
dělal whack-a-mole. Zvaž jen když A) narazí; pak nutně s regresním během etalonu
po každém kroku.

### C) Jen query-side normalizace (levné, ale poloviční)

Normalizovat jen termíny dotazu na kanonický uzel. **Nefunguje plně:** vrácená
hodnota zůstane v pádu, jakým je uložená (`Josefa Čapka`), a fakty jsou pořád
roztříštěné mezi uzly → rekurze dál padá. Použitelné leda jako dočasná berlička.

### D) Externí morfologie (ZAMÍTNUTO)

UDPipe/MorphoDiTa nespolehlivé (viz výše). Neinvestovat.

## Ověření (definition of done)

* `.venv/bin/python benchmark/run_etalon.py` → **JÁDRO 10/11** (projde „Kdo byl
  bratr Josefa Čapka?" → obsahuje „Karel"); žádná regrese ostatních.
* Ruční: „Kde se narodil bratr Karla Čapka?" → **Hronově** (Josefovo rodiště, ne
  Karlovo z fallbacku) — důkaz, že rekurze běží doopravdy.
* Přidat do `benchmark/etalon.jsonl` nové případy: obě strany symetrie + jeden
  víceskokový (rekurzní) dotaz, ať blocker nikdy nezregreduje.
* `.venv/bin/python -m pytest -q` zelené (206+).

## Soubory, kterých se to dotkne

* `jellyai/graph/canon.py` — dolaď `_stem` (dativ `-ovi`), možná API pro resolver.
* `jellyai/graph/graph.py` — nový `resolve_entities(graph)` volaný na konci
  `build_graph` (za `_decompose_dates`); navěsit mapu na graf pro query-side.
* `jellyai/answerer/graph_answerer.py` — `_resolve_topic`/`_solve` normalizují
  termín přes sdílenou mapu před matchem.
* `benchmark/etalon.jsonl` — nové normativní případy.
* `tests/` — test resolveru (sloučení pádů, konzervativnost proti otec/syn,
  determinismus) + test query-side normalizace.

## Otevřené otázky pro rozhodnutí

1. Kanonický tvar: `Karel Antonín Čapek` (nejfrekventovanější, 3-slovný) vs.
   `Karel Čapek` (2-slovný nominativ)? Subject-weighting vybere ten s víc podměty —
   ověřit, který to je, a jestli substring-etalon obojí snese.
2. Kam se subset-mergem (délkové fragmenty)? Řešit v resolveru, nebo nechat na
   stávajícím `_canonical_persons` a resolver dělá jen pády?
3. Geo uzly (`Slezsku`↔`Slezsko`) taky tříštěné — zapojit později stejným passem,
   nebo držet `person`-only?

---
Souvisí s [[jellyai3-fact-graph]] (blocker #1 tříštění entit). Handoff stav:
refaktor `b9aee99`, canon.py parkovaný a otestovaný, etalon 9/11 baseline.
