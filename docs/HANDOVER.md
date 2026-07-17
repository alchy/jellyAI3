# HANDOVER — předání projektu jellyAI3 (2026-07-18, večerní stav)

Tento dokument předává práci další session (jiná instance, pravděpodobně menší
model). Čti ho CELÝ před první změnou kódu. Doplňuje `docs/BACKLOG.md`
(CO dělat) o JAK pracovat a ČEHO se vyvarovat.

## 1. Zákony projektu (neporušitelné)

1. **Logika nikdy fixně v kódu.** Chování Iris řídí JSON karty
   (`jellyai/iris/patterns/cs/*.json`); kód smí obsahovat jen MECHANISMY
   (výpočet rysů, aritmetika, matching). Nový vzor chování = nová karta,
   ne `if` v kódu. I prahy rozhodování nesou karty.
2. **Dialog > figly.** Priorita je aktivace správných uzlů. Při nejistotě se
   ptát (QueryAssurance pod prahem karty → nabídka), bez odpovědi upřímný
   terminál („Nepodařilo se mi zaostřit… nejbližší: A, B"). Nikdy nehádat.
3. **Jazyk jako data.** Vše jazykově specifické patří do
   `jellyai/lang/cs.json` (+ normalizace v `jellyai/lang/__init__.py`).
   Nový jazyk = nový JSON. Do kódu nikdy český řetězec pravidla.
4. **Každá změna se měří.** Před commitem MUSÍ projít: pytest, etalon,
   focus, dialog (viz §3). Normativy neklesají. Když číslo klesne,
   je to nález — vyšetři, neobcházej.
5. **Korpusová evidence > lokální tag.** Tagger v jedné větě lže (mis-tagy
   VERB/NOUN, špatné pády). Rozhoduje hlasování přes korpus
   (`jellyai/graph/hygiene.py` — vzor pro nové guardy).
6. **Žádná zpětná kompatibilita.** Graf je prototyp — refaktoruj bez zátěže;
   nepoužívané cesty do `conserved_` (viz docs), ne mrtvý kód.
7. **Po každé změně restart služeb** (drží starý kód i graf):
   `pkill -f iris_service.py; lsof -ti :8080 | xargs kill; pkill -f "cli.py web"`
   pak `./jelly web` (a případně
   `.venv/bin/python services/iris_service.py --port 8084 --model data/graph.pkl`).

## 2. Stav (2026-07-18, commit 5e825a5)

- **Metriky:** 457 testů, etalon 29/29 (+5 gap řádků), focus 12/12,
  dialog 21/21 — vše 100 %.
- **Hotovo:** faktový graf; pseudo-QL šablony (hybrid, UDPipe jen fallback);
  Iris karty + QueryAssurance; Chronos; Mnemos (deník `data/memory.jsonl`,
  připsané fakty s elidovaným subjektem); hygiena (upos/pád/životnost/
  uvozovky/lemma hlasování); kanonizace v1+v2 (nominativizace id);
  instanční vrstva fáze 1 (srůst jen z textového tvrzení „X řečený Y");
  REST :8084; web 3 okna (viewBase — vlastní repo uživatele);
  subsystémový půdorys S0–S3 (`jellyai/iris/subsystems/`, spec
  `2026-07-18-subsystemy-iris.md`): připomínky s vlastním vláknem hodin,
  okno Reminder, kanály, tvrdý časový filtr.
- **Klíčová měření** (nedělej znovu, věř jim): kontextový otisk NEROZLIŠÍ
  identitu (Ježíš–Nazaretský 0.31 ≈ Jan–Křtitel 0.28 — první táž osoba,
  druzí dva lidé; manželé Němcovi 0.70). Proto srůst jmen jen z textového
  tvrzení. Viz `docs/superpowers/specs/2026-07-18-jmenny-uzel-instance.md`.

## 3. Testování a hodnocení kvality (před KAŽDÝM commitem)

```bash
.venv/bin/python -m pytest -q                     # musí: N passed, 0 failed
.venv/bin/python benchmark/run_etalon.py          # musí: JÁDRO 29/29 (100 %)
.venv/bin/python benchmark/run_focus.py           # musí: 12/12
.venv/bin/python benchmark/run_dialog.py          # musí: 18/18
.venv/bin/python benchmark/run_coverage.py        # diagnostika (sleduj trend)
```

- **Nikdy nespouštěj pytest přes `| tail` v řetězu s `&&`** — maska exit
  kódu už jednou pustila SyntaxError do main. Nejdřív celý běh, pak čti.
- **Nová feature = nový řádek benchmarku.** Odpovědní chování → řádek
  `benchmark/etalon.jsonl` (`{"q", "expect", "cat"}`; negativa přes
  `"reject"`); dialogové chování → scénář `benchmark/dialog.jsonl`
  (fixní hodiny `datetime(2026,7,17,12,0)` — determinismus); aktivace →
  `benchmark/focus.jsonl` (uzly v top-K jasu).
- **Gap řádky**: známý nedostatek se NEmaže, přidá se `"gap": "proč + odkaz
  na backlog"` — etalon ho reportuje zvlášť a hlídá, že se nezhorší.
- **Očekávání smí následovat zlepšení**: když se id uzlů zlepší
  (Betlémě→Betlém), aktualizuj očekávání benchmarku a ZDŮVODNI v commitu.
  Nikdy neaktualizuj očekávání, aby zakrylo regresi.
- E2E ručně: `./jelly web` a ptej se; nebo
  `curl -s -X POST localhost:8084/query -H 'Content-Type: application/json'
  -d '{"question":"…"}'`.

## 4. Mapa kódu (kde co žije)

| Soubor | Role |
|---|---|
| `jellyai/iris/automaton.py` | jádro Iris: turn() — hodiny→volba→focus-shift→konstatování→odpověď→karty |
| `jellyai/iris/patterns.py` + `patterns/cs/*.json` | balíček karet (deck.best = benefit-výběr) |
| `jellyai/iris/subsystems/{chronos,mnemos,topos}.py` | čas (intervaly, připomínky+plán, tep hodin, tvrdý filtr), paměť (deník, memorize/recall), prostor (gazetteer, kontejnment, učení za pochodu) |
| `jellyai/answerer/query.py` | pseudo-QL šablonový parser otázek (bez UDPipe) |
| `jellyai/answerer/graph_answerer.py` | match nad grafem, _resolve_topic (patra evidence), aktivační pole |
| `jellyai/graph/extract.py` | extrakce faktů z anotací (spony, apozice, aliasy „řečený") |
| `jellyai/graph/hygiene.py` | korpusová hlasování (upos/pád/životnost/lemma) + scruby + nominativize |
| `jellyai/graph/instance.py` | instanční vrstva (srůst z tvrzení, name_families) |
| `jellyai/graph/graph.py` | FactGraph, build, resolve_entities, remap_nodes, kontext asociace |
| `jellyai/lang/cs.json` | VŠECHNA česká data (tabulky, koncovky, karty-podpůrné seznamy) |
| `jellyai/tasks.py` | build pipeline: annotate→scrub_entities→build→recover→resolve→scruby→nominativize→instance |
| `services/iris_service.py` | REST :8084 (spouštět s `--port 8084 --model data/graph.pkl`) |
| `benchmark/run_*.py` + `*.jsonl` | čtyři benchmarky (guardrail) |

## 5. Zbývající features — implementační tipy

Pořadí dle BACKLOGu. U každého: kudy do toho a na co si dát pozor.

### #9 Detail uzlu (rychlá výhra — začni tímhle)
Rozkliknutí uzlu ve webu má ukázat tvary/aliasy (`graph.aliases[id]`),
kmen a vysvětlené role řádků obj/subj. Kód: `jellyai/viz/detail.py`
(+ `viewbase_view.py`). Čistě prezentační — benchmarky neohrozí. Dobrý
první úkol na osahání projektu.

### Ranking identit (etalon gap „Kdo je jezis?" → Kristus)
Poslední překážka gapu: šumová spona `být(Ježíš, Bůh)` (ze „Syn Boha")
přebíjí `jmenovat(Kristus)`/`druh(Mesiáš)`. NEřaď natvrdo v kódu — zvaž
kartu (např. identitní odpověď s více patry → nabídka kandidátů dialogem)
NEBO oprav data (fakt „Syn Boha" je vadná extrakce apozice — syn čeho/koho
je vztah, ne identita). Datová cesta je čistší.

### #5 zbytek — clarify karty
`clarify-period` („Kdy?" bez období → nabídka století/roků z dat grafu),
`clarify-relation`. Potřebují nové UDÁLOSTI v automatu (`deck.best("…")`
voláš v místě rozhodnutí, kartu přidáš do `patterns/cs/`). Vzor: jak turn()
hlásí `data.overflow` s `area_lit` guardem. Plus glow-dominantní řazení
výčtu po volbě oblasti (dnes aktivace jen řadí remízy).

### #10 — HOTOVO v S2 (tvrdý filtr, tests/test_time_filter.py); navazuje S3 Topos (spec §5).

### #24 Negace dějů
Deník má `neprší (Už)`. Negační prefix „ne-" je jazykové pravidlo →
cs.json (např. `"negation_prefix": "ne"`). Párování: predikát „prší" ↔
„neprší" jsou OPAKY — „Prší?" s faktem neprší(čas T) → „Ne, od T neprší."
Implementuj v answereru jako mechanismus (rozpoznání páru), text odpovědi
kartou. Souvisí s #10 (od kdy).

### #11 Metron („Kolikrát letos pršelo?")
Nová díra typu POČET VÝSKYTŮ: tázací tabulka cs.json („kolikrát") →
qtype count; answerer spočítá fakty predikátu (s Chronos filtrem z #10).
Zavře i etalon gap „Kolik měla dětí BN?" (počet distinct obj u faktů).

### #8 fáze 2 — instance per odstavec (VELKÉ, čti spec!)
Spec: `docs/superpowers/specs/2026-07-18-jmenny-uzel-instance.md`.
Neslučuj statisticky (měření v §2!). Kroky: (a) rozpuštění dvou-osobových
slepenců („Áronovi Mojžíš" — kmeny pokrývají 2 nezávislé osoby s otiskem
k oběma → remap k silnější), (b) instance per odstavec při buildu,
(c) jmenovka jako uzel typu `jméno` + hrana jmenovat. POZOR na poučení
z „Le": bezkmenná/prázdná množina je vakuově kompatibilní se vším;
kanon = nejkratší člen umí rozšířit chybu na celý graf. Po každém kroku
všechny benchmarky.

### #13 Sharpener, #12 Topos, #7 učení pojmů, #14 čistý řez
- #13: kontext hrany slabší při šíření (config váhy), K-křivka v run_focus.
- #12: hierarchie míst (Praha ⊂ Čechy) — paralela Chronosu (containment
  intervalů → containment míst), tabulky v cs.json ne natvrdo.
- #7: NEBEZPEČNÉ (kvalita/konflikty) — deník verzovat, zdroj=uživatel,
  fakty od uživatele nikdy nesmí tiše přepsat korpus. Kriticky promysli.
- #14: UDPipe pryč z query strany — gate splněn (etalon 28/28 v režimu
  `--mode templates`); smaž fallback větev v answer(), `question_pattern`/
  `analyze_question` přesuň do `conserved_`, run_etalon --mode udpipe zruš.

## 6. Pasti (draze zaplacené — neopakuj)

1. **Dvě chyby modelů se kryjí.** NER slepí jména a UDPipe současně rozbije
   syntax, která by NER vyvrátila („Ježíš Martu": NER P kontejner + Ježíš
   tagnut jako VERB). Jediná obrana: korpusová hlasování.
2. **Vakuová logika.** `all(...)` nad prázdnou množinou je True — „Le"
   (bezkmenné id) pohltilo Ježíše i Marii. U množinových podmínek vždy
   ověř neprázdnost.
3. **JSON karty**: uvnitř `"teach"` řetězců NIKDY ASCII uvozovky `"` —
   rozbité karty = 21 padlých testů. Používej „české".
4. **Maskování exit kódů** (pytest | tail) — viz §3.
5. **Benchmarky závisí na id uzlů.** Přestavba grafu může legitimně změnit
   id (nominativizace) — rozliš regresi od zlepšení, očekávání aktualizuj
   jen se zdůvodněním.
6. **`iris_service.py` vyžaduje `--port 8084 --model data/graph.pkl`** —
   bez argumentů spadne. Port 8083 drží morpho služba — při kolizi
   `lsof -ti :8083 | xargs kill`.
7. **Commit zprávy s uvozovkami**: použij `git commit -F - <<'MSG'` heredoc
   (inline -m s českými uvozovkami rozbíjí bash).
8. **Nečti celé velké soubory** (`graph_answerer.py` ~950 řádků) — griduj
   (`grep -n`) a čti okna. Kontext šetři na měření a testy.

## 7. Pracovní smyčka (doporučený postup na 1 úkol)

1. Přečti řádek BACKLOGu + související kód (grep, ne celé soubory).
2. Napiš test/benchmark řádek NAPŘED (červený).
3. Minimální implementace — mechanismus do kódu, rozhodování do karet/dat.
4. Celá sada: pytest + 4 benchmarky (§3). Vše 100 %? → dál.
5. Přestavěl jsi graf? Zkontroluj namátkou známé uzly (Ježíš, Božena
   Němcová, Betlém) — váhy nesmí nesmyslně skákat.
6. Commit (věcná CZ zpráva se zdůvodněním), push, restart služeb,
   aktualizuj BACKLOG (✓ + co zbývá).
7. Ověř E2E přes REST/web to, co uživatel uvidí.

Uživatel komunikuje česky, chce vidět tabulku dalších kroků, a průběžně
hlásí anomálie z webu — ber je vážně, každá zatím vedla k reálné chybě
v datech nebo mechanismu (viz git log).
