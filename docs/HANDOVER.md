# HANDOVER — předání projektu jellyAI3 (2026-07-20 večer, po dni otázkového grafu)

Tento dokument předává práci další session (jiná instance, pravděpodobně menší
model). Čti ho CELÝ před první změnou kódu. Doplňuje `docs/BACKLOG.md`
(CO dělat) o JAK pracovat a ČEHO se vyvarovat. K architektuře čti
`docs/ARCHITEKTURA.md` (kap. 7 = dva grafy) a pro hloubku
`docs/architektura-web/index.html` (17 kapitol offline). Specy dne
2026-07-20 (adresář `docs/superpowers/specs/`):
`2026-07-20-otazkovy-graf.md` (+ dotažení, jednotný dispatch,
testování Bible) — otázkový graf je nyní JEDINÝM dispatcherem
vstupu a answerer má kaskádu poctivých odpovědí.

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
7. **Po každé změně restart služeb** (drží starý kód i graf). PRACOVNÍ
   POSTUP (ověřený 2026-07-19): Iris běží SAMOSTATNĚ — po změně jádra
   stačí `kill $(lsof -ti :8084)` + start
   `.venv/bin/python services/iris_service.py --port 8084 --model data/graph.pkl`
   a GUI (:8080) běží dál (napojí se dalším dotazem). Web restartuj jen
   při změně viz/cli vrstvy. DEV MOST: řádky připsané do
   `data/web_inbox.txt` zpracuje web jako vstup dialogového okna —
   testy jsou vidět v GUI (echo "Prší?" >> data/web_inbox.txt).

## 2. Stav (2026-07-20 večer — den otázkového grafu, ~50 commitů)

- **Metriky (po bloku P1–P4, 2026-07-20 noc):** 622 testů; etalon
  42/42 (GAP 12/4), focus 12/12, dialog 45/45 (GAP 8/0), ZÁPIS 34/34,
  qgraph harness 5 rovin 100 % v obou variantách (etalon rovina
  60/60). Etalon ověřen bitově shodný napříč PYTHONHASHSEED. Graf:
  7433 uzlů / 18841 faktů (ADJ pasivum +140 faktů, fold participií
  73, klauzulové slepence −20, dvou-osobové slepence −5).
- **Blok P1–P4 (2026-07-20 noc) NASAZEN na produkci:** P1 = zbytky
  dávky D hotové (ADJ participia: extrakce + fold_participles +
  karty q-pasivum obou slovosledů, pacient=obj; scrub_clause_objects
  s hlasy POVRCHOVÝCH tvarů + výrok na pohybovém slovese); P2 = C10
  zárodek Echo (výroky po jednom v uvozovkách, děje středníkem,
  theme/time/num mimo obsah, nabídka bez 2× výpisu); P3 = #8 fáze 2
  bod 2 (dissolve_glued_persons — ČTYŘNÁSOBNÁ evidence, klíčová je
  „komponenty spolu vystupují v týchž faktech": bez ní padl i Josef
  Čapek, holé uzly jsou smíšené a otisk lže); P4 = karty q-s-kym
  („S kým mluvil Hospodin?" → Mojžíš) + q-rekl-podmet-adresatovi
  (nález parity: dativ s podmětem býval druhý subj).
- **Refaktor blok (2026-07-20 pozdě večer):** TurnResult
  (`answerer.turn`, `begin_turn()`, pick_focus/query_card parametry
  answer()), lint karet (KNOWN_EVENTS + QUERY_ACTION_KEYS — mrtvá
  resolve-miss zakonzervována), JEDNO osvětlení na tah (vítězná
  dotazová karta hintem), `facts_of_predicates` primitivum,
  DOTAZOVÁ polovina dávky D (nárok oblasti u děrových otázek,
  druhý ořez klíčů gazetteeru, díra ukotvená místem — „Kdo se
  narodil v Betlémě?" → Ježíš).
- **Otázkový graf (#57 + #51) KOMPLETNÍ:** kompilát karet s PĚTI
  rodinami uzlů (otazka/vyrok/prikaz/worker/clarify) je JEDINÝM
  dispatcherem vstupu — přímí experti přes registr claimů
  (`jellyai/iris/claims.py`), výroky/příkazy/recall vítězem osvětlení
  (`_command_turn`), hranice dotaz×výrok = rys `otaznik` na kartách,
  stavové tahy = pozice (`DialogPosition`). Rodinné karty s dimenzemi
  (`_expand_family` — q-otaz.json, q-zjistovaci.json), odvozené
  hrany, instance ze schématu (`predicate_roles`+`instance_lit`),
  proaktivní nabídky (`answer-offer-roles`).
- **Kaskáda poctivých odpovědí (princip user: tápání ≠ terminál):**
  data se VŽDY ověří — chytrá clarifikace (role neexistuje: „…kde
  nevím"), ČÁSTEČNÁ odpověď (fakty jsou, díra neplnitelná: „kdo
  nevím; vím kde: …"), NABÍDKA kandidátů s volbou (téma predikát
  nenese — volba přehraje otázku substitucí). Kontextové patro jen
  bez verdiktu schématu nebo pro řídká témata (hub hranice
  `context_hub_limit` — Ježíš 209 sousedů × R.U.R. 24; POZOR:
  „Kdo napsal R.U.R.?" JE asociační odpověď, fakt napsat neexistuje).
- **Další novinky:** vidové páry + třída setkání (predicate_synonyms),
  POHYBOVÁ třída zvlášť (movement_predicates — jen místní díry!),
  třídy dějů (predicate_classes: zázrak → q-trida-deju), supletivní
  katalog (přišel→přijít), početní karta q-kolik-pocet (počet =
  vlastnost situace), typové guardy děr (theme/loc/time/num ≠ osoba).
- **Dokumentace:** `docs/architektura-web/` (17 kapitol, offline,
  vč. `postrehy-refaktor.md` — 20 nálezů review, částečně provedeno).
- **Klíčová měření (nedělej znovu, věř jim):** kontextový otisk
  NErozliší identitu (spec instance); váhy telemetrie v dispatch =
  mrtvá větev (0 remíz); hub hranice asociací změřena (209×24).

## 3. Testování a hodnocení kvality (před KAŽDÝM commitem)

```bash
.venv/bin/python -m pytest -q                     # musí: N passed, 0 failed
.venv/bin/python benchmark/run_etalon.py          # musí: JÁDRO 100 %
.venv/bin/python benchmark/run_focus.py           # musí: 12/12
.venv/bin/python benchmark/run_dialog.py          # musí: 100 % (gap zvlášť)
.venv/bin/python benchmark/run_mnemos.py          # musí: ZÁPIS 100 %
.venv/bin/python benchmark/run_qgraph.py          # musí: všech 5 rovin 100 %
.venv/bin/python benchmark/run_qgraph.py --variant weights   # i s vahami
.venv/bin/python benchmark/run_coverage.py        # diagnostika (sleduj trend)
```

- **Nikdy nespouštěj pytest přes `| tail` v řetězu s `&&`** — maska exit
  kódu už jednou pustila SyntaxError do main. Nejdřív celý běh, pak čti.
- **Parity gate `--mode templates` je ZRUŠEN řezem #14** (2026-07-19):
  šablony (vzorové karty + pseudo-QL) jsou JEDINÁ dotazová cesta, UDPipe
  fallback i `query_mode` neexistují. Kartová regrese se pozná přímo na
  etalonu — karty mají před pozičními šablonami přednost; krádež otázky
  jiného smyslu (pasti 9–11) hlídej dál.
- **Deck při načtení LINTUJE karty**: neznámý event nebo neznámý klíč
  `query` akce = ValueError nahlas (rejstříky `KNOWN_EVENTS`
  a `QUERY_ACTION_KEYS` v `patterns.py`). Nový event/klíč = rozšířit
  kód, který ho konzumuje, I rejstřík. Testy mechaniky decku se
  syntetickými eventy: `PatternDeck(dir, known_events=None)`.
- **Nová feature = nový řádek benchmarku.** Odpovědní chování → řádek
  `benchmark/etalon.jsonl` (`{"q", "expect", "cat"}`; negativa přes
  `"reject"`); dialogové chování → scénář `benchmark/dialog.jsonl`
  (fixní hodiny `datetime(2026,7,17,12,0)` — determinismus); aktivace →
  `benchmark/focus.jsonl` (uzly v top-K jasu); ZÁPIS Mnemos (výrok→parse)
  → řádek `benchmark/mnemos.jsonl` (`{"u", "kind", "predicate", "objects"
  podmnožinou, "reject_objects", …}`; `kind: null` = nemá být rozpoznán;
  `--nom` měří i nominativizaci — chce běžící ÚFAL služby).
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
| `jellyai/iris/subsystems/{chronos,mnemos,topos,metron}.py` | čas (intervaly, připomínky+plán, tep hodin, tvrdý filtr), paměť (deník, memorize/recall), prostor (gazetteer, kontejnment, učení za pochodu), aritmetika (výrazy, slovní číslovky) |
| `jellyai/iris/qgraph.py` | kompilát otázkového grafu (compile/illuminate/decorate, DialogPosition) — JEDINÝ dispatch vstupu (#57/#51) |
| `jellyai/iris/claims.py` | registr nároků přímých expertů (Metron/Chronos/meta-focus) — pořadí bran = data |
| `jellyai/lang/lexer.py` | JEDEN určovač druhů slov: TaggedToken s MNOŽINOU hypotéz tříd (byt=spona i subst.) — #46 fáze 1 |
| `jellyai/lang/matcher.py` | vykonavatel vzorů: regulární sekvence tříd, spany `uzel+`, zbytek `*`, vylučování `!`, grok-zkratky `%{…}` (tabulka `pattern_aliases`) |
| `jellyai/answerer/query.py` | `_card_query` (vzorové karty event `utterance.query` — PŘEDNOST) → poziční pseudo-QL šablony (fallback); UDPipe v dotazu NEEXISTUJE (řez #14) |
| `jellyai/answerer/graph_answerer.py` | match nad grafem, _resolve_topic (patra evidence), aktivační pole; výsledek tahu = `answerer.turn` (TurnResult, `begin_turn()` ho vymění celý) |
| `jellyai/graph/extract.py` | extrakce faktů z anotací (spony, apozice, aliasy „řečený") |
| `jellyai/graph/hygiene.py` | korpusová hlasování (upos/pád/životnost/lemma) + scruby + nominativize |
| `jellyai/graph/instance.py` | instanční vrstva (srůst z tvrzení, name_families) |
| `jellyai/graph/graph.py` | FactGraph, build, resolve_entities, remap_nodes, kontext asociace |
| `jellyai/lang/cs.json` | VŠECHNA česká data (tabulky, koncovky, karty-podpůrné seznamy) |
| `jellyai/tasks.py` | build pipeline: annotate→scrub_entities→build→recover→resolve→scruby→nominativize→instance |
| `services/iris_service.py` | REST :8084 (spouštět s `--port 8084 --model data/graph.pkl`) |
| `benchmark/run_*.py` + `*.jsonl` | čtyři benchmarky (guardrail) |

## 5. Priority a implementační tipy (stav 2026-07-20 večer)

Plná tabulka priorit: BACKLOG + protokol
`docs/superpowers/specs/2026-07-20-testovani-bible-nalezy.md`. Shrnutí:

| P | Bod | Tip kudy do toho |
|---|---|---|
| 1 | **D — HOTOVO obě poloviny** ✅ (viz BACKLOG #2). Zbytky dávky: ADJ participia pasiva (409 vět), dotazová strana pasiva (gap „Kde byl Ježíš pokřtěn?"), klauzule jako objekt | mechanismy hotové jsou vzorem: `scrub_false_persons` + `name_position_votes` (hlasování jmennosti), pasivní větev v `extract_facts`, fold v `make_fact` |
| 2 | **C10 — formát odpovědí (zárodek Echo #20)** | výroky po jednom v uvozovkách, agregace větami, nominativizace hodnot v textech, kandidáti nabídky bez 2× výpisu; šablony v cs.json, skládání v answereru |
| 3 | **#8 fáze 2 — střepy Ježíše** | spec `2026-07-18-jmenny-uzel-instance.md`; NEslučuj statisticky (změřeno!); po každém kroku benchmarky |
| 4 | **T5/E1b — předložkové dekorační prvky karet** („s kým", „k Ježíšovi") | rozšíření rodin q-otaz o volitelné dekorační prvky; pasti 9–11 |
| 5 | **#41 oceán vrstev** | zárodek nad ději = `predicate_classes`; entity přes druh-hrany; přímý fakt > zděděný |
| 6 | **Odpovědní graf + TurnResult** (park → otevřít) | spec dotažení §1; sjednotí 9 side-channelů answereru (postrehy-refaktor 2.1) a dokončí rozklad `_turn` |
| 7 | **#39 provenience, #47 event log, #17 nález** (memory.jsonl tracked × reminders ne) | před dalšími zapisovateli (Ollama #30, STT #42) |
| — | Drobné: #28 okno Paměť, #36 font, #21 httpx, #15 kbelík, #50 sparse | „při nejbližším dotyku" |

Deploy: produkce `jelly.ithosaudio.eu` (server mail1.lordaudio.eu,
user tech) — po pushi `git pull` + restart Iris (`--port 8084
--model data/graph.pkl`); handshake `GET /version` MUSÍ ukázat
aktuální SHA (past: běžela ranní verze a testovala se stará).
POZOR: produkční `data/memory.jsonl` je paměť uživatele — při pullu
nesmí být přepsána (je tracked — nález #17).

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
8. **Nečti celé velké soubory** (`graph_answerer.py` ~1000 řádků) — griduj
   (`grep -n`) a čti okna. Kontext šetři na měření a testy.
9. **Hypotézové třídy se KRYJÍ**: „byl" je l_tvar I spona — vzorová karta
   s prvkem `l_tvar` ukradla identitní otázky („Kdo byl robot?").
   Vylučuj prvkem `l_tvar!spona` (%{SLOVESO_MINULE}).
10. **Volné orákulum spanů**: `_span_is_node` toleruje skip slova →
    span „Pavla v" projde! Matcher proto zakazuje spany začínající/končící
    třídou funkcni — NEOSLABUJ to.
11. **Karta se NEHLÁSÍ, když predikát nezná** (`_verb_match` → None):
    překlep „sworil" musí propadnout na poziční šablony (UDPipe
    fallback zrušen řezem #14). Kartová cesta nikdy nebere povrchový
    tvar neznámého slovesa.
12. **Vokativ z izolovaného taggeru je šum**: „Marcele" → vok. „Marcel"
    (rodový flip). Nominativizace jmen tag s pádem 5 ZAHAZUJE.
13. **ASCII uvozovky v JSON kartách/sadách** (stará past 3 platí dvojnásob):
    „…" v teach/gap řetězcích VŽDY české — rozbité karty = desítky testů.
14. **Grok-zkratky**: víceprvková zkratka (splice) POSOUVÁ indexy $N
    v action — odkazy míří na ROZVINUTÉ prvky. Jednoprvkové zkratky
    indexy nemění.
15. **cs.json NIKDY needituj sedem/ručním stringem** — vždy
    json.load/dump (ensure_ascii=False); visící čárka rozbila celý
    jazyk. Python heredoc s českými texty: uvnitř "" řetězců jen
    „…“ uvozovky (ASCII " ukončí string — spadlo to 2×).
16. **Rodinné karty**: soubor `q-otaz.json` vyrábí karty
    q-otaz-minuly… — grep jména karty soubor NENAJDE; prázdný slot
    smí být jen POSLEDNÍ prvek vzoru ($N se neposouvají).
17. **Synonyma × třídy**: `predicate_synonyms` = vidové páry/těsná
    synonyma (kruh všude); `movement_predicates` = SMĚROVÁ třída
    (jen místní díry — u osob obrací děj: odejít≠přijít);
    `predicate_classes` = kategorie dějů (agregace členů). Nemíchat.
18. **Kaskáda prázdných odpovědí**: sponové/dekompoziční predikáty
    (`cascade_skip_predicates`) do ní NEpatří (identita má vlastní
    patra); s místním filtrem se nespekuluje; verdikty rolí přes
    `_ring_roles` (normalizace smí vybrat člen kruhu bez faktů).
19. **Nedeterminismus přes iteraci setů** (#58, VYŘEŠENO — poučení
    trvá): vítěz remízy braný z neseřazené množiny závisí na
    PYTHONHASHSEED („Udělat"ד udělat" v `_verb_match` střídalo
    odpovědi mezi běhy). Nové iterace přes množiny VŽDY řadit — vzor
    `_canon_first` v query.py (malopísmenný predikát před
    kapitalizovaným šumem, pak abecedně); determinismus ověřuj
    bitovým diffem benchmarků pro dva různé hash seedy.
20. **Deploy: rsync bez --delete nechává MRTVÉ karty** — lint decku
    (KNOWN_EVENTS) je odmítne a Iris na produkci padá při startu
    (stalo se 2026-07-20: resolve-miss + 4 staré rodinné soubory
    q-otaz-*/q-zjistovaci-*). Po rsync VŽDY porovnat výpis
    `patterns/cs/` se stagingem a přebytky smazat. Pozn.: porty ÚFAL
    v config.py (8091/92/93) jsou už v upstreamu — dřívější „jediná
    živá odchylka" zanikla; /version na archivním deployi hlásí
    sha=unknown (poctivě nesoudí).
21. **Kapitalizované tvary v hypotézách tříd**: „Božena" končí na
    -ena (tvar participia) — vzorové prvky nových tříd vylučuj
    idiomem `trida!jmeno` (jako `l_tvar!spona`, past 9), jinak karta
    ukradne výběr a shodí kartovou cestu na šablonu. Odhalí parita
    qgraph (dekorace/etalon rovina).

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
