# Fáze 1 — Průzkum (knižní dokumentace hypothesis-one)

Výstup fáze 1 podle zadání: registr nálezů, tabulka pokrytí staré dokumentace, skutečné fáze
pipeline vs. zadání, slovník pojmů, seznam otevřených bodů, posbíraná naměřená čísla.
Vede se v souboru (práce přesahuje jedno sezení). Stav ke commitu `b8b2d5c`, větev `hypothesis-two`.

---

## A. Registr nálezů

Formát: **označení | co zjištěno | kde v projektu | kategorie | kam v dokumentaci**.
Kategorie dle zadání 1.1.

| # | Zjištění | Kde | Kategorie | Umístění |
|---|---|---|---|---|
| N1 | Runtime je DETERMINISTICKÝ; jediná živá neuronová služba je UDPipe 2 (rozbor otázky), zbytek symbolický | answering.py, config services.udpipe_url:8092 | věcný poznatek | Díl I, Díl IV |
| N2 | „R.U.R." se tokenizuje na R/U/R → mountne se bible → doména selhává; merge_abbreviations na korpus opraveno, na živý dotaz REVERTNUTO (regrese) | annotate_corpus, normalize | past + otevřený bod | 4.5, Díl VIII |
| N3 | is_factual vracelo False při JAKÉMKOLI „?" tokenu → stray „?" v závorce („Vídeň ?") zahodil definiční větu → OPRAVENO (jen koncová otázka) | grammar_vzor.is_factual | rozpor/oprava | 4.5, Díl III fáze syntéza |
| N4 | Kopula HLAVNÍ klauzule má přednost před slovesem VEDLEJŠÍ věty (fallback next(VERB) hijackoval kopulu „Králík je býložravec, který VYUŽÍVÁ…") → OPRAVENO | synth_registry | past + oprava | 4.5, fáze syntéza |
| N5 | Naučená assurance brána PŘEUČUJE na 76 vzorcích × 13 rysů (LOO −2 vs běh −19 potlačených) → gate_enabled=false | answering._gate_p, gate.json, config | vědomé provizorium | Díl VIII, Díl V |
| N6 | Kvantifikátory (množství/řada) nejsou nikdy faktická odpověď → filtr jako echo (nonanswer_lemmas) | answering._candidates, config | věcný poznatek | Díl III fáze inference |
| N7 | Alias merge (Barbora Panklová ≡ Božena Němcová) rozpouští rozpadlé jméno v assurance | dataloader _aliases, answering._canon | věcný poznatek | Díl IV |
| N8 | Relation-extrakce (vztah=hrana) funguje jen pro KOPULOVÝ vzor; oblique/pronoun-genitiv zkoušeny, MĚŘENO net-neutral (swap + šum) → vráceno | extract_relations | rozpor záměr/skutečnost | Díl VIII |
| N9 | Test-pater je MÁLO: jen test_determiner.py + test_phase1.py (jednotkové); zbytek jsou etalony (eval_*) = metrické/regresní; chybí smluvní, vlastnostní, zátěžové | test_*.py, eval_*.py | nedodělek | Díl V |
| N10 | Data (korpus/index/fakty/šablony) jsou gitignored → reprodukovatelné z raw + build_*; na deploy NUTNÉ přestavět | .gitignore, data/ | past | Díl II, Díl VII |
| N11 | gate.json je COMMITNUTÝ artefakt (malé váhy), ale gate_data.json (trénink) je gitignored → reprodukce přes train_gate.collect() | train_gate.py, gate.json | věcný poznatek | Díl V, Díl VI |
| N12 | Vazba na PŮVODNÍ TEXT: fakt nese (doc, sent) proveniencí (Fact.doc/sent), token nese start/end offset; napříč fázemi se drží doc+sent | fact_store.Fact, annotate_corpus | věcný poznatek | 3.1 rozhraní fází |
| N13 | Honest-negative kategorie 7/7 (100 %) — systém POCTIVĚ mlčí na fakty mimo korpus (nehádá); je to VLASTNOST, ne chyba | eval_large, answering assurance | věcný poznatek | Díl IV, Díl V |
| N14 | Confident-wrongy soustředěny v relation/authorship-who/attribute — systém tam NEmlčí, hádá profesní slovo (bratr→spisovatel) | eval_large baseline_large.json | past | 4.5, Díl V |
| N15 | Bible plošně znečišťuje mount (i pro Čapek-dotaz ~0.7 aktivace) — pre-existing chování select_files (doclink co-mount) | dataloader.select_files | známé omezení | Díl VIII |
| N16 | Řetězení kontextu (carry_context) — navazující otázka bez entity zdědí téma, otázka s entitou přepne (žár+mount zhasnou) | answering.answer | věcný poznatek | Díl III inference, Díl IV |
| N17 | Existují SIMULAČNÍ skripty (sim_*.py) a starší phase1.py/roles.py — ověřit, zda jsou živé, nebo mrtvé (kandidáti na známé omezení / úklid) | sim_*.py, phase1.py | otázka na tým | dotazy na tým |

> **Pozn.:** registr je živý — při psaní kapitol se doplňuje a promítá zpětně (zadání 1.1).

---

## B. Tabulka pokrytí staré dokumentace

24 HTML souborů v `docs/`. Přebírá se OBSAH, ne forma; pořadí ani formulace se nepřebírají.

| Starý soubor | Věcné jádro | Kam v nové knize | Poznámka (stálost) |
|---|---|---|---|
| index.html | rozcestník + „co stavíme" | Díl I (přepis) | aktuální |
| stav.html | aktuální metriky, co hotové/ne | Díl I + Díl V (čísla) + Díl VIII | AKTUÁLNÍ (nejnovější) |
| dialogovy-automat.html | JÁDRO konceptu (světlo→VZOR→assurance) | Díl IV | aktuální, flagship |
| aktivace-dialog.html | aktivace/světlo, §4b větev kontextu, Nedostatky | Díl IV + 4.5 | aktuální |
| vzor-matcher.html | šablony × aktivace, matcher | Díl IV + Díl III (VZOR fáze) | aktuální |
| faze-1-preprocessing.html | ① UDPipe anotace | Díl III fáze 1-2 | aktuální |
| faze-2-dataloader.html | ② tf·idf, indexy | Díl III fáze indexace | ověřit subject/doclink/alias doplnění |
| faze-3-grammar.html | ③ VZOR, role_catalog | Díl III fáze gramatika | aktuální |
| faze-4-synthesis.html | ④ registry, fakty, šablony | Díl III fáze syntéza | ověřit bio/relations/chronos doplnění |
| faze-5-queryparser.html | ⑤ answering, kandidáti, assurance | Díl III fáze inference + Díl IV | doplněno carry_context, polar, gate, relations |
| naucene-smerovani.html | směr NN nad VZORY, pilot, brána | Díl IV + Díl VIII | aktuální |
| aliasy-identita.html | vztah tokenů=identita, alias merge | Díl IV | aktuální |
| koreference.html | mechanismus plnění děr | Díl III/IV | ověřit proti fill_holes |
| koreference-do-faktu.html | pro-drop podmět do who-slotu | Díl III fáze syntéza | aktuální |
| diry.html | placeholdery (zájmena, elidovaný podmět) | Díl IV slovník/koncept | ověřit |
| dynamicke-nahravani.html | mount dle matche (#60) | Díl III + Díl IV | aktuální koncept |
| anotacni-cache.html | offline cache, formát shardů | Díl III fáze 1-2 + 2 | aktuální |
| externi-zdroje.html | UDPipe 2, NameType, Ollama | Díl I slovník + Díl II | ověřit Ollama (živé?) |
| terminy.html | WORD/SLOT/VZOR/DÍRA klíče | Díl I slovník (rozšířit) | základ slovníku |
| pouziti.html | jak spustit (ask.py), přidat data | Díl II + Díl VI | aktuální |
| fuze.html | fúzní studie parent×experiment | Díl IV zdůvodnění + Díl VIII | historický kontext |
| otevrene-otazky.html | „stav 64 %", priority P1-P3 | Díl VIII (přepsat) | ZASTARALÉ (64 %, staré priority) → rozpor s kódem |
| k-doreseni.html | běžící backlog nedostatků | Díl VIII registr | ověřit aktuálnost |
| napady.html | příležitosti „kde využít" | Díl VIII možné cesty | ověřit aktuálnost |

> Rozpory kód×dokumentace (→ Díl VIII + tato tabulka): otevrene-otazky.html tvrdí 64 % (kód dnes
> velký etalon 43 %, malý 72 %); k-doreseni/napady mohou obsahovat už vyřešené body.

---

## C. Skutečné fáze pipeline vs. zadání (oddíl 3.1)

Zadání dává 12 obecných fází; projekt má tuto skutečnou cestu (ověřeno v kódu):

| Skutečná fáze | Modul | Odpovídá fázi zadání |
|---|---|---|
| 1. Vstup / raw korpus | data/raw/*.txt | Extrakce dat |
| 2. Anotace (UDPipe 2 → CoNLL-U, shardy) + merge zkratek | annotate_corpus.py, normalize | Preparsing + Normalizace + Segmentace + Tokenizace + Jazyková analýza (SLITY do UDPipe) |
| 3. Indexace (tf·idf, idf, subjects, doclinks, aliases) | dataloader.build_indexes | Reprezentace (částečně) |
| 4. Gramatika / VZOR (frame_sig, role_catalog) | grammar_vzor.py, role_catalog.py | Reprezentace (VZOR) |
| 5. Syntéza faktů (registry → fakty → šablony; +bio/relations/chronos/gazetteer) | synth_registry, build_facts_all, build_templates_all, extract_* | Trénink (stavba mapy faktů) |
| 6. Inference / dotaz (mount, aktivace, kandidáti, assurance, polar, relations, gate) | answering.py | Inference |
| 7. Odpověď / mode | answering.answer → {answer, mode, offer} | Postprocessing |
| 8. Vyhodnocení (etalony) | eval_large, eval_answers, eval_coref | Vyhodnocení |
| — | (uložení = shardy na disk, gitignored) | Uložení a export |

> **Rozpor „jak by mělo být rozdělené" vs „jak je":** normalizace/segmentace/tokenizace/jazyková
> analýza jsou SLITY do jednoho kroku (UDPipe). To patří do Dílu VIII jako pozorování, ne do
> výkladu jako záměr. Merge zkratek (R.U.R.) je jediná ruční normalizace nad UDPipe.
> **Vazba na původní text (zadání 3.1):** ANO — token nese start/end offset, fakt nese (doc, sent);
> dohledání zpět k větě funguje. Popsat v kapitole o rozhraních fází.

---

## D. Slovník pojmů (seed pro Díl I)

VZOR (frame_sig = pole slotů kolem pivotu, šířka r) · WORD/SLOT/DÍRA · aktivace / světlo / glow
(idf váhy uzlů, vodivost hran) · dialogový stavový automat · assurance (jasno/nejasno/nehádej) ·
mount (#60 fragmentový graf — nahrání jen horkých souborů) · fakt (predikát + role-sloty +
provenience) · role díry (who/where/when/whom_what/state/how_much…) · koreference / díra
(pro-drop podmět) · echo-guard · glow-orders-ties (base rozhoduje, glow řadí remízu) · etalon /
gold · confident-wrong · honest-negative · carry_context (řetězení kontextu) · alias / identita ·
polární smyčka (ano/ne) · naučená brána (gate) · UDPipe 2 · CoNLL-U · lemma / upos / deprel /
NameType · ÚFAL · #60.

---

## E. Naměřená čísla (seed pro 4.8.6) — VÝHRADNĚ skutečná měření

- **Velký etalon (eval_large, gold_large.json, 145 otázek):** PASS 63/145 = 43 %; confident-wrong 30.
  Rozpad po kategoriích: honest-negative 7/7, copula 16/21, spatial 12/17, polar 9/14,
  temporal 9/25, authorship-who 6/20, relation 4/18, attribute 0/11, count 0/7,
  authorship-what 0/3, taxonomy 0/2. Formulační robustnost 8/14 skupin parafrází.
- **Malý etalon (eval_answers, gold_answers.json, 25 otázek):** 18/25 = 72 % (HIT 18, WRONG 2,
  CLARIFY 4, NO_QVZOR 1). Faktický (bez teologie) historicky ~64 %.
- **Naučená brána (gate_pilot, LOO-CV, 77 answer-mode případů):** přesnost 68.8 %; práh 0.30 →
  −13 confident-wrong / −3 správné (LOO); NASAZENÍ přeučuje (běh −19) → VYPNUTA.
- **NN router pilot (nn_pilot, LOO-CV, 20 rysů):** 75.9 % vs 17.2 % majorita.
- **Oracle routing-strop:** +4 PASS / −6 confident-wrong (routing minor páka).
- **Křivka velkého etalonu:** 33 → 40 (polar+extrakce) → 43 % (vztahy); confident-wrong 35 → 30.
- **Fakty:** 16028 + 24 bio + 114 vztahy nad 52 soubory.

> Ke každému číslu v knize doplnit: data, velikost sady, commit, datum, příkaz pro zopakování,
> zdroj (zadání 4.8.6). Rozpad po kategoriích = nejcennější (ukazuje slabiny).

---

## F. Otevřené body (seed pro Díl VIII, 4 kategorie)

| Bod | Kategorie |
|---|---|
| Node-based retrieval (entita=uzel s fakt-hranami) — obejde R.U.R./doménu i rozpad jmen | nedodělek / směr |
| Naučená brána (overfit na 76 vzorcích) | vědomé provizorium (vypnutá) |
| Relation zbytek (verb-vztahy, oblique, pronoun-genitiv) — narážejí na entitní identitu | nedodělek |
| R.U.R. / doména (tokenizace R/U/R → bible mount) | známé omezení (čeká na node-based) |
| Temporal 36 % (datumy v běžném textu mimo bio-závorku) | nedodělek |
| VZOR-síť do šířky r přes věty (uzavřená smyčka fact↔query↔answer) | nerozhodnuto / směr |
| Mnemos — zápis (uživatel učí fakty) | nedodělek |
| Bible plošné znečištění mountu | známé omezení |

---

## G. Dotazy na tým (nelze zodpovědět z kódu)

1. Jsou sim_*.py, phase1.py, roles.py, gen_variants.py, demo_table.py ŽIVÉ součásti, nebo mrtvé/experimentální? (kandidáti na úklid / známé omezení)
2. Je Ollama (externi-zdroje.html) skutečně zapojená, nebo jen zamýšlená?
3. Cílový čtenář a umístění výstupu — viz FÁZE-2 (návrh ke schválení).
