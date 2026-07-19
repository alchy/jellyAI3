# Vzorová gramatika — jeden jazyk pro karty i analýzu vět (návrh)

> Zadání (user, 2026-07-19): „Karty rozhodují dialogové akty, gramatika je
> Python — nemělo by smysl, aby karty nebo Python používaly rekurzi a
> podmíněné přepínání stavovým automatem podle konstrukce věty na základě
> identifikace druhů slov? Jak zjednodušit definici karet i Python gramatiky?"
>
> Odpověď v jedné větě: **ANO stavovému automatu, ale stavy a přechody musí
> být DATA (vzory na kartách), vykonavatel mechanismus; rekurzi jen JEDNU —
> klauzulovou; a celé to stojí na JEDNOM sdíleném určovači druhů slov,
> který dnes existuje čtyřikrát po kouskách.**

---

## 1. Diagnóza: co je dnes duplikované a co chybí

Projekt má DVĚ analýzy vět, které dělají totéž jiným kódem:

| | Zápis (Mnemos `parse_statement`) | Dotaz (`build_query`) |
|---|---|---|
| identifikace slov | stop-listy + koncovky + katalogy | `_verb_match`/`_exact_predicate` + interrogatives + spanové dělení |
| řízení | ploché rysy → karta (requires/forbids) | ~100 podmínek Pythonu (poziční šablony) |
| výstup | {kind, predicate, objects, places, time} | Pattern {predicate, known, hole_role/type, filtry} |

Oba výstupy jsou **týž rámec**: typované spany + predikát + role. A obě
strany už dnes NEZADRŽITELNĚ tečou do dat — `interrogatives` mapuje tázací
slovo → (role díry, typ díry), `predicate_catalog`, `event_verb_forms`,
`question_words`, `particle_words`, `finite_verb_forms`… 45 tabulek nese
fragmenty gramatiky. Co v datech NENÍ, je **sekvence**: co smí následovat
po čem. Ta je zamčená v Pythonu (query.py) a v plochých rysech karet — proto
každý nový větný tvar (#44 „Kdo bydlí v Petrovicích?") znamená psát kód,
a proto homografy (#45 „jí", „byt") nejdou vyřešit: plochý rys nevidí
POZICI slova ve stavbě věty.

## 2. Princip: tři vrstvy, každá svým materiálem

```
   tokeny
     │
 [L] LEXER — jeden určovač DRUHŮ slov          mechanismus + TABULKY (existují)
     │        TaggedToken: {tvar, množina tříd}  (slovo smí mít VÍC tříd!)
     ▼
 [V] VZORY — sekvence tříd na KARTÁCH           DATA (nový klíč „pattern")
     │        vykonavatel = malý konečný automat  mechanismus (~100 řádků, 1×)
     ▼
 [S] VÝBĚR — nejtěsnější vzor vyhrává           deck.best (UŽ EXISTUJE)
```

**[L] Lexer** sjednocuje dnešní roztroušené určování: OTAZ (z
`interrogatives` — nese rovnou roli+typ díry), SPONA, SLOVESO (koncovky +
`finite_verb_forms` + l-tvar + katalog), PŘEDL (v/na/do/pro…), ČÁSTICE,
ČAS (Chronos primitiva), JMÉNO (kapitalizace/known-person), ENTITA (span
grafu), EMAIL, ZÁJMENO. Klíčové: token nese **množinu hypotéz** —
„byt" = {SPONA?, SUBSTANTIVUM}, „jí" = {SLOVESO?, ZÁJMENO}. Lexer
NEROZHODUJE dvojznačnost; jen ji poctivě přizná.

**[V] Vzor na kartě** je REGULÁRNÍ sekvence (žádné vnořování, žádné
podmínky, žádné proměnné — jen třídy, literály, volitelnost):

```json
{"name": "q-kdo-sloveso-misto",
 "trigger": {"event": "utterance", "pattern":
   ["OTAZ:kdo", "SLOVESO", "PŘEDL:v|ve|na", "MÍSTO"]},
 "action": {"query": {"hole_from": "$1", "predicate": "$2",
                       "place_filter": "$4"}}}
```

Dvojznačnost řeší VZOR, ne pravidlo: „Roník jí granule." matchne
`[JMÉNO, SLOVESO, SUBSTANTIVUM]` jen tak, že slot SLOVESO obsadí „jí" —
protože nic jiného ho obsadit neumí a věta bez predikátu nematchne nic.
Tím padá #45 bez univerzálního algoritmu: vzor je enumerace (zákon:
rozšiřovat, ne univerzalizovat), jen konečně na SPRÁVNÉ úrovni — na
stavbě věty, ne na izolovaném tvaru.

**[S] Výběr**: víc vzorů matchne → vyhrává nejtěsnější/prioritní — TÝŽ
mechanismus `deck.best`, který dnes vybírá dialogové karty. Tím se S2
uzavírá doslova: dialogové akty i gramatika jsou karty v jednom decku,
vybírané jedním výběrem, měřené jednou telemetrií.

## 3. Rekurze: ano, ale jen KLAUZULOVÁ (jedna úroveň)

Plná rekurzivní gramatika (ATN, Woods 1970 — přesně „rekurzivní stavový
automat s registry") historicky zemřela na neudržovatelnost: síť přechodů
s podmínkami je program, jen hůř čitelný. To je táž past jako „interpret
v JSONu". Projekt ji NEPOTŘEBUJE: věty dialogu jsou krátké a selhání
(#45 souvětí, vsuvka) mají jediný společný tvar — **víc klauzulí v jedné
větě**. Stačí tedy:

1. **Klauzulátor** (mechanismus): rozdělí větu na klauzule podle čárek a
   spojek (tabulka spojek — data). „Roník jí stravu, má však rád i maso."
   → [„Roník jí stravu", „má však rád i maso"].
2. Každá klauzule jde do TÉHOŽ automatu vzorů (to je celá „rekurze").
3. Skládání je mechanismus s malou tabulkou: klauzule bez podmětu dědí
   podmět předchozí; vsuvka „to co X V je Y" = vzor s klauzulovým slotem
   `["TO", "CO", KLAUZULE, "JE", SUBSTANTIVUM]` → definiční fakt.

Hloubka 1 pokryje vše, co runtime kdy potkal. Hlubší vnoření ať skončí
poctivým terminálem — to je lepší než zmršený zápis.

## 4. Co tím zjednodušíme (a o kolik)

- **query.py**: ~100 podmínek pozičních šablon → lexer + vykonavatel
  vzorů; šablony samy = řádky v kartách. Nový tázací tvar (#44 celá
  rodina: kdo-v-místě, „Kdo další", „Čím krmí X Y?") = nová karta,
  žádný Python.
- **Karty konstatování**: requires/forbids rysy (first_person, l_verb…)
  se stanou čitelným vzorem (`["JMÉNO?", "PRVNÍ_OSOBA", "L_SLOVESO", …]`);
  rysový mezikrok zmizí. `question_words` veto z 2026-07-19 přestane být
  speciální kontrola — plyne z toho, že výrokové vzory prostě nezačínají
  třídou OTAZ.
- **Write/query se potkají konstrukcí**: jeden lexer → predikát zapsaný
  = predikát dotazovaný (slib V4 brány, doručený inkrementálně ve V3).
- **Smaže se**: čtvery stop-list průchody, `_l_form`/`_finite_verb`
  duplikáty vůči query straně, ad-hoc dělení spanů.

## 5. Mapování na otevřené gapy

| Gap | Co ho zavře |
|---|---|
| #44 kdo-v-místě, čím, kdo-další | vzorové karty dotazů (fáze 2) |
| #45 „jí"/„sní" krátká slovesa | slot SLOVESO ve vzoru (jediný kandidát) |
| #45 „byt"≡„být" | hypotézová třída + vzor `[X, SPONA, Y]` vyžaduje sponu UPROSTŘED; koncové „byt" za adjektivem sponou být nemůže |
| #45 souvětí, vsuvka | klauzulátor (fáze 4) |
| #33 typ osoba/pojem | třída JMÉNO/ENTITA z lexeru rovnou do rolí faktu |
| #32 bezdiakritika | lexer = jediné místo, kam patří restore-katalogy |

## 6. Migrace (každá fáze měřená — benchmarky jsou na to stavěné)

1. **Lexer extrakcí** (žádná změna chování): dnešní kontroly obou cest se
   přestěhují do `classify(tokens) → TaggedToken[]`. Guard: 494 testů +
   všech 5 benchmarků beze změny.
2. **Vykonavatel vzorů + dotazy**: interrogatives → vzorové karty; šablony
   query.py po jedné, parity gate `run_etalon --mode templates` (osvědčený
   postup z pseudo-QL arcu). Tady se zavře #44.
3. **Výrokové karty dostanou `pattern`** (requires/forbids zůstává jako
   fallback — karta bez vzoru se chová postaru). Guard: zápisový etalon
   34/34 + dialog 27/27.
4. **Klauzulátor** → #45 souvětí/vsuvka gap řádky GAP-FIXED.
5. **Úklid**: smazat vyprázdněné cesty (zákon: žádná zpětná kompatibilita).

## 7. Zákazy (aby z toho nevznikl interpret v JSONu)

- Vzor je REGULÁRNÍ: třídy, literály, volitelnost, klauzulový slot.
  ŽÁDNÉ podmínky, proměnné, aritmetika, vnořené vzory.
- Ranking vzorů POUZE přes deck.best — žádný druhý výběrový mechanismus.
- Dvojznačnost, kterou vzory nerozhodnou, končí poctivým dialogem
  (clarify karta), ne heuristikou v kódu.
- Katalogy výjimek zůstávají v lexeru (datech) — vzory je nedublují.

## 8. Vztah k jellyV4

Toto JE brána [1] z `docs/proposed-architecture/jellyV4.md`, stavěná
inkrementálně uvnitř V3: lexer = jádro brány (zatím bez ÚFAL kontextu),
vzorové karty = interpret [2] nad typovanými tokeny. Kdyby V4 přestavba
přišla, obě vrstvy přežijí beze změny záměru; kdyby nepřišla, V3 z nich
má užitek okamžitě.
