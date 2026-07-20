# Fragmentový graf — kontext souborů jako vrstva aktivace (#60)

Koncept user (2026-07-20 noc, intuitivní zadání), rozvedení. Podnět:
meziklauzulová inference (#59) má zohlednit VZDÁLENOST mezi klauzulemi
→ „může to být další typ grafu".

## Zadání (user, parafráze)

Každá část textu — soubor — je fragment s podobným žánrem; nese
specifická slova, v nichž jsou silnější vazby. Kapitoly/dokumenty
umístěné v souborech mají bližší vazby. Když se v grafu rozsvítí
konkrétní slova, mají se rozsvítit i datové soubory, které je nesou
s větší mírou — a ty zpětně přihřát své související vazby. Zacílit
správné soubory frekvenční analýzou podobných a rozsvícených slov.
Cíl: mluví-li se o Čapkovi jako o spisovateli, svítí soubory Karla
Čapka (ošetřit kontext). Při tisících souborů má mít každý soubor
vlastní index (slovo → váha oproti jiným), aby se do paměti/grafu
nahrávaly nejbližší soubory podle indexu.

## Co UŽ stojí (nestavět znovu — DRY)

| Mechanismus | Kde | Co dělá |
|---|---|---|
| `graph.doc_links` | graph.py:115, build :401 | graf DOKUMENTŮ: doc → {doc: síla} podle sdílených entit (13 souborů, plná matice) |
| `source_context` + `_warm_sources` | graph_answerer.py:1463 | aktivační pole nad soubory: zdroje ODPOVĚDI se rozsvítí a vyzáří po doc_links (top-8, spread_falloff), pak pohasínají |
| `_source_bonus` | graph_answerer.py:1476 | fakt z horkého souboru dostává bonus při matchi — „dvě Marie z různých korpusů" už dnes řeší provenience, ne rozpad uzlu |
| `fact.source` | všude | provenience faktu na úrovni dokumentu |
| anotace klíčované `(doc, idx_věty)` | annotations.pkl | jemná textová POZICE existuje v datech (podklad metriky vzdálenosti) |
| `Retriever` TF-IDF | retriever.py | frekvenční analýza (tf, idf, kosinus) už v projektu žije — extraktivní fallback |
| okno Aktivní dokumenty | docs_window, #28 | vizualizace jasu souborů (přejmenování na „Paměť" čeká) |

## Mezera (delta konceptu)

1. **Slova rozhovoru soubory NErozsvěcují** — svítí až zdroje hotové
   odpovědi (`_warm_sources` po faktu). Pro ZACÍLENÍ před rozřešením
   tématu je to pozdě: kontext má pomáhat už výběru tématu/homonyma.
2. **Hrana slovo↔soubor s relativní vahou není reifikovaná** —
   doc_links váží jen sdílené entity MEZI soubory; „tento soubor nese
   toto slovo silněji než ostatní" (tf-idf) graf nezná.
3. **Fragmenty nemají hierarchii** (soubor → kapitola → odstavec) —
   chybí textová metrika vzdálenosti pro #59 i provenienční okno pro
   #8 bod 1 (instance per odstavec!).
4. **Vše v RAM** — lazy-load podle indexů neexistuje (u 13 souborů
   nevadí; formát indexu ale navrhnout shardovatelně hned).

## Návrh

### Pojmy

- **FRAGMENT** = uzel provenienční jednotky; strom: korpus → soubor →
  kapitola → odstavec. Materializovat zprvu soubor + odstavec
  (kapitoly až budou v datech značené).
- **INDEX FRAGMENTU**: slovo → relativní váha (tf × idf, týž
  mechanismus jako Retriever — sdílet kód, ne kopírovat). Uložení per
  soubor (`data/index/<doc>.pkl`, shardovatelné) + globální idf
  tabulka. Malý korpus → práh min. výskytů (vzor `_MIN_VOTES`).
- **HRANY**: `nese(fragment, uzel, w=tfidf)`; `vedle(fragment,
  fragment)` = sousedství v textu (odstavec n↔n+1, kapitoly v témž
  souboru) — POZOR, jiná sémantika než doc_links (tematická blízkost
  sdílenými entitami); obě hrany vedle sebe, nemíchat váhy.

### Aktivace — dva směry, tlumené

- **DOPŘEDU (nové)**: každý tah → slova otázky + žhavé uzly těžiště
  rozsvěcují fragmenty: skóre(f) = Σ teplo(slovo) × w_nese(f, slovo).
  Fragmentové pole = zobecněný `source_context` (ActivationField).
- **ZPĚT (zobecnit dnešní princip, NE plošné zahřívání)**: horký
  fragment NEzahřívá slova plošně — runaway a rozmazání domény (kód
  to už ví: „vyzařování po doc_links by doménu rozmazalo",
  graph_answerer.py:316). Zpětný tok = BONUS při vážení evidence:
  `_source_bonus` rozšířit z faktů i na KANDIDÁTY rozřešení tématu
  a na patra identity; slabé vyzařování po `vedle`/doc_links zůstává
  jak je (_warm_sources).
- **Užití**: (a) patro evidence v `_resolve_topic` — „Čapek" se při
  spisovatelském kontextu rozřeší na Karla (a smíšené holé uzly typu
  „Josef" dostanou konečně oporu — přesně to, co při #8 bodu 2 lhalo
  otisku); (b) komponenta QueryAssurance — kandidát bez podpory
  horkých fragmentů = nejistota → clarifikace (dialog > figly, prior
  není veto!); (c) okno Paměť (#28) ukazuje fragmentové teplo.

### Vzdálenost pro #59

Textová metrika d: 0 táž klauzule · 1 sousední klauzule souvětí ·
2 sousední věta · 3 týž odstavec · 4 táž kapitola · 5 týž soubor ·
6 soused po doc_links. Odvozený fakt nese confidence = falloff^d;
NIKDY nepřebije přímý fakt (defeasible, jako #41); provenience
„odvozeno z (doc, věta_a × věta_b)" — čeká na #39. Úrovně 0–2 dává
extrakce z anotací `(doc, idx)` už dnes, 3–6 dá fragmentový strom.
#59a (účelová klauzule, gap Jordán) = d=1, nejvyšší jistota — proto
jde první a fragmentový graf NEBLOKUJE.

### Škálování (park, až tisíce souborů)

Graf v paměti drží jen HORKÉ fragmenty; studený soubor = jen index
na disku. Zacílení: skóre z indexů (Σ teplo(slovo) × idf × tf) →
mount subgrafu (build je per-anotace už dnes; unmount = odebrat
fakty se `source=doc` — umožní provenience #39). Práh s hysterezí
(žádné flapování). NEstavět dnes — jen formát indexu per soubor
od F1, aby shardování uneslo.

## Fáze (každá měřená; parita + ≥1 nový řádek, jinak nepřijmout)

- **F0 MĚŘENÍ**: frekvenční profily 13 souborů; sanity: „spisovatel,
  drama, hra" → wiki_čapkovské, „faraón, Hospodin" → bible_*; čísla
  sem do spec; rozhodnout prahy (min. výskyty, top-K).
- **F1 STAVBA**: fragmentové uzly (soubor + odstavec) + indexy per
  soubor + hrany nese/vedle v buildu; okno Paměť čte fragmentové
  teplo. Odpovědní chování BEZE ZMĚNY (parita celé baterie).
- **F2 ZACÍLENÍ**: dopředné rozsvěcení + fragmentový prior jako
  patro evidence/komponenta assurance; nový benchmark řádek:
  disambiguace Čapků dialogem (hovor o malíři → „Co napsal Čapek?"
  ≠ hovor o dramatech → táž otázka), + focus rovina.
- **F3 = #59a**: metrika d pro účelovou klauzuli (gap Jordán) —
  viz BACKLOG #59; fragmentový strom dodá úrovně 3+.
- **F4 LAZY-LOAD**: park do doby tisíců souborů.

## Pasti

1. Zpětný tok plošným zahříváním slov = runaway → zpět jen bonusem
   při vážení evidence (viz komentář graph_answerer.py:316).
2. Žánrová blízkost ≠ tematická: Bible knihy sdílejí žánr, čapkovské
   wiki téma — hrany `vedle` × `doc_links` držet oddělené.
3. Fragmentový prior je PATRO, ne veto — přímá evidence otázky vždy
   přebíjí; konflikt → clarifikace (zákon 2).
4. idf nad malým korpusem je vratké — prahy (vzor _MIN_VOTES).
5. Determinismus: skóre fragmentů vždy řadit (past 19).

## Vztahy

#59 (vzdálenost — F3), #8 bod 1 (fragment = provenienční okno
instance; smíšené holé uzly jsou hlavní motivace F2), #28 (okno
Paměť), #13 (sharpener — fragmentové hrany jako další typ pro
vyzařování), #41 (útlum po vzdálenosti = týž princip), #39/#47
(provenience odvozených faktů, event log), Retriever (sdílený tf-idf
mechanismus).
