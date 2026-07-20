# Jednotný dispatch (#51): výroková polovina — graf jako jediný směrovač

> Kontext (brainstorm 2026-07-20 pozdě večer): po fázi D (#57) směruje
> otázkový graf přímé experty a dotazové karty; mimo něj zbývá výroková
> typologie, příkazy a stavové tahy — ~38 z 50 dialogových tahů (nález 3
> experimentu). Cíl: graf jako JEDINÝ dispatch celého vstupu, ruční
> pořadí větví v `_turn()` zmizí. Rozhodnutí brainstormu (user):
> **jeden kompilát s rodinami uzlů**, hranice dotaz × výrok jako **rys
> rodiny v datech**, **příkazové karty + claims tokenů**, metodologie
> **P1 — shadow-first po rodinách** (velký třesk vědomě zavržen už
> u S2, dialog 2026-07-19).

---

## 1. Model: jeden kompilát, pět rodin

| Rodina | Uzly z | Worker | Díra | Poznámka |
|---|---|---|---|---|
| `otazka` | dotazové karty (existuje) | graf | ano | beze změny |
| `worker` | claims registr (existuje) | metron/chronos/iris | — | beze změny |
| `clarify` | clarify karty (existuje) | dialog | — | odvozené hrany (E4) |
| **`vyrok`** | karty `utterance.statement` | **brána E** (zápis) | ne | NOVÁ rodina |
| **`prikaz`** | NOVÉ příkazové karty | mnemos/chronos | ne | NOVÁ rodina |

T6 disciplína se drží uvnitř kompilátu: výrokové uzly nemají díry
a jejich „odpovědí" je zápis (`parse_clauses` + `_memorize` —
mechanismus brány E se NEMĚNÍ, graf přebírá jen VÝBĚR, kdo tah
odbaví).

**Hranice dotaz × výrok:** tahu se spočítá povrchový rys `otaznik`
(a případné další interpunkční rysy) a rodiny ho deklarují na kartách
EXISTUJÍCÍM mechanismem `requires`/`forbids` — žádný nový kód
rozhodování, jazykový fakt je v datech. Budoucí STT vstup bez
interpunkce (#42) si karty přenastaví, mechanismus zůstane.

## 2. Kompilace

- **`vyrok` uzly:** z dnešních karet `utterance.statement`. Vzorové
  karty (mají `pattern`) se osvětlují matcherem jako otázky; RYSOVÉ
  karty (requires/forbids nad rysy first_person/copula/l_verb…) se
  osvětlují týmiž rysy, kterými je dnes vybírá `deck.best`
  v `parse_statement` — kompilace nezavádí druhé rozhodování, jen
  zviditelňuje dnešní (přesně jako u dotazové poloviny).
- **`prikaz` uzly:** nové karty (event `utterance.command`) se vzorem
  z literálových zkratek nad DNEŠNÍMI frázovými tabulkami cs.json:
  `%{ZAPAMATUJ}` ← `memorize_phrases`, `%{PRIPOMEN}` ←
  `reminder_phrases`, `%{ZAPOMEN}`, `%{POSUN}`/`%{ZRUS}` ←
  `plan_move_words`/`plan_cancel_words`, `%{POSLI}` (#54); volný
  zbytek `*`; `worker` atribut karty říká, kdo odbaví. Tabulky
  zůstávají jediným zdrojem frází (zkratka = pohled, ne kopie).
- **Claims tokenů** (chronos `claim_words`, dativ, čísla) dál
  DEKORUJÍ — nesoutěží (T3); tvar příkazu nese karta, nárok expert
  (vzor #54 „Pošli zítra mail Marci").
- Workers a clarify beze změny (E2/E4).

## 3. Stavové tahy = pozice, ne dispatch

Pending identity (#43), pending pick (overflow volba), rozpracovaná
připomínka = dialog STOJÍ v uzlu grafu a tah je krok po hraně
`navrat` (`DialogPosition`, měřeno rovinou „stav"). Dispatch dostává
jen volné tahy. Složitá logika převzetí (takeover rozpracované
připomínky novým příkazem) zůstává mechanismem uvnitř kroku —
hrany nenesou podmínky (zákaz ATN trvá).

## 4. Měření a migrace (P1 — shadow-first po rodinách)

1. **Kompilace + rovina 5 harnessu:** `_actual_route`
   v `benchmark/run_qgraph.py` se naučí číst výrokové a příkazové
   cesty z metadat odpovědi (kind, memorized, karty reminder-*/…);
   shadow měří shodu osvětlení se skutečnou větví na dnešních ~38
   „mimo rozsah" tazích. Nic se nepřepíná.
2. **Přepnutí VÝROKŮ:** výběr výrokové cesty osvětlením (brána E
   mechanismus nezměněn); gate = shadow 100 % pro rodinu + bitová
   parita všech 5 benchmarků.
3. **Přepnutí PŘÍKAZŮ:** po jednom druhu (memorize → forget →
   reminder/plan → send), každý s paritou; ruční větev mizí ihned po
   přepnutí svého druhu.
4. **Zbytek:** recall („Co jsem ti řekl včera?"), focus-shift
   („v kontextu Bible") — malé rodiny týmž postupem.
5. **Smazání ručního pořadí** v `_turn()` — teprve když je dispatch
   všech rodin grafem; `_turn` zbude: hodiny → pozice (stavové
   kroky) → osvětlení → worker/brána.

Kritérium přijetí každého kroku (rámec spec dotažení §3): parita
drží a aspoň jeden nový řádek/rovina zezelená; co neprojde, se
nepřijímá a zůstane zapsaným nálezem.

## 5. Co se vědomě nedělá

- brána E (mechanika zápisu, klasifikace druhů výroků) — beze změny;
- pending mechanismy (identity/pick/reminder) — beze změny, jen se
  stanou kroky pozice;
- odpovědní graf — dál zaparkován;
- žádné nové prahy/heuristiky — rozhodují karty (zákon 1);
- vícejazyčnost a STT — jen připravenost (rysy v datech), ne cíl.

## 6. Otevřené otázky (vyřeší plán/měření)

- Výpočet výrokových rysů pro osvětlení: sdílet přesně tentýž kód
  s `parse_statement` (jedna funkce rysů tahu), aby parita byla
  konstrukční, ne testovaná.
- Telemetrie příkazů: po přepnutí pokryje karty příkazů automaticky
  (krmivo pro triage #38) — ověřit formát.
- Priorita mezi rodinami při vzácném souběhu (výrok se vzorem ×
  otázka bez otazníku): měří harness (počítadlo remíz běží dál).
