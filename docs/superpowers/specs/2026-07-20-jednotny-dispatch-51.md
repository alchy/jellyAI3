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

---

## Empirie (2026-07-20 pozdě večer, větev jednotny-dispatch) — PŘIJATO

**Fáze 0 (postřehy):** instance_lit ke grafu (4.1), jeden sdílený deck
(1.2), recognize claimu vrací výsledek (1.3+1.4 — konec dvojího
výpočtu), harness čte jména worker uzlů z registru (4.5). Vše čistá
parita.

**Fáze 1:** `turn_features` (povrch + výrokové + příkazové rysy jednou
funkcí), výrokové karty `forbids: ["otaznik"]` (hranice v datech),
rodina `vyrok` v kompilátu (worker brána E) se SDÍLENÝM jádrem
těsnosti `trigger_specificity` (postřeh 1.1 — žádná další implementace
výběru). Shadow rovina výroků: **15/15 (100 %)**; volba identity
podmětu (#43) správně přesunuta do STAVOVÉ roviny (4/4) — výroková
clarify je pozice, ne dispatch (výrok → statement.* hrany zrcadlí E4).

**Fáze 2:** konstatování řídí vítěz osvětlení (uzel vyrok) — dialog
benchmark bitově beze změny.

**Fáze 3:** rodina `prikaz` = RYSOVÉ karty cmd-* (KOREKCE SPECU:
frázové tabulky jsou substringové/tokenové — literálové vzory by
měnily sémantiku; rys zrcadlí sémantiku PŘÍSLUŠNÉHO handleru: forget
TOKENOVĚ po slovech, „Nezapomeň…“ je správně memorize). Priority
karet nesou dnešní pořadí větví (forget 19 → focus 14, recall 13).
KOREKCE PLÁNU: separátní shadow rovina příkazů nahrazena BITOVOU
paritou dialogu po druzích — instrumentace mizejících větví by byla
vyhozená práce; scénáře kryjí všechny druhy. Dispatch STRIKTNÍ:
selhavší handler vítěze = poctivý propad (žádné zkoušení dalších
příkazů) — parita na celém benchmarku držela napoprvé.

**Fáze 4:** recall dispatchem (cmd-recall smí otazník, prioritou
přebíjí dotazové karty). VĚDOMĚ ODLOŽENO: kosmetický rozklad _turn
na fáze-metody — dispatch už je v datech (pořadí větví sémantiku
nenese), hlubší dělení má smysl až s TurnResult / odpovědním grafem.
Mimo rozsah zůstal dotaz na plán („Mám naplánované úkoly?" — brána Q
Chronosu v dotazové cestě) a stavové mechanismy (pozice).

**Čísla:** 580 testů; etalon 33/33, focus 12/12, dialog 45/45 (GAP
6/0), zápis 34/34; harness [tiers i weights]: dialog 11/11, výroky
15/15, stav 4/4, dekorace 45/45, etalon 45/45 — vše 100 %, 0 remíz.
Graf: 5 rodin uzlů (otazka/worker/clarify/vyrok/prikaz), ruční řetěz
příkazů a výroků v `_turn()` nahrazen `_command_turn` + osvětlením.
