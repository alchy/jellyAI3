# Subsystémy Iris — společný půdorys (Chronos, Topos, Mnemos, …)

> Koncept dle zadání uživatele (2026-07-18): víme zhruba, co od subsystémů
> chceme — to ke kvalitnímu designu nestačí. Tento dokument definuje SPOLEČNÝ
> PŮDORYS, brány zapojení a plán refaktoru báze Iris i extraktu.

## 1. Teze

**Subsystém je doménový expert nad jednou osou reality** — Chronos nad časem,
Topos nad prostorem, Mnemos nad pamětí uživatele (Metron nad množstvím, Echo
nad kompozicí). Iris je DIRIGENT: vede dialog, drží jistotu (QueryAssurance)
a rozhoduje KARTAMI; subsystémy nikdy nemluví s uživatelem přímo — vracejí
strukturované výsledky, texty a rozhodnutí nesou karty.

Společné rysy (zadání): každý subsystém ukládá **JSON vzory** (rozpoznání /
extrakce / aktivace) a **JSON záznamy** (deník, připomínky, definice,
gazetteer). Kód subsystému je jen MECHANISMUS (parsing, aritmetika os,
kontejnment) — ZÁKON projektu platí i uvnitř subsystémů.

## 2. Anatomie subsystému — čtyři schopnosti

Každý subsystém implementuje týž port (cílový protokol refaktoru):

| Schopnost | Otázka | Chronos | Topos | Mnemos |
|---|---|---|---|---|
| **POZNÁNÍ** (patterns) | „Je tenhle výraz můj?" | „za chvíli", „21. června", „připomeň mi" | „v Praze", „kde", pohybová slovesa (putovat, dlít, přespat…) | tvary konstatování (1. os., l-příčestí…) |
| **KANONIZACE** | jazyk ↔ kanonický záznam | „za čtvrt hodiny" ↔ timestamp; „ráno" ↔ 07:00; 12:00 ↔ „dvanáct hodin" | „v Praze" ↔ uzel Praha; hierarchie Praha ⊂ Čechy | výrok ↔ fakt s časovou kotvou |
| **AKTIVACE** (reflektor) | „Které uzly rozsvítit?" | interval → časové uzly uvnitř | místo → uzel + VŠE UVNITŘ (kontejnment strom) s útlumem | user-fakty tématu |
| **ZÁZNAMY** (records) | co si pamatuje | reminders.jsonl; naučené definice („ráno = 8:00") | gazetteer (kontejnment), naučená místa | memory.jsonl (deník) |

```python
class Subsystem(Protocol):          # jellyai/iris/subsystems/port.py
    name: str                                        # "chronos", "topos", …
    def claim(tokens, now, context) -> Claim | None  # brána Q: nárok na tokeny
    def extract(sentence, entities) -> list[Fact]    # brána E: extrakce
    def illuminate(claim, graph, field) -> None      # brána A: reflektor
    # vzory: subsystems/<name>/patterns/<jazyk>/*.json (deck jako u Iris)
    # záznamy: data/<name>.jsonl (schéma definuje subsystém)
```

## 3. Tři brány zapojení (kde subsystém potkává tok dat)

```
KORPUS ──► brána E (extrakce) ──► GRAF ◄── brána E ── DIALOG (Mnemos zápis)
                                    ▲
OTÁZKA ──► brána Q (nárok) ──► pseudo-QL pattern + OMEZENÍ (constraint)
                                    │
                          brána A (reflektor) ──► ActivationField ──► ranking
```

- **Brána E (extrakce, build i dialog).** Pipeline extraktu se REFAKTORUJE
  z pevných producentů na REGISTR rozpoznávačů: subsystém dostane větu
  a entity, vrátí fakty/normalizované účastníky. Chronos už to ad hoc dělá
  (parse_date, zanořená data); Topos přidá kontejnment („Betlém v Judsku"
  → uvnitř(Betlém, Judsko)) a normalizaci loc rolí. Uživatelský a korpusový
  graf jsou PRINCIPIÁLNĚ STEJNÉ — dialogová cesta (Mnemos) jen přidává
  faktům timestamp dialogu. Extrakční vzory = JSON subsystému.
- **Brána Q (dotaz).** Parser pseudo-QL nabídne tokeny otázky subsystémům;
  kdo pozná své, NÁROKUJE je (claim) a vrátí OMEZENÍ: Chronos interval
  („letos"), Topos místo/díru loc („kde", „v Čechách") + slovesnou třídu
  (putovat ≈ cestovat ≈ pobývat ≈ dlít ≈ přespat — JSON třída, expanze
  dotazu), Metron počet („kolikrát"). Nárokované tokeny NEvstupují do
  rozlišení entit (dnes ad hoc: časová slova filtrujeme — zobecnit).
- **Brána A (aktivace).** Omezení se promění v osvětlení: Chronos rozsvítí
  časové uzly intervalu (existuje: `_warm_interval`), Topos rozsvítí místo
  a jeho kontejnment podstrom s útlumem (analogie intervalu!), Mnemos svítí
  user-fakty. **Reflektor je jediný způsob, jak subsystém ovlivňuje
  odpověď** — ranking answereru je jediný konzument jasu. Tvrdá omezení
  (interval jako filtr, backlog #10) jsou druhý stupeň téže brány.

Symetrie os: **Chronos = intervaly na časové ose, Topos = kontejnment strom
na ose prostoru.** `TimeInterval.contains_date` ↔ `Topos.contains_place`.
Oba odpovídají na „spadá X dovnitř Y?" — a oba tím rozsvěcují graf.

## 4. Chronos v2 — reminder jako plnohodnotný subsystém

Stav v1 (commit 294a5f2): resolve_due (ofsety, denní čas, datum, předstih),
sklad reminders.jsonl, vlastní VLÁKNO hodin (ChronosTicker), push do konzole
webu (REST event terminal_write), karty reminder-set/when/due.

**v2 dle zadání:**

1. **Výchozí termín místo otázky.** Bez času se nečeká: karta
   `reminder-default` naplánuje „za čtvrt hodiny" a řekne: „Nespecifikoval
   jsi kdy — připomenu za čtvrt hodiny. Chceš-li jindy, řekni kdy."
   Navazující „až zítra ráno" PŘEPLÁNUJE poslední připomínku (stav
   pending-reschedule). Obě chování (ptát se × default) jsou KARTY —
   výběr nese balíček, ne kód.
2. **Neurčité denní časy jako znalost.** `day_parts` v cs.json dává
   defaulty (ráno 07:00, dopoledne 10:00, poledne 12:00, odpoledne 15:00,
   večer 19:00, noc 22:00); uživatel je UČÍ dialogem: „ráno je v 8" →
   záznam v definicích Chronosu (Mnemos je brána zápisu, cíl záznamu je
   subsystém). Pořadí čtení: záznamy uživatele > jazykové defaulty —
   týž princip jako uživatelský graf > korpus.
3. **Kanály.** Záznam připomínky nese `channel`; registr kanálů ve službě:
   `console` (dnešek: konzole služby + push do webu), `window` (statické
   okno „Reminder" ve viewBase — VISÍ, dokud ho uživatel nezavře; closable
   mechanismus existuje), budoucí `alarm-audio`, `email`, `whatsapp` —
   adaptéry se REGISTRUJÍ, jádro se nemění. Fráze smí kanál volit kartou
   („vzbuď mě" → alarm-audio; akce karty `{"channel": "alarm"}`).
4. **Rozpoznání kartou.** Fráze (připomeň mi, vzbuď mě, upozorni mě,
   houkni na mě…) zůstávají jazyková data; VÝBĚR chování se přesune z kódu
   do karty `utterance.reminder` (rysy: reminder_phrase, due_present) —
   dnes částečně v automatu, v2 dočistí.

## 5. Topos v1 — návrh

1. **Data:** kontejnment jako modelový primitiv `uvnitř(Praha, Čechy)` —
   extrakcí z korpusu („Praha leží v Čechách", „Betlém v Judsku": nmod/obl
   vzory) + kurátorský GAZETTEER (JSON záznam Topos; Čechy ⊂ Česko ⊂
   Evropa) + učení dialogem („Kaudland je v Africe" → záznam).
2. **Brána Q:** tázací „kde" (existuje) + nárok na „v(e) MÍSTO" fráze;
   **slovesná třída pohybu/pobytu** (putovat, cestovat, pobývat, dlít,
   přespat, jít, přijet…) jako JSON — „Kde putoval Ježíš?" expanduje přes
   celou třídu (zobecnění predicate_synonyms na doménové třídy).
3. **Brána A:** „v Čechách" rozsvítí Čechy + kontejnment podstrom
   s útlumem (falloff jako spread); loc-hrany faktů pohybové třídy dostanou
   preferenci. Odpověď „Kde putoval Ježíš?" = výčet loc účastníků faktů
   třídy pohybu (glow řadí).
4. **Etalon:** „Kde putoval Ježíš?", „Pršelo v Čechách?" (kontejnment:
   fakt s loc Praha ⊂ Čechy → Ano) — druhý zavře uživatelův dávný příklad.

## 6. Mnemos — brána zápisu pro všechny subsystémy

Mnemos přestává být jen „deník epizod": je to JEDINÁ CESTA, kterou dialog
ZAPISUJE. Cíl zápisu určí karta: epizoda/pozorování/připsaný fakt → graf
(dnešek); definice času („ráno je v 8") → záznamy Chronos; definice místa
(„Kaudland je v Africe") → záznamy Topos; připomínka → sklad Chronos.
Subsystémy registrují SCHÉMATA svých záznamů; Mnemos drží mechaniku
(persistence, replay, provenience „zdroj = uživatel", timestamp dialogu).

## 7. Plán refaktoru (fáze, každá měřená)

| Fáze | Obsah | Riziko |
|---|---|---|
| **S0** | ✅ HOTOVO — balíček `jellyai/iris/subsystems/` s portem, Chronos/Mnemos přesunuty beze změny chování | nízké |
| **S1** | ✅ HOTOVO (jádro) — day_parts, „za chvíli/čtvrt hodiny", reminder-default + přeplánování, okno Reminder, registr kanálů; zbývá: definice uživatele („ráno je v 8" → S5), kanály alarm/email | střední (UI) |
| **S2** | ✅ HOTOVO (jádro) — tvrdý interval (#10): time_filter v answereru, explicitní datum jako primitivum, „příští týden"; dotaz na plán (reminder-list). Zbývá: formální `claim()` v parseru, správa plánu (BACKLOG #27) | střední |
| **S3** | Topos v1 (§5) + etalonové řádky | střední |
| **S4** | Brána E: registr rozpoznávačů v extraktu (refaktor extract.py na pipeline producentů; parse_date → Chronos.extract) | vyšší — jistí etalon+coverage |
| **S5** | Mnemos brána zápisu pro subsystémy (§6); učení „ráno je v 8" E2E | střední |

Zásady refaktoru: žádná zpětná kompatibilita (prototyp), každá fáze končí
zelenými benchmarky, chování VŽDY v kartách/JSON — kód jen mechanismy.

## 8. Co se NEmění

Jeden graf (korpus + uživatel, user-fakty s timestampem); ActivationField
jako jediné médium vlivu na ranking; karty Iris jako jediný hlas dialogu;
jazyk jako data; benchmark-gated vývoj. Subsystémy tohle všechno POUŽÍVAJÍ,
nic z toho neobcházejí.
