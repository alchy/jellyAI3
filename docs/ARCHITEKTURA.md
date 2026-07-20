# Architektura jellyAI3 — od věty k faktu a zpět

*Výkladový průvodce systémem (stav 2026-07-20). Praktické doplňky:
HANDOVER.md (jak pracovat, pasti), BACKLOG.md (co dál),
`superpowers/specs/` (návrhové dokumenty s měřeními).*

---

## Úvod: co jellyAI3 je a čemu věří

jellyAI3 je česky mluvící znalostní systém postavený na **symbolickém
zpracování jazyka**: z textů (korpus) a z dialogu s uživatelem staví
**reifikovaný faktový graf** a nad ním odpovídá, pamatuje si, plánuje
a učí se. Žádná neuronová magie uvnitř — jazykové modely ÚFAL
(MorphoDiTa, NameTag, UDPipe) slouží jen jako anotační nástroje na
okraji; rozhodování uvnitř je čitelné, deterministické a měřené.

Tři přesvědčení drží celou stavbu pohromadě:

1. **Rozhodnutí patří do dat, mechanismus do kódu.** Chování systému —
   od dialogových aktů po gramatiku otázek — nesou JSON karty a jazykové
   tabulky. Kód umí jen mechanismy: klasifikovat slova, matchovat vzory,
   procházet graf, sčítat aktivaci. Nový vzor chování znamená novou
   kartu, ne nový `if`.
2. **Dialog je víc než figly.** Když si systém není jistý, zeptá se;
   když neví, řekne to. Poctivé „nenašel jsem" je architektonická
   hodnota, ne selhání.
3. **Každá změna se měří.** Pět benchmarků (etalon odpovědí, focus
   aktivace, dialogové scénáře, zápisový etalon paměti, coverage) tvoří
   bránu každého commitu. Známé nedostatky nežijí v hlavách, ale
   v „gap" řádcích benchmarků, kde na sebe upozorní, jakmile je někdo
   opraví — nebo rozbije.

Zbytek dokumentu sleduje cestu jedné české věty systémem: od znaků přes
tvarosloví a stavbu věty k faktům, aktivaci a odpovědi. Kapitola 7 pak
staví vedle faktového grafu druhý — **otázkový** — a ukazuje na
příkladech, co mají společné, kde se liší a kde se překrývají.

---

## 1. Cesta věty I: tokeny a tvarosloví (lexer)

Čeština je flektivní jazyk s volným slovosledem — tatáž informace má
desítky povrchových podob („Marcela bydlí v Petrovicích" / „V Petrovicích
bydlí Marcela") a tentýž tvar mívá mnoho čtení (klasický příklad z
literatury: „jarní" nese až 27 možných morfologických značek). Systémy,
které se pokusí tvar rozhodnout izolovaně a předčasně, se spolehlivě
spálí — v jellyAI3 na to existuje památník: tvar „Marcele" přečetl
tagger jako vokativ mužského jména „Marcel" a ze ženy udělal muže.

Proto první vrstva, **lexer** (`jellyai/lang/lexer.py`), dvojznačnost
zásadně NEROZHODUJE. Každému tokenu přiřadí **množinu hypotéz** — druhy
slova, kterými by token mohl být. (Pozor na čtení: názvy tříd jako
`otaz`, `cas`, `funkcni` nejsou překlepy, ale krátké TECHNICKÉ
identifikátory přesně v podobě, v jaké je píší vzory na kartách —
`otaz` = tázací slovo, `cas` = časový výraz, `funkcni` = funkční slovo,
`l_tvar` = l-ové příčestí. V kartách je ale obvykle zakrývají čitelné
grok-zkratky `%{TAZACI}`, `%{SLOVESO_MINULE}` — viz kapitola 2.)

| token | hypotézy | proč |
|---|---|---|
| „byt" | spona, substantivum | deakcentovaně ≡ „být" |
| „byl" | l-tvar, spona | minulé příčestí i spona |
| „Kdo" | tázací, jméno | kapitalizace na začátku věty |
| „Marcela" | jméno, l-lookalike | po ořezu koncovky končí na -l |
| „Můj" | přivlastňovací, jméno | kapitalizace |
| „prsi" | sloveso (katalog) | bezdiakritický katalogový tvar |

Třídy se počítají výhradně z jazykových tabulek `cs.json` (tázací slova,
spony, částice, časové výrazy, přivlastňovací, katalogy…) plus tří
mechanismů: regexp pro hodnoty mimo přirozený jazyk (e-mail), ořez
l-ového příčestí a rozpoznání prézentního slovesa (koncovky + katalog
+ „entity-first" veto: slovo, které je doslovným jménem uzlu grafu,
sloveso není — „nádraží" končí na -í, ale je to věc).

Katalogy jsou zde záměrná filozofie místo univerzálních pravidel:
koncovka „-i" by jako pravidlo viděla sloveso v každém plurálu („psi"),
kdežto katalogový řádek `"prsi" → prší` spraví přesně to, co spravit má.
Poučení je zaplacené: jeden „univerzální" fix l-tvarů kdysi rozbil
klasifikaci celého korpusu.

Kdo rozhodne, kterou hypotézu token nakonec dostane? Stavba věty —
další vrstva.

## 2. Cesta věty II: stavba věty (matcher a vzorové karty)

Klasická počítačová lingvistika řešila stavbu věty rekurzivními
přechodovými sítěmi (ATN) a bezkontextovými gramatikami. ATN je přesně
„stavový automat s podmínkami a rekurzí" — a historicky zemřel na
neudržovatelnost: síť přechodů s podmínkami je program, jen hůř čitelný.
jellyAI3 jde vědomě jinudy: **vzory jsou přísně regulární sekvence**
a žijí jako DATA na kartách; vykonavatel (`jellyai/lang/matcher.py`)
je mechanismus o pár desítkách řádků.

Vzor je posloupnost prvků nad třídami lexeru. Prvek umí: třídu
(`otaz`), třídu s konkrétními tvary (`otaz:kdo|koho`), literál
(`:v|ve|na`), **vyloučení** (`l_tvar!spona` — „byl" je hypotézově obojí
a identitní otázky „Kdo byl robot?" patří jinému mechanismu), **span**
`uzel+` (jeden až n tokenů, které graf potvrdí jako entitu — „Karel
Čapek", „Válka s mloky"; hladově nejdelší, umí ustoupit, aby zbytek
vzoru vyšel, a nesmí začínat ani končit funkčním slovem), **zbytek**
`*` a volitelnost `?`. Match je ukotvený na celou větu — vzor si nemůže
„ukrást" podobnou větu jiného smyslu.

Aby vzory zůstaly čitelné a DRY, píší se **pojmenovanými zkratkami**
po vzoru grok/logstash (tabulka `pattern_aliases` v cs.json):

```json
"pattern": ["%{TAZACI}", "?%{SE}", "%{SLOVESO_MINULE}", "%{ENTITA}"]
```

je vzor otázky „Kdo napsal R.U.R.?" i „Kde se narodil Karel Čapek?".
Zkratky se rozvíjejí rekurzivně, víceprvková zkratka se vkládá (splice)
a překlep jména spadne nahlas — tichá chyba v gramatice je nepřípustná.

Dvojznačnost z vrstvy 1 řeší právě pozice ve vzoru: ve větě „Roník jí
granule" obsadí slot slovesa „jí", protože nic jiného ho obsadit neumí
a věta bez predikátu nematchne nic. Plochý příznak tohle nikdy vidět
nemohl — dvojznakové sloveso je pod délkovým guardem; stavba věty ano.

Vzorové karty existují ve dvou rodinách:

- **Dotazové** (event `utterance.query`): mají přednost před staršími
  pozičními šablonami v `query.py`; když žádná nesedí — nebo když
  predikát nezná slovník grafu (překlep „sworil") — věta propadne
  pozičním šablonám, které mají vlastní léky (UDPipe fallback
  z dotazové cesty zmizel řezem #14; ÚFAL slouží už jen anotaci
  korpusu a nominativizaci zápisu).
  Z karty vzniká pseudo-QL `Pattern`: díru (roli a typ odpovědi) určuje
  tázací slovo tabulkou `interrogatives`, známé účastníky spany.
- **Výrokové** (event `utterance.statement` s klíčem `pattern`):
  soutěží o výrok s klasickými rysovými kartami JEDNOU prioritou decku
  — e-mailový atribut (priorita 12) tak správně přebije kartu krátkých
  sloves (8) na větě „Karel má email…".

Rekurze je v systému jediná a plochá: **klauzulová**. Souvětí „Roník jí
i vegetariánskou stravu, má však rád i maso." se rozdělí po čárkách
a každá klauzule projde týmž automatem; zachrání se aspoň první
rozpoznaná (částečný zápis je lepší než ztráta výroku). Hlubší vnoření
ať skončí poctivým dialogem — to je čitelná hranice.

## 3. Cesta věty III: význam (reifikovaný faktový graf)

Výsledkem obou cest — zápisu i dotazu — je týž rámec: **predikát +
typovaní účastníci v rolích**. Fakt je v grafu samostatný uzel
(reifikace) s vahou a hranami rolí: subj, obj, loc, time, num, pred,
attr a theme. „Marcela bydlí v Petrovicích", vysloveno v dialogu
19. července 2026, se stane faktem:

```
bydlí( obj: Marcela ⟨concept⟩,
       loc: Petrovice ⟨geo⟩,
       theme: uživatel ⟨person⟩,      ← pozorovatel, ne účastník!
       time: 19. července 2026 19:53 )
```

Uzel „uživatel" je u dějů pozorovatelem (theme) — metadata výpovědi.
Z toho plyne důležitá odpovědní zásada: **pozorovatel není odpověď**.
„Kdo bydlí v Petrovicích?" nesmí vrátit „uživatel", i kdyby byl uzel
sebežhavější; stejně tak časová kotva odpovídá jen časové díře.

Korpusová a uživatelská fakta žijí v JEDNOM grafu — otázka „Kdy jsem
měl knedlíky?" jde toutéž cestou jako „Kdy zemřel Karel Čapek?". Liší
se proveniencí (uživatelská nesou timestamp dialogu) a tou se řídí
i speciality: negovaný predikát (`neprší`) je evidence opaku a při
existenční otázce vyhrává **nejnovější** evidence — datovaná paměť
přebije nedatovaný korpus, „Prší?" po „Už neprší." odpoví
„Ne, od 19. července 2026 neprší."

Kvalitu korpusové části grafu hlídá při buildu **hygiena korpusových
hlasování**: tagger v jedné větě lže (mis-tagy, špatné pády), rozhoduje
proto hlasování tvaru přes celý korpus. Následuje kanonizace
a nominativizace identifikátorů (Betlémě→Betlém), slévání pádových
tvarů (aliasy) a instanční vrstva (srůst jmen JEN z textového tvrzení
„X řečený Y" — kontextový otisk identitu prokazatelně nerozliší,
změřeno: Ježíš–Nazaretský 0.31 ≈ Jan–Křtitel 0.28).

## 4. Aktivace: pozornost systému

Nad grafem leží **aktivační pole** (`ActivationField`) — jas uzlů,
obdoba klasické spreading activation. Je to jediné médium, kterým smí
cokoli ovlivnit pořadí odpovědí: rozřešení jmen otázky rozsvítí
kandidáty, časový interval rozsvítí časové uzly, volba z nabídky
zahřeje vybraného kandidáta, zapsaný výrok své účastníky. Ranking
odpovědi je pak váha faktu + role/typ díry + jas — a remízy řadí
aktivace, nikdy nerozbíjí (stabilita výčtů napříč konverzací).

Vedle měkkého světla existují **tvrdé filtry** (brány Q subsystémů):
časový interval otázky („v 19. století") fakty mimo interval vyřadí,
místní oblast („v Čechách") pustí jen fakty uvnitř kontejnmentu.

## 5. Dirigent Iris a karty

`IrisAutomaton.turn()` je dirigent dialogu. Jeho rozhodování nesou
**karty** (`jellyai/iris/patterns/cs/*.json`) — každá má trigger (kdy
sedím), dialogovou šablonu (co říkám), akci (co dělám s aktivací či
pamětí) a `teach` (výklad pro člověka). Kartami jsou dialogové akty
(nabídka homonym, přetékající výčet, upřímný terminál, potvrzení
zápisu), klasifikace výroků, a od #46 i gramatika otázek. Deck měří
kartám telemetrii: použití a měřený zisk aktivace.

`QueryAssurance` počítá jistotu rozřešení otázky z evidence; pod prahem
karty automat neodpovídá, ale ptá se — a volbou uživatele otázku
přehraje zaostřenou. Týž princip chrání zápis: jméno, které se na osobu
grafu rozřeší jen částečně („Emil" → malíř „Emil Filla"), se nepřipíše
mlčky — karta clarify-identity nabídne existující osobu i založení nové.

## 6. Doménoví experti: Chronos, Mnemos, Topos

Subsystémy (`jellyai/iris/subsystems/`) jsou experti nad osami reality
se společným půdorysem: POZNÁNÍ (vzory a tabulky), KANONIZACE (jazyk ↔
kanonický záznam), AKTIVACE (reflektor do pole) a ZÁZNAMY (JSONL
sklady). S uživatelem nikdy nemluví přímo — mluví Iris. Zapojují se
třemi branami: E (extrakce z výroku), Q (nárok na tokeny otázky →
tvrdý filtr), A (osvětlení grafu). Jsou záměrně srostlí s grafem —
nejsou to knihovny; nezávislé jsou naopak **kanálové adaptéry**
(konzole, okno, alarm, e-mail; budoucí audio STT/TTS), které jen nosí
zprávy.

**Chronos (čas).** Kanonizuje časové výrazy na intervaly: „letos",
„v 19. století", „21.1.1900", „za čtvrt hodiny", dny v týdnu, části dne
(ráno = 7:00, uživatel může přeučit). Vede PŘIPOMÍNKY: sklad
`reminders.jsonl`, vlastní vlákno hodin (ChronosTicker) tepe nezávisle
na dialogu a dozrálé připomínky doručuje kanály (konzole, okno ⏰ přes
REST event). Zadávání má chytré defaulty (událost „v pět" → předstih
2 hodiny; „zítra" → ráno; „příští týden" → neděle večer — vždy
s nabídkou změny), správa se vede dialogem („zruš všechno na zítra",
„posuň všechny ze zítra na čtvrtek", „přeplánuj to ze 17 na 20")
a výpis plánu se zužuje intervalem otázky. Děje se kotví s momentovou
granularitou („Venku prší" = den + hodina:minuta) — paměť nikdy nedrží
relativní slovo („dnes"), vždy absolutní kotvu.

**Mnemos (paměť uživatele).** Brána zápisu: konstatování klasifikují
karty na druhy — epizoda 1. osoby („Dnes jsem měl knedlíky"), sponové
pozorování („Roník je pes"), prézentní děj („Venku prší"), připsaný
fakt (po „Měl KČ rád knedlíky?" a odpovědi „nenašel" doplní „ano, měl
rád knedlíky" subjekt z konverzačního těžiště), atribut s hodnotou mimo
jazyk (e-mail regexpem obchází tagger) a doslovná poznámka (přísloví po
příkazu „zapamatuj si"). Fakt jde do grafu i do deníku `memory.jsonl`
(append-only, po restartu se přehraje). Jména a místa se před zápisem
nominativizují (jména morpho lemmatem s vokativním guardem, místa
UDPipe kontextem s délkovou pojistkou a výjimkou toponymních koncovek
-ice/-any: Petrovicích→Petrovice, ale Lhotě↛Lhot). Zapomínání je
trojí — fakt, celá entita („zapomeň na Ronika", kmenová shoda přes
pády), období („zapomeň, co jsem řekl včera") — a maže konzistentně
graf i deník. Nové predikáty z výroků rozšiřují slovník dotazového
parseru: co se zapíše, na to se lze hned ptát.

**Topos (prostor).** Pseudo-mapa bez souřadnic: gazetteer
(`sub_topos_gazetteer.jsonl`) drží kontejnment (Praha ⊂ Čechy ⊂ Česko)
a sousedství (near). Plní se kurátorským seedem, diktátem uživatele
a hlavně UČENÍM ZA POCHODU: výrok „Pavla a Matěj bydlí na Barrandově
v Praze" naučí Barrandov ⊂ Praha mimochodem. Klíče míst zvládají pády,
palatalizaci (Praze↔Praha) i epentezi (Plzni↔Plzeň) — mimochodem týž
mechanismus, jaký klasická dvouúrovňová morfologie řeší konečnými
automaty. Brána Q dělá z oblasti otázky filtr („Pršelo v Čechách?" —
fakt s Prahou projde kontejnmentem), slovesná třída přesunu rozšiřuje
„kde" na trasy („Kde putoval Ježíš?"). Topos je zrcadlem Chronosu:
interval na ose času ⟷ strom kontejnmentu na ose prostoru.

**Metron (počty a výrazy).** Nejmladší expert (`subsystems/metron.py`):
aritmetika s prioritou operátorů a slovními číslovkami vč. složenin
(„Kolik je dvěstě plus osmdesátdva?" — nikdy eval, #56) a početní
díra nad fakty (počet je VLASTNOST situace: „Kolik měla dětí BN?" →
čtyři, karta q-kolik-pocet). Zbývá počítání výskytů („Kolikrát letos
pršelo?", #11).

**Plánovaný expert:** Echo (kompozice odpovědí, #20).
Koncepčně největší je **oceán vrstev**
(#41): dědění vlastností po druh-hranách s útlumem („Psi mají rádi
maso" → Roník je pes → Roník má nejspíš rád maso; přímý fakt vždy
poráží zděděný) — klasické defeasible inheritance, jehož zárodky
(predikát `druh`, typový join, kontejnment Toposu) už v systému stojí.

## 7. Dva grafy: faktový a otázkový

Až potud měl dokument jeden graf — faktový. Od #57 stojí vedle něj
druhý, **otázkový** (`jellyai/iris/qgraph.py`, spec
`2026-07-20-otazkovy-graf.md`); začal jako experiment na větvi,
dnes je zamergovaný a od #51 je JEDINÝM dispatcherem vstupu. Nejsou
to konkurenti; jsou to dvě různé věci, které jen sdílejí tvar.

**Jednou větou:** faktový graf ví, CO systém zná; otázkový graf ví,
JAK se lze ptát a KAM otázku poslat.

### 7.1 Co mají společné

Oba jsou grafy uzlů a hran, oba se **rozsvěcují** (aktivace jako jediné
médium vlivu), oba vznikají z **dat, ne z kódu** (faktový z korpusu
a dialogu; otázkový kompilací karet a schématu predikátů), a oba jsou
**měřené** (etalon/focus/dialog × shadow harness). Sdílejí i disciplínu:
co graf nerozhodne, končí poctivým dialogem, ne heuristikou.

### 7.2 Kde se liší

| | Faktový graf | Otázkový graf |
|---|---|---|
| Uzel | fakt / entita („bydlí(Marcela, Petrovice)") | typ otázky („kdo-udělal-co"), worker, zpřesnění |
| Hrana | role účastníka (subj, obj, loc, time…) | digging: `zpresneni`, `navrat` |
| Životnost | trvalý, roste zápisem | trvalý, ale **kompilát** — přegeneruje se ze zdrojů |
| Kdo ho píše | korpus + uživatel (paměť) | karty + schéma predikátů + telemetrie |
| Průchod znamená | odpověď (match → díra) | směrování tahu a vedení dialogu |
| Když je prázdný | „nenašel jsem" | tah propadne pozičním šablonám |
| Hlavní metody | `FactGraph.add_fact`, `facts_of`, `_match`, `_answer_from`, `_identity_vote`, `ActivationField.warm/step` | `compile_qgraph`, `illuminate`, `decorate`, `DialogPosition` |

### 7.3 Kde se překrývají

Tři místa, a všechna jsou zajímavá:

1. **Schéma predikátů.** Otázkový graf odvozuje své instance
   z faktového: podpis `napsat{subj: person, obj: dílo}` implikuje
   otázky „Kdo napsal X?", „Co napsal Y?", „Napsal X Y?". *Otázky jsou
   stín schématu faktů* — proto se sémantika nekurátoruje ručně.
2. **Orákulum entit.** Vzorový prvek `uzel+` se ptá faktového grafu,
   jestli je rozpětí entita (`_span_is_node`) — bez datového grafu by
   otázkový graf neuměl poznat „Válku s mloky" od dvou slov.
3. **Aktivace.** Efemérní **větný graf** tahu (tokeny + hypotézy lexeru
   + nároky expertů) rozsvěcuje uzly otázkového grafu; jas
   *faktového* grafu (těžiště konverzace) zase vybírá téma, když je
   otázka elidovaná („Kdy se narodil?").

### 7.4 Ukázka A: jednoduchá otázka

„Kdo napsal R.U.R.?" — jeden osvětlený uzel, žádné zpřesňování.

```
větný graf:   kdo{otaz}  napsal{l_tvar}  R.U.R.{uzel+ ✓ orákulum}
                 │
otázkový graf: [q-otaz-minuly] svítí (jediný)   worker = graf
                 │                               dekorace: —
faktový graf:  _match(napsat, {R.U.R.}, díra subj/person)
                 │
odpověď:       Karel Antonín Čapek
```

Aktivace faktového grafu po tomto tahu (skutečné okno z provozu):

```
Karel Antonín Čapek = 2.68   ← odpověď svítí nejvíc
R.U.R.              = 2.10   ← rozřešená entita otázky
Josef Čapek         = 0.51   ← vyzářeno po hranách (bratr)
člověk, bratr       ≈ 0.35   ← okolí tématu
```

Následující tah těží právě z tohoto jasu: **„Kdy se narodil?"** nemá
podmět, ale nejteplejší rodově shodná osoba je Karel Čapek — odpověď
„2. října 1842". Tomu se říká pro-drop na straně dotazu a je to čistý
příklad toho, že aktivace není ozdoba, ale nosný mechanismus.

### 7.5 Ukázka B: složitá otázka

„Co řekl Ježíšovi?" — dvě věci najednou: soutěž uzlů **a** dekorace.

```
větný graf:   co{otaz}  řekl{l_tvar}  Ježíšovi{dativ ⇒ hypotéza adresáta}
                 │
otázkový graf: svítí DVA uzly, vyhrává těsnější:
                 [q-rekl-adresatovi]  (priorita 6)  ← vítěz
                 [q-otaz-minuly]      (priorita 4)
                 │
dekorace:      role:adresat   ← NEsoutěží, věší se na vítěze
                 │
faktový graf:  _match(říci, {Ježíš}, díra obj) + rolová vazba theme
                 │            └ známý MUSÍ být v roli theme (adresát)
odpověď:       „umřel"   (Martina slova Ježíšovi — ne Ježíšova vlastní)
```

Rozdíl mezi soutěží a dekorací je jádro modelu. Kdyby Chronos, Topos
nebo dativní vazba **soutěžily** s typem otázky, přebily by ji — ve
skutečnosti jsou to omezení: „Kdy jsem měl v tomto roce knedlíky?"
vyhraje uzel otázky a nese dekorace `chronos:interval` (filtr času)
a `mnemos:prvni-osoba` (podmět = uživatel). Měření na etalonu:
predikované dekorace odpovídají skutečně aplikovaným filtrům
(`time_filter`, `place_filter`, `_theme_bound`) ve **41/41** případech.

### 7.6 Ukázka C: dialog jako procházka grafem

Nejsložitější případ nemá jednu odpověď — má jich 143.

```
❓ Co řekl Ježíš?
   → uzel [q-co-vime / q-otaz-*] svítí, faktový graf vrací PŘETEČENÍ
   → hrana `zpresneni` → uzel [focus-offer-overflow]
💬 Mám tu více dat… Nabízí se: Jeremiáš, Petr, bůh, celník, chleb…
❓ Petr
   → stojíme ve zpřesňovacím uzlu; volba = hrana `navrat`
   → otázka se PŘEHRAJE s dominantní oblastí (pick_focus)
💬 satane, nemohli jedinou hodinu bdít se mnou
```

Zpřesňovací uzly jsou plnohodnotnými uzly grafu: je-li takový uzel
aktivní, dialog pokračuje zpřesňováním a **graf si sám ostří focus**.
Stav dialogu pak není roztroušený po objektech (`PendingFocus`, drill,
`pick_focus`), ale je to jediná věc: **pozice v grafu** (`DialogPosition`).

### 7.7 Jak je uspořádaná báze karet

Klíčové zjištění pro čtenáře, který hledá „kde to bydlí": **karty jsou
jen jedny** — všechny leží v `jellyai/iris/patterns/cs/*.json` a čte je
jeden `PatternDeck`. Liší se **událostí** v triggeru, a právě podle ní
se dělí mezi oba světy:

| Událost triggeru | Role karty | Kam patří |
|---|---|---|
| `utterance.query` (+`pattern`) | tvar otázky (vzor tříd lexeru) | uzel `otazka` otázkového grafu |
| `utterance.statement` | typ výroku (vzor nebo ploché rysy) | zápis do faktového grafu |
| `resolve.ambiguous`, `data.overflow`, `focus.low`, `statement.subject` | dialogový akt | uzel `clarify` (zpřesnění) + jeho hrany |
| `memory.*`, `reminder.*`, `metron.compute`, `focus.query` | odpověď experta | uzel `worker` |
| `utterance.command` (rysy `cmd:*` z frázových tabulek) | tvar příkazu | uzel `prikaz` (#51) |

Od #51 je graf **jediným dispatcherem vstupu**: rodiny `vyrok`
(worker = brána E, zápis) a `prikaz` doplnily otázky, workery
a clarify; ruční pořadí větví v `turn()` nahradily priority karet
a osvětlení (hranici dotaz × výrok nese rys `otaznik` na kartách).
Výběr karty dělá v obou světech **týž mechanismus**: `deck.best()` —
těsnost triggeru, pak priorita, pak jméno. Otázkový graf tedy nezavádí
druhé rozhodování; jen dává plochému výběru **strukturu**: uzly dostanou
hrany (digging), workery atribut a telemetrie váhu. Karty zůstávají
zdrojovým kódem, graf je jejich zkompilovaná podoba — proto se nesmí
rozejít a proto je jeho jedinou bránou parity gate.

### 7.8 Co měření ukázalo (a kde experiment skončil)

Experiment běžel shadow-first: `benchmark/run_qgraph.py` nejdřív
chování neposouval — porovnával, kudy by šel graf, s tím, kudy systém
reálně šel. Po 100% shodě byl dispatch PŘEPNUT (fáze D) a rozsah
dorostl na celý vstup (#51): dnes harness měří **pět rovin — dispatch
dialog 11/11, výroky 15/15, stav dialogu 4/4, dekorace 50/50, etalon
50/50, vše 100 % v obou variantách** — a slouží jako parity gate
kompilace (karty se od grafu nesmí rozejít). Dva nálezy z shadow fáze
stojí za zapamatování:

- **Vrstva dekorací je popis reality**, ne nová abstrakce — to, co
  model předpovídá, answerer už dělá.
- **Váhy z telemetrie jsou zatím mrtvá větev**: naměřeno **0 remíz**
  základního klíče, takže provoz nemá kde routing ovlivnit — a to
  při jakémkoli objemu. Otázka „kolik provozu je potřeba" má dřívější
  otázku „nastávají vůbec remízy?".

Hodnota grafu se potvrdila tam, kde přesnost nestačila — ve
**struktuře**: dispatch přímých expertů je smyčka nad registrem claimů
(pořadí bran = data, `claims.py`), rodinné karty s dimenzemi zrušily
klony (E1), instance ze schématu daly chytrou clarifikaci prázdné díry
(E3 — „Kde napsal R.U.R.?" → „…kde nevím") a odvozené hrany proaktivní
nabídku rolí (E4 — „Mohu doplnit: kdy"). Zaparkovaný zůstává graf
ODPOVĚDNÍ (efemérní reifikace tahu) — budoucí směr, viz spec dotažení.

## 8. Build korpusu (offline)

```
data/raw/*.txt → index → annotate (UDPipe+NameTag, služby :8081–8083)
              → extrakce faktů (spony, apozice, koordinace, pro-drop,
                 aliasy „řečený") → hygiena (korpusová hlasování)
              → kanonizace + poziční merge → nominativizace id
              → instanční vrstva → data/graph.pkl
```

Vlastní korpus: vhoď `.txt` do `data/raw/` a spusť `./jelly index →
annotate → graph`. Etalon je guardrail — po přestavbě grafu se smí id
uzlů legitimně zlepšit (očekávání se aktualizují SE zdůvodněním), nikdy
zhoršit.

## 9. Provoz: služby, web, vývojová smyčka

- **REST Iris** (`services/iris_service.py`, :8084) — jediný vstup
  dialogu; vlastní automat, graf i aktivaci; tahy zrcadlí do konzole.
- **Web** (`./jelly web`, :8080, viewBase) — pasivní 3D displej grafu:
  dialogové okno, ⚡ aktivační okno, 📄 dokumenty, ⏰ Reminder okna.
  Webová kopie grafu se udržuje mostem `memorized/forgotten` z REST
  odpovědí.
- **ÚFAL služby** (:8081–8083) — lazy subprocesy, health-first sdílení.
- **Vývojová smyčka** (ověřená v provozu): Iris běží samostatně — po
  změně jádra se restartuje JEN :8084 a GUI přežije; řádky připsané do
  `data/web_inbox.txt` zpracuje web, jako by je uživatel napsal do
  dialogového okna (testování je vidět v GUI včetně aktivace).

## 10. Měření: pět benchmarků jako brány

| benchmark | co měří | normativ |
|---|---|---|
| `run_etalon.py` | odpovědi na korpusem kryté otázky | 37/37 (jádro 100 %) |
| `run_focus.py` | aktivace správných uzlů (top-K jasu) | 12/12 |
| `run_dialog.py` | scénáře automatu (fixní hodiny, čerstvý graf per scénář); umí GAP scénáře | 45/45 |
| `run_mnemos.py` | ZÁPIS: výrok → parse (týmž soukolím jako runtime); `--nom` i nominativizaci | 34/34 |
| `run_qgraph.py` | otázkový graf: dispatch, výroky, stav, dekorace, etalon (5 rovin, obě varianty) | vše 100 % |
| `run_coverage.py` | podíl vět korpusu bez faktu | trend |

Aktuální čísla nese `docs/HANDOVER.md` §2 (normativy rostou s gap
řádky — tabulka je spodní hranice k datu 2026-07-20).

Gap řádky jsou institucionalizovaná poctivost: známý nedostatek se
nemaže, ale sleduje; jakmile projde, benchmark ohlásí GAP-FIXED —
a jakmile ho něco znovu rozbije, číslo klesne. Nová feature bez svého
benchmarkového řádku se nepovažuje za hotovou.

## 11. Shrnutí principů

- **Karty, ne kód** — dialogové akty i gramatika jsou JSON; kód jen
  mechanismy.
- **Hypotézy, ne předčasná rozhodnutí** — dvojznačnost přiznat, nechat
  rozhodnout stavbu věty, zbytek poctivým dialogem.
- **Katalogy, ne univerzální pravidla** — výjimky se enumerují; jeden
  „chytrý" algoritmus už jednou rozbil klasifikaci.
- **Dialog > figly** — ptát se je lepší než hádat; pozorovatel není
  odpověď.
- **Jedna znalost, jedno pole** — do odpovědí smí zasáhnout jen
  aktivace a filtry; faktů se držíme v jednom grafu (korpus i paměť).
- **Data vědí, co se ví; struktura ví, jak se ptát** — faktový graf
  nese znalost, otázkový (kapitola 7) směrování a vedení dialogu;
  otázky jsou stín schématu faktů.
- **Korpusová evidence > lokální tag** — hlasování přes celek.
- **Vše měřeno** — pět benchmarků, gap řádky, parity gate; u nových
  konceptů shadow režim dřív než přepnutí.
