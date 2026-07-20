# Otázkový graf — teoretické rozpracování (#57, koncept user)

> Zadání (user, 2026-07-20): „Máme faktový graf pro zapamatované. A co
> funkční otázkový graf pro digging typy otázek a směrování dotazů na
> workery? … Answerer by svým modelovaným grafem práci s kartami
> zjednodušoval, nebo je nahrazoval?"
>
> Tento dokument koncept PROVÁDÍ mentálním modelem (konkrétní tahy
> z provozu), podle výsledků ho KORIGUJE a kriticky hodnotí. Není to
> implementační plán — je to test nosnosti myšlenky před stavbou.

---

## 1. Formální model (první nástřel)

Tři grafy, každý s jinou životností:

| Graf | Životnost | Obsah | Existuje dnes? |
|---|---|---|---|
| **Datový (faktový)** | trvalý | fakty + aktivační pole (těžiště konverzace) | ✓ plně |
| **Otázkový** | trvalý | typy otázek + digging hrany + workery + váhy | ne — plochý deck karet |
| **Větný** | jeden tah | tokeny + hypotézové množiny + nároky expertů | ✓ v malém (lexer + matcher) |

**Uzel otázkového grafu** = typ otázky: díra (role+typ), tvar povrchu
(vzor karty), worker (graf / chronos / metron / mnemos / iris-meta).
**Hrana** = legitimní pokračování dialogu (digging): upřesnění, drill,
zúžení, clarify. **Aktivace**: větný graf tahu (rozsvícené tokeny —
nároky Chronos/Metron, spany entit orákulem datového grafu, tázací
hypotézy) osvětlí uzly otázkového grafu; nejsilněji osvětlený uzel tah
směruje. **Váhy**: telemetrie (#38) — zásahy/missy per uzel.

**Odvození** (závěr brainstormu): sémantická vrstva uzlů se NEkurátoruje
— odvodí se ze schématu predikátů datového grafu. Podpis
`napsat{subj: person, obj: dílo}` implikuje uzly „Kdo napsal X?",
„Co napsal Y?", „Napsal X Y?" — otázky jsou stín schématu faktů.
Karty zůstávají jen pro POVRCH (čeština) a dialogové akty.

## 2. Mentální model — průchody konkrétních tahů

Následující tahy jsou skutečné (z benchmarků a živého provozu).
U každého: co by v modelu nastalo, a co to o modelu prozrazuje.

### T1 „Kdo napsal R.U.R.?" — základní tah

Větný graf: `kdo{otaz}`, `napsal{l_tvar}`, `R.U.R.{uzel}` (span
potvrzený datovým grafem). Osvětlí uzel Q:kdo-udělal, jeho odvozená
instance (predikát *napsat* ve schématu existuje) svítí silně;
worker = graf → odpověď Karel Čapek.

**Prozrazuje:** uzly potřebují DVĚ úrovně — generický typ
(Q:kdo-udělal) a instanci (typ × predikát). Telemetrická váha má smysl
na INSTANCI („Co řekl X?" je žhavá, „Co vyrobil X?" spí); povrchová
karta na TYPU. Instance se materializují líně (schéma predikátů je
konečné, ale stovky × typy = tisíce uzlů — předem je stavět netřeba).

### T2 „Kolik je 1 plus 1?" vs. „Kolik měla dětí Božena Němcová?"

První: větný graf rozsvítí čísla a operátory (nárok Metronu) →
Q:výpočet (worker metron) svítí, Q:počet (kolik + entita) nesvítí
(žádná entita). Druhá: entita svítí, čísla v řádku žádná → Q:počet
(worker graf, dnes #11) svítí, Q:výpočet je tmavý.

**Prozrazuje:** dispatch osvětlením PŘIROZENĚ řeší to, co dnes drží
ruční pořadí bran (Metron musel být zapojen „před hodinami" — past
#51). Dvě rodiny „kolik" se rozdělí samy podle obsahu řádku. Toto je
nejsilnější argument konceptu.

### T3 „Kdy jsem měl v tomto roce knedlíky?" — SOUBĚH, ne soutěž

Svítí: Q:kdy-udělal (tázací + l-tvar + entita), nárok Mnemos
(1. osoba → subjekt uživatel), nárok Chronos („v tomto roce" →
interval). Kdyby uzly jen SOUTĚŽILY, Chronos by prohrál — ale jeho
nárok je FILTR, ne odpověď.

**KOREKCE MODELU (zásadní):** rozsvěcení má dva druhy.
**Soutěžící uzly** (typy otázek — vyhrává jeden) a **dekorující
nároky** (Chronos interval, Topos oblast, Mnemos 1. osoba, dativní
adresát) — ty se na vítězný uzel VĚŠÍ jako omezení. Dnešní
`time_filter`/`place_filter`/`user_subject`/`_theme_bound` jsou přesně
dekorace; model je legitimizuje, ne vynalézá. Claim() z #26 je jazyk
dekorací; vzory karet jazyk soutěžících uzlů.

### T4 „Co řekl Ježíš?" → overflow → „Petr" → … „Kdy?" — DIGGING

Odpověď přeteče (143 výroků) → dnešní nabídka oblastí je v modelu
HRANA `zúžení-oblastí` z uzlu Q:co-udělal(říci) do téhož uzlu
s dekorací oblasti; volba „Petr" = krok po hraně (pick_focus).
Následné holé „Kdy?" = hrana `drill-role(time)` z ODPOVĚDNÍHO faktu.

**Prozrazuje:** stav dialogu = POZICE v otázkovém grafu. Dnešní
ad-hoc objekty (PendingFocus, _prev_trace drill, PendingIdentity) jsou
roztroušené reprezentace téhož: „kde v grafu stojím a kudy vedou
hrany ven". **KOREKCE/OBJEV:** hrany se nemusí enumerovat — digging
hrany se ODVODÍ z rolí žhavého faktu: „neosvětlené role odpovědního
faktu jsou legitimní další otázky" (fakt narodit(KČ, Hronov, 1890):
po „Kde?" vedou hrany na time a subj). Ručně zbývají jen dialogové
akty (clarify-identity, clarify-period #5).

### T5 „Kdo je Bedřich Smetana?" — MISS a učení

Q:identita svítí, worker graf → nenašel → assurance-fail (hrana
`clarify-nejbližší`). Telemetrie zapíše miss na uzel; triage (#38)
shlukuje. **Prozrazuje:** váhy uzlů dostáváme ZDARMA (telemetrie už
karty loguje); miss-težký uzel = ukazatel na díru v datech NEBO na
chybějící instanci/povrch. Učební smyčka: triage → kandidát → nová
karta (člověk v okruhu) → rekompilace grafu. Graf se neučí sám —
učí se jeho ZDROJE (karty, schéma, váhy).

### T6 „Roník jí granule." — HRANICE MODELU

Výrok, ne otázka. Otázkový graf mlčí; výroková rodina má vlastní
(menší) typologii — zrcadlový „výrokový graf" by šel postavit stejně
(typy výroků už jsou karty utterance.statement), ale je to JINÝ graf:
bez děr, s bránou E. **Prozrazuje:** #57 je dotazová polovina #51;
jednotný utterance dispatch = otázkový graf ∪ výroková typologie ∪
příkazy. Nekřížit předčasně.

## 3. Korigovaný model (po mentálním testu)

1. **Uzly dvou úrovní**: generický typ otázky (povrch = karty) ×
   líně materializovaná instance (typ × predikát ze schématu faktů;
   nese telemetrickou váhu).
2. **Dva druhy rozsvěcení**: soutěžící uzly (jeden vyhrává tah) ×
   dekorující nároky (čas/místo/osoba/adresát — věší se na vítěze).
3. **Hrany se odvozují**: digging = neosvětlené role žhavého faktu +
   zúžení při přetečení; ručně jen dialogové akty. Hrana NEnese
   podmínky (zákaz ATN, spec vzorové gramatiky §7) — jen typ přechodu.
4. **Stav dialogu = pozice v grafu** (sjednocuje PendingFocus, drill,
   pick_focus do jedné reprezentace).
5. **Kompilace, ne dvojí pravda**: graf je deterministický KOMPILÁT
   (karty + schéma predikátů + váhy telemetrie). Zdroje se editují,
   graf se přegenerovává; parity gate = bitová shoda dispatch
   rozhodnutí s dnešním pořadím větví na celém dialog benchmarku.

## 4. Kritické zhodnocení

**Co koncept řeší (reálné bolesti):**
- ruční pořadí bran v turn() (past #51) → výběr osvětlením (T2);
- roztroušený stav dialogu → pozice v grafu (T4);
- klony karet (minulý/prézens/pro-drop) → jeden typ + varianty;
- digging bez systému (#5 clarify, drill, overflow) → odvozené hrany;
- telemetrie bez struktury → váhy uzlů, triage ukazuje NA UZEL.

**Co koncept neřeší (nepředstírat):**
- povrch češtiny — karty zůstávají (homografy, slovosled, dativ);
- kvalitu dat (střepy „Ježíš Nazaretský" ruší anaforu i sebelepší
  routing — #8);
- výrokovou polovinu a extrakci (jiný graf, T6).

**Rizika:**
1. **ATN převlečený za graf** — hrany s podmínkami = program v datech;
   disciplína: hrany jen typované, tabulky, žádné výrazy. (Vědomě
   odmítnuto už ve spec vzorové gramatiky §8.)
2. **Dvojí pravda** karty × graf — řeší kompilace + parity gate; bez
   toho NEZAČÍNAT.
3. **Kombinatorika instancí** — líná materializace + váhy jen pro
   navštívené; jinak tisíce mrtvých uzlů.
4. **Předčasná obecnost**: dnešní deck.best s prioritami JE degenerovaný
   otázkový graf (uzly bez hran). Přechod se vyplatí, až bolí hrany
   (digging/stav), ne dřív — zisk T2 (dispatch) umí i #51 bez hran.

**Verdikt:** koncept je nosný a správně pojmenovává strukturu, která
v projektu vzniká živelně (dekorace filtrů, pending stavy, drill).
Hodnota není v náhradě karet, ale ve SJEDNOCENÍ: dispatch (#51),
nároky (#26), digging (#5), telemetrie (#38) a stav dialogu dostanou
jeden model. Doporučené pořadí: nejdřív #26 (jazyk dekorací — už se
píše: claim_words, _theme_bound, place_filter), pak kompilace decku
na uzly (bez hran, parity gate), hrany až nakonec.

## 5. Náčrt migrace (každá fáze měřená)

1. **Kompilace k nahlédnutí**: `./jelly qgraph` — postaví otázkový
   graf z dnešních karet + schématu predikátů a VYPÍŠE ho (uzly,
   instance, váhy z telemetrie). Žádná změna chování — jen zviditelnit.
2. **Dekorace formálně** (#26 S2): nároky Chronos/Topos/Mnemos/dativ
   jako jednotné claim() rozhraní (zárodky: claim_words, _theme_bound).
3. **Dispatch osvětlením**: turn() směruje výběrem nad zkompilovanými
   uzly; parity gate = dialog benchmark bitově shodný; teprve pak
   smazat ruční pořadí větví.
4. **Hrany**: reifikovat drill/overflow/clarify jako typované hrany;
   stav dialogu = pozice (nahradí PendingFocus/_prev_trace).
5. **Odvozené instance + váhy**: líná materializace, telemetrie per
   instance, proaktivní nabídky (top-k hran po odpovědi — dnešní
   „souvislosti" dostanou strukturu).

## 6. Otevřené otázky (k dalšímu dialogu)

- Persistence vah: per instance × per typ? Kolik provozu je potřeba,
  než váhy něco znamenají? (Telemetrie zatím dny, ne měsíce.)
- Vícejazyčnost: uzly jsou jazykově neutrální (díra+worker), povrch
  jazykový — sedí to na zákon „jazyk jako data"? (Zřejmě ano — silný
  bonus konceptu: nový jazyk = nové povrchové karty, týž graf.)
- Proaktivní hrany v UI: nabízet po odpovědi „mohu doplnit Kdy/Kde"?
  (Dialog > figly — nabídka ano, vnucování ne.)
- Vztah k oceánu vrstev (#41): dědění po druh-hranách je v DATOVÉM
  grafu; otázkový graf ho jen konzumuje (díry přes zděděné fakty) —
  žádná interakce modelů se nečeká, ověřit na T-průchodech až u #41.
