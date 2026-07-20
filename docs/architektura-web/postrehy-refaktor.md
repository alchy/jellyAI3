# Postřehy k refaktoru a členění projektu

*Review provedené přirozeně při psaní architektonické dokumentace (stav ke commitu
`a63e83f`, 2026-07-20). Čtyři osy: DRY, konzistence, pochopitelnost, členění.
U každého nálezu: co, kde, proč to bolí (ideálně jak se to projevilo při psaní
dokumentace) a hrubý návrh směru.
Řada nálezů má už dnes svůj backlogový bod; kde ho znám, odkazuji.*

*Stav k 2026-07-20 večer (po #51): ✅ provedeno 1.2 (shared_deck),
1.3 (recognize vrací výsledek — Metron/Chronos), 4.5 (harness z registru),
**2.5 + 3.3** (lint decku: rejstříky KNOWN_EVENTS a QUERY_ACTION_KEYS,
load spadne nahlas na překlep — první úlovek: mrtvá karta resolve-miss,
zakonzervována), **2.1** (TurnResult: `answerer.turn` + `begin_turn()`,
pick_focus je parametr answer() — bitová parita, dialog ověřen napříč
hash seedy), **jedno osvětlení na tah** (nový nález nad rámec seznamu:
_turn volal illuminate 2× a dotazová polovina vítěze nepoužívala —
teď jediné osvětlení, dotazová rodina tagovaná entity-first vetem
answereru a vítězná karta předaná answereru hintem s fallback skenem);
částečně 1.1 (sdílené jen jádro těsnosti
`trigger_specificity`) a 4.1 (instance_lit přesunut ke grafu).
Zbytek otevřen — provádět „při nejbližším dotyku", každý se svým
měřením.*

---

## Osa 1 — DRY (duplicity mechanismů, dat i konceptů)

### 1.1 Tři implementace téhož výběrového klíče (priorita × délka/těsnost)
- **Co:** rankingový klíč karet existuje třikrát: `PatternDeck.best`
  (`jellyai/iris/patterns.py:198–216`, těsnost → priorita → jméno),
  `_card_query` (`jellyai/answerer/query.py:201–214`, klíč
  `(priorita, délka vzoru)` vlastním cyklem mimo deck) a `illuminate`
  (`jellyai/iris/qgraph.py:218–224`, klíč `(tier, priorita, délka vzoru)`).
- **Proč to bolí:** spec vzorové gramatiky §7 říká „ranking POUZE přes
  deck.best" — reálně jsou místa tři a shodu dnes drží jen parity gate
  (run_qgraph). Při psaní kapitoly 14.4 jsem musel výslovně vysvětlovat,
  proč se tři místa „nesmí rozejít" — to je přesně věta, která v dobré
  architektuře není potřeba.
- **Návrh:** jeden sdílený helper (např. `rank_key(card)` v patterns.py),
  který všechna tři místa importují; harness pak hlídá jen sémantiku, ne
  aritmetiku klíče.

### 1.2 Dvě instance decku za běhu
- **Co:** `IrisAutomaton` drží svůj `self.deck` (`automaton.py:131–134`)
  a dotazová cesta si drží modulový singleton `_QUERY_DECK`
  (`jellyai/answerer/query.py:171–179`).
- **Proč to bolí:** karty se čtou dvakrát a existují dvě kopie téže pravdy;
  kompilát otázkového grafu se staví z automatového decku, ale otázky
  matchuje singleton — kdyby se lišily (test s vlastním deckem, hot-reload
  karet), dispatch a rozbor by se rozešly tiše. Do dokumentace jsem musel
  napsat „karty jsou jen jedny“, což platí na disku, ale ne v paměti.
- **Návrh:** injektovat deck do answereru (parametr konstruktoru), singleton
  zrušit; nebo aspoň `_query_deck()` sytit z automatu.

### 1.3 Rozpoznávač experta běží dvakrát za tah
- **Co:** claim `recognize` volá `metron.compute(text)` /
  `chronos.clock_answer(...)` (`jellyai/iris/claims.py:31–38`) a vzápětí
  handler `_metron_query` / `_clock_response` (`automaton.py:451–489`) volá
  tutéž funkci znovu, aby dostal výsledek.
- **Proč to bolí:** dvojí výpočet je drobnost výkonově, ale koncepčně jde
  o dvojí pravdu „kdo tah bere": recognize může říct ano a handler pak None
  (vrstvy se mohou rozjet). V kapitole 8 jsem to musel obejít formulací
  „tenké obálky nad branami Q".
- **Návrh:** `recognize` ať vrací výsledek (nebo `claim.handle(text, now)
  -> Response | None` přímo), handler ho jen zabalí kartou.

### 1.4 Meta-focus fráze čtené na dvou místech
- **Co:** `_meta_focus` v `claims.py:41–45` i `_focus_query`
  v `automaton.py:491+` čtou tutéž tabulku `focus_query_phrases`.
- **Proč to bolí:** speciální případ bodu 1.3 — logika rozpoznání meta-otázky
  je zduplikovaná (jednou pro dispatch, jednou v handleru).
- **Návrh:** viz 1.3 (claim vrací výsledek), pak zbyde jediné čtení.

### 1.5 Číslovky ve dvou tabulkách
- **Co:** `word_numerals` (`jellyai/lang/cs.json:101–111`, Metron)
  a `temporal.numerals` (`cs.json:210–216`, Chronos) — dvě tabulky slovních
  číslovek s odlišným pokrytím a odlišnými klíči (Metron bez pádových tvarů
  „dvema/tremi", Chronos bez stovek/tisíců).
- **Proč to bolí:** „kolik je dvěma plus tři" a „za dvěma hodiny" čtou jiná
  data; rozšíření jedné tabulky druhou neovlivní — přesně ten druh tichého
  rozjezdu, který jinde projekt řeší jedním zdrojem pravdy.
- **Návrh:** jedna tabulka číslovek + doménové rozšíření (pádové tvary jako
  aliasy), oba experti čtou z ní.

### 1.6 Pět nezávislých systémů koncovek/kmenování
- **Co:** `suffixes` (`cs.json:4–7`, kmenování), `_NAME_SUFFIXES` natvrdo
  v kódu (`jellyai/iris/subsystems/mnemos.py:33`), `feminine_name_suffixes`
  (`cs.json:11`), `place_nominative_endings` (`cs.json:153`),
  `dative_endings` (`cs.json:100`) + ořez `rstrip("aioy")` v `l_stem`
  (`jellyai/lang/lexer.py:60`).
- **Proč to bolí:** příbuzné mechanismy („jak z tvaru na kmen/nominativ")
  žijí na šesti místech; `_NAME_SUFFIXES` a `"aioy"` jsou navíc jazyková
  data v Pythonu — přímé napětí se zákonem 3 (jazyk jako data). Při psaní
  kapitoly 4.6 jsem musel koncovkové tabulky vyjmenovávat po jedné, protože
  nemají společné jméno ani místo.
- **Návrh:** až přijde vlastní lematika (BACKLOG doporučení „kořenový
  parser"), sjednotit do jedné kmenové vrstvy v `lang/`; do té doby aspoň
  přesunout `_NAME_SUFFIXES` a `"aioy"` do cs.json.

### 1.7 `decorate()` je vědomý stín skutečných filtrů
- **Co:** `qgraph.decorate` (`jellyai/iris/qgraph.py:105–133`) predikuje
  dekorace, které answerer nastavuje nezávisle (`time_filter`,
  `place_filter`, `_theme_bound`, `user_subject`, `novelty`
  v `graph_answerer.py` / `query.py`).
- **Proč to bolí:** je to přiznaná shadow duplicita („popis reality") držená
  paritou 42/42 — ale trvale hrozí rozjezd: nová dekorace vyžaduje změnu na
  dvou místech. Dokumentace to musí vysvětlovat jako „čtou z týchž tabulek",
  což je pravda jen konvencí.
- **Návrh:** až se dekorace formalizují (#26 plné claim() se spany), ať
  answerer filtry *odvozuje z dekorací* (jedna produkce, dva konzumenti),
  místo dnešního dvojího čtení tabulek.

### 1.8 Mrtvá tázací mapa vedle živé
- **Co:** `_HOLE` v `jellyai/answerer/pattern.py:24–29` (zakonzervovaná
  UDPipe cesta) vs. živá tabulka `interrogatives` (`cs.json:41–53`);
  k tomu překryv `question_words` (`cs.json:147–151`) s klíči
  `interrogatives`.
- **Proč to bolí:** při psaní kapitoly 10.1 jsem si musel ověřovat, která
  mapa je živá; `question_words` a `interrogatives` se musí udržovat
  synchronně ručně (nové tázací slovo = dva zápisy).
- **Návrh:** `question_words` generovat z klíčů `interrogatives`
  (+ příslovce), `_HOLE` přesunout do conserved souboru, ať v živém modulu
  neleží.

## Osa 2 — Konzistence (stejné věci řešené různě)

### 2.1 Výsledek tahu answereru jako 9 side-channel atributů
- **Co:** `answer()` komunikuje s automatem přes mutované atributy:
  `last_trace`, `_prev_trace`, `last_resolution`, `last_overflow`,
  `last_query_card`, `last_empty_role`, `last_pattern`, `pick_focus`,
  `_theme_bound`, `_last_values` (`graph_answerer.py:107–109, 1021–1045,
  1101–1106`); automat je čte přes `getattr(..., None)` a jeden dokonce
  nastavuje zvenčí (`setattr(self.answerer, "last_query_card", None)`,
  `automaton.py:174`).
- **Proč to bolí:** návratová hodnota (`Answer`) nese jen část výsledku;
  zbytek je rozprostřen po objektu s křehkým protokolem nulování („nesmí
  přežít z minula" — komentáře na třech místech). Při psaní kapitol 11–13
  jsem pro každý mechanismus musel dohledávat, který atribut a kdy se
  nuluje; dvě chyby z Empirie (last_empty_role vs. assurance-fail) jsou
  přímým důsledkem tohoto stylu.
- **Návrh:** `TurnResult` dataclass (answer + trace + overflow + empty_role
  + query_card + pattern), vracená z `answer()`; per-tah stav (pick_focus,
  theme_bound) předávat jako parametr, ne atribut.

### 2.2 Tri-state a None-konvence vedle výjimek
- **Co:** `instance_lit` vrací `bool | None` (tri-state, qgraph.py:136),
  matchery vrací `None` = nesedí, ale `expand_pattern` na neznámou zkratku
  padá výjimkou (matcher.py:50); `_card_query` vrací None jak při „žádná
  karta", tak při „karta sedí, ale predikát nezná" (query.py:260–262).
- **Proč to bolí:** None znamená v různých vrstvách „nesedí", „nevím"
  i „vzdávám to ve prospěch fallbacku" — při dokumentování toku
  (kap. 3.3) je nutné každý None vysvětlit zvlášť. Tri-state je zdůvodněný
  (past 2), směšování „None jako program flow" s „výjimka jako lint" je ale
  nepsané pravidlo.
- **Návrh:** zapsat konvenci do HANDOVERu (None = legitimní ne-výsledek,
  výjimka = chyba dat/gramatiky) a v `_card_query` rozlišit oba None
  případy aspoň komentářem/log stopou pro triage.

### 2.3 Tři jmenné konvence v jednom vzoru
- **Co:** třídy lexeru česky bez diakritiky (`otaz`, `funkcni`,
  `sloveso_fin`), grok-aliasy VELKÝMI (`%{TAZACI}`), role anglicky
  (`subj`, `obj`, `loc`) — všechny tři se potkají v jedné kartě
  (`q-otaz.json`).
- **Proč to bolí:** čtenář (i dokumentátor) musí držet tři slovníky; kapitola
  4 potřebovala výslovný rámeček „pozor na čtení názvů tříd". Není to chyba,
  ale tichá bariéra vstupu.
- **Návrh:** neřešit přejmenováním (rozbilo by karty), ale doplnit do
  cs.json/HANDOVERu jednu převodní tabulku třída ↔ alias ↔ role; dokumentace
  ji nyní supluje.

### 2.4 Worker atribut má dva slovníky hodnot
- **Co:** `QNode.worker` nabývá `"graf" | "metron" | "chronos" | "iris"
  | "dialog"` (qgraph.py:40), ale handler mapa v `_expert_turn`
  (`automaton.py:446–448`) zná jen tři z nich a jmenuje se jinak než
  subsystémy (meta-focus → `"iris"`).
- **Proč to bolí:** položka `"iris"` znamená „meta introspekce", `"dialog"`
  znamená „clarify karta" — sémantika hodnot není nikde vypsaná (dokumentace
  kap. 5.6 ji rekonstruuje z použití). Nový claim s překlepem workeru tiše
  propadne (záměr — ale bez logu se to nedozvíte).
- **Návrh:** enum/konstanty pro worker hodnoty + debug log „neznámý worker
  X propadá"; mapu worker→handler přesunout k registru claimů (dnes je to
  druhá ruční tabulka hned vedle data-driven dispatch — polovina migrace).

### 2.5 Události automatu: řetězce bez rejstříku
- **Co:** eventy `"utterance.query"`, `"data.overflow"`, `"focus.low"`,
  `"answer.offer-roles"`, `"statement.subject"`, `"metron.compute"`… jsou
  volné řetězce roztroušené v kódu (deck.best volání) a kartách;
  `_CLARIFY_EVENTS` (qgraph.py:26) je jediný částečný rejstřík.
- **Proč to bolí:** překlep eventu na kartě = karta se nikdy nevybere, tiše;
  seznam všech eventů pro kapitolu 7.8 jsem musel skládat grepem přes kód
  i karty.
- **Návrh:** jeden modul/konstanta se seznamem eventů + test, že každá karta
  používá známý event (levný lint, chytí překlepy).

## Osa 3 — Pochopitelnost (kde jsem se při psaní zasekl)

### 3.1 `IrisAutomaton._turn` — 160 řádků, pořadí větví = sémantika
- **Co:** `automaton.py:199–360`; posloupnost claims → resume identity →
  resume pick → (bez otazníku: pending reminder → forget → memorize →
  plan_manage → send → reminder → focus_shift → statements) → recall →
  plan → interval warm → answer → karty.
- **Proč to bolí:** při psaní kapitoly 11.2 jsem sled četl třikrát, než jsem
  si byl jistý pořadím a podmínkami přeskoků (např. že `takeover` u pending
  reminderu je vnořená výjimka výjimky). Pořadí větví je nosná sémantika,
  ale čitelná jen z kódu — přesně bolest, kterou pro dotazovou polovinu
  vyřešil qgraph dispatch (#51 pro zbytek).
- **Návrh:** pokračovat v započatém: výrokovou/příkazovou polovinu převést na
  claims/karty (etapa #51); do té doby aspoň rozdělit `_turn` na pojmenované
  fáze (`_direct_experts`, `_resume`, `_command_turn`, `_query_turn`).

### 3.2 `GraphAnswerer._match` — 116 řádků, 8 vyřazovacích guardů
- **Co:** `graph_answerer.py:554–670` — guardy theme/čas/geo/tautologie/
  vztažná jména + tři skórovací bonusy + overflow + pick_focus dominance
  v jednom průchodu.
- **Proč to bolí:** každý guard nese komentář-příběh (dobré!), ale funkce
  jako celek se nedá vyložit jinak než odstavec po odstavci (kap. 10.7 je
  proto seznam odrážek). Křehké je pořadí guardů a sdílené proměnné
  (`matched`, `base`).
- **Návrh:** guardy vytáhnout jako pojmenované predikáty
  (`_observer_guard(part, fact)`, `_time_anchor_guard(...)`) sestavené do
  seznamu — pořadí zůstane explicitním datem funkce; skórování oddělit od
  filtrace.

### 3.3 `_card_query` — malý interpret akcí karty
- **Co:** `query.py:182–297` — klíče `hole`, `date_part`, `copula`,
  `hole_role` (literál vs. $ref), literální `predicate`, `user_subject`,
  `novelty`, default role známých… každý s vlastní výjimkou.
- **Proč to bolí:** je to de facto mini-DSL akce karty; při dokumentování
  (kap. 14.6) jsem musel číst každý klíč zvlášť a dva (copula vs. attr,
  hole_role literál) vícekrát. Roste-li DSL dál, hrozí „program v datech"
  zadními vrátky.
- **Návrh:** tabulka podporovaných klíčů akce s krátkou specifikací přímo
  v docstringu/spec + odmítání neznámých klíčů (dnes se tiše ignorují —
  překlep `hole_rol` by prošel bez povšimnutí).

### 3.4 Vakuový guard vzorových karet v `_specificity`
- **Co:** `patterns.py:160–164` — vzorová karta vrací z rysové cesty None,
  jinak by vyhrála „naprázdno" (past 2).
- **Proč to bolí:** klíčová pojistka je vedlejší větev pomocné metody; bez
  komentáře by byla nepochopitelná a při čtení jsem se k ní vracel. Je to
  přesně místo, kde by test s názvem
  `test_pattern_card_never_wins_by_features` dokumentoval záměr líp než
  komentář (možná existuje — pak jen provázat).
- **Návrh:** explicitní early-return s vlastní metodou
  (`_is_pattern_card`) + odkaz na past 2 v HANDOVERu.

### 3.5 Dvojí význam „theme"
- **Co:** role `theme` = pozorovatel (metadata zápisu) i adresát řeči
  (`Co řekl Ježíšovi?" — theme faktu říci) — `extract.py:16`
  (iobj → theme), `_match` guard `graph_answerer.py:583`,
  `_theme_bound` tamtéž.
- **Proč to bolí:** kapitola 6.2 musela věnovat odstavec vysvětlení, že táž
  role nese dvě sémantiky rozlišené jen druhem faktu; guard „pozorovatel
  není odpověď" proto musí testovat `node == user_entity`, ne roli samotnou.
- **Návrh:** dlouhodobě oddělit (role `addressee` vs. `observer`), krátkodobě
  aspoň pojmenovat konstanty a sepsat sémantiku do spec faktového grafu;
  souvisí s #39 (provenience jako metadata, „autor ≠ uzel grafu" u #34/#47).

## Osa 4 — Členění (hranice modulů a vrstev)

### 4.1 Závislost answerer → iris (proti směru vrstev)
- **Co:** `graph_answerer.py:14–15` importuje
  `iris.subsystems.chronos/topos`, `graph_answerer.py:1116` importuje
  `iris.qgraph.instance_lit`, `query.py:175` importuje
  `iris.patterns.PatternDeck`.
- **Proč to bolí:** vrstevní obraz (iris → answerer → graph → lang), který
  kreslí kapitola 2, má reálně čtyři šipky opačným směrem. `instance_lit`
  je čistá funkce nad schématem (patří ke grafu), `resolve_temporal`/Topos
  jsou „doménová primitiva" používaná odspodu, `PatternDeck` je obecný
  kartový mechanismus. Kruhové importy se dnes obcházejí lokálními importy
  ve funkcích — funguje to, ale je to symptom.
- **Návrh:** (a) `instance_lit` přesunout do `jellyai/graph/` (nebo přímo
  metodou FactGraph vedle `predicate_roles`); (b) `PatternDeck` vytáhnout
  z `iris` do neutrálního modulu (`jellyai/cards.py`?) — karty používají
  obě vrstvy; (c) kanonizační primitiva Chronos/Topos oddělit od dialogové
  části subsystémů (viz 4.3).

### 4.2 `automaton.py` (1480 ř.) dělá příliš mnoho věcí
- **Co:** dirigent tahu + sklad připomínek (I/O, zámek, `automaton.py:362–410`)
  + správa plánu (`_plan_manage`, ~100 ř.) + odesílání zpráv
  (`_send_command`) + nominativizace (`_nominativize_*`, :1187–1258)
  + zapamatování/zapomínání + telemetrie.
- **Proč to bolí:** kapitola 2 popisuje automat jako „dirigenta, který jen
  hlásí události" — skutečný soubor obsahuje i výkonné části Chronosu
  (sklad připomínek) a Mnemosu (nominativizace), takže hranice
  automat/subsystém je v kódu jinde než v konceptu. Past 8 („nečti celé
  velké soubory") je přiznáním téhož.
- **Návrh:** sklad připomínek přesunout do `subsystems/chronos.py`
  (ZÁZNAMY jsou deklarovaná schopnost subsystému), nominativizaci do
  Mnemos/lang, `_send_command` do vlastního modulu příkazů; automat si
  nechá jen dispatch a dialogové akty.

### 4.3 Subsystémy míchají primitiva a dialogovou logiku
- **Co:** `chronos.py` (607 ř.) = kanonizace intervalů + resolve_due
  + ticker + clock_answers; `mnemos.py` (521 ř.) = parse_statement
  + persist/replay + forget selektory.
- **Proč to bolí:** answerer potřebuje jen kanonizační primitiva
  (`resolve_temporal`), ale importem si přitáhne celý subsystém (viz 4.1);
  v dokumentaci (kap. 9) jsem musel rozlišovat „brána Q" vs. „přímý expert"
  vs. „primitivum" — kód toto členění nemá.
- **Návrh:** rozdělit subsystém na `core` (čistá kanonizace, bez závislostí
  na iris/graph) a `expert` (brány, karty, sklady); vrstevní šipky se tím
  narovnají samy.

### 4.4 Jméno souboru ≠ jméno karty po rozvinutí rodin
- **Co:** `patterns/cs/q-otaz.json` vyrobí karty `q-otaz-minuly`,
  `q-otaz-prezens-prodrop`… (`_expand_family`, patterns.py:23); telemetrie
  a qgraph uzly nesou jména rozvinutá.
- **Proč to bolí:** kdo hledá „kde bydlí karta q-otaz-prezens-prodrop",
  soubor toho jména nenajde (grep selže). Dokumentace to řeší poznámkou,
  kód nijak.
- **Návrh:** `./jelly` subcommand (nebo rozšíření `qgraph dump`) „karta →
  zdrojový soubor"; případně konvence komentáře v teach.

### 4.5 Harness zná jména uzlů natvrdo
- **Co:** `benchmark/run_qgraph.py:43–58` `_actual_route` mapuje komponenty
  odpovědi na jména uzlů (`"metron" → "metron-vypocet"`,
  `"iris" → "meta-focus"`, `["chronos"] → "chronos-hodiny"`).
- **Proč to bolí:** přejmenování claimu v registru by harness tiše rozladilo
  (shoda by padala jako „mimo rozsah", ne jako chyba). Přesně proti duchu
  „jedna pravda v registru".
- **Návrh:** mapu odvozovat z `default_claims()` (worker → name), ne
  z literálů.

### 4.6 `Query` vs. `Pattern` — dvě poloviny jednoho pojmu
- **Co:** `Pattern` (`answerer/pattern.py:47`) nese neúplný fakt, `Query`
  (`answerer/query.py:38`) ho balí + signály (qtype, gender, place, card,
  novelty); docstring přiznává „duck-type náhrada QuestionAnalysis".
- **Proč to bolí:** historické vrstvení je vidět: část informací o otázce
  žije v Pattern, část v Query, answerer bere obě (`qa, pat`); v dokumentaci
  (kap. 5.4) je nutné vysvětlovat obě třídy naráz. `pattern.py` navíc nese
  zakonzervovanou UDPipe sekci vedle živých dataclass.
- **Návrh:** sloučit do jednoho `Query` (pattern jako pole), konzervu
  přestěhovat do conserved modulu; udělat až s další větší změnou dotazové
  cesty (zákon 6 — bez zpětné kompatibility).

### 4.7 Jazyková data v kódu (drobné, ale proti zákonu 3)
- **Co:** `_NAME_SUFFIXES` (`mnemos.py:33`), `rstrip("aioy")` + koncovka
  „l" (`lexer.py:58–61`), default role díry („obj" při díře subj,
  `query.py:271`), prahy délek (dativ ≥ 5, sloveso ≥ 3, `lexer.py:120,126`).
- **Proč to bolí:** jsou to česká fakta zapsaná v Pythonu; nový jazyk by je
  musel hledat v kódu. Každé jednotlivě je maličkost — dohromady tvoří
  „šedou zónu" zákona 3, kterou dokumentace musí vyjmenovat ručně.
- **Návrh:** přesun do cs.json při nejbližším dotyku těch míst (bez
  samostatné kampaně); prahy délek jako klíče vedle příslušných tabulek.

---

## Shrnutí priorit (subjektivní)

1. **2.1 TurnResult místo side-channel atributů** — největší poměr
   užitek/riziko; zpřehlední automat i answerer a odstraní třídu chyb,
   která se už dvakrát projevila.
2. **4.1 + 4.3 narovnání vrstev** (instance_lit ke grafu, deck mimo iris,
   subsystémy core/expert) — laciné přesuny, velký zisk pro čitelnost
   architektury.
3. **1.2 jeden deck** a **1.3 recognize vrací výsledek** — malé, mechanické,
   ruší dvojí pravdu za běhu.
4. **3.1 rozklad `_turn` na fáze** — přirozeně vyplyne z etapy #51; do té
   doby jen kosmetika, nedělat velkým třeskem.
5. Zbytek jsou udržovací poznámky — provádět „při nejbližším dotyku",
   každou se svým měřením (zákon 4).
