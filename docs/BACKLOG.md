# BACKLOG — otevřené body (živý dokument)

> Aktualizuj při každém uzavření/přidání bodu. Stav ke commitu: viz git log.
> Metriky teď (2026-07-19 noc, po sklizni živého dialogu): **494 testů,
> etalon 29/29 (3 gap-fixed / 5 gap), focus 12/12, dialog 27/27 (13 scénářů
> + 2 gap scénáře #44/#45), ZÁPIS 33/33 (11 gap-fixed / 5 gap: #45 ×4 +
> #35-nom) — jádra 100 %. Opraveno ze živého dialogu: částice
> občas/většinou/však (particle_words), otázka s tečkou („Kdo je Roník.")
> nevetuje-tabulkou question_words, vokativ guard („Marcele"→Marcel flip).**
>
> **➡️ PŘEDÁNÍ PRÁCE: čti nejdřív `docs/HANDOVER.md`** — zákony projektu,
> testovací smyčka, implementační tipy ke každému otevřenému bodu, pasti.

## Otevřené (v pořadí priority)

| # | Oblast | Bod | Řešení / poznámka | Priorita |
|---|--------|-----|-------------------|----------|
| 37 | Benchmark | ✅ **HOTOVO** (2026-07-19): **Zápisový etalon** — `benchmark/run_mnemos.py` + `mnemos.jsonl` (41 řádků, 11 kategorií). Měří TÝMŽ soukolím jako runtime (`parse_statement` + karty + `_known_word` veto, fixní hodiny jako dialog). Baseline: **jádro 29/29 (100 %), 12 gap řádků** (#31 ×4, #32 ×3, #24, #35, + 4 NOVÉ nálezy — viz sekce runtime). `--nom` režim měří i nominativizaci (vyžaduje ÚFAL služby). Testy kontraktu `tests/test_run_mnemos.py`. | Guardrail pro práci na #31–35/lematice. Deník neukládá surový výrok (jen parse) → sklizeň z provozu čeká na #38; sada zatím syntetická z dokumentovaných pastí. | ✓ |
| 26 | Subsystémy | **Refaktor na společný půdorys** — spec `2026-07-18-subsystemy-iris.md`: port Subsystem (poznání/kanonizace/aktivace/záznamy), tři brány (E extrakce, Q nárok v parseru, A reflektor), fáze S0–S5. Obsahuje Chronos v2 (reminder-default „za čtvrt hodiny" + přeplánování, day_parts ráno=7:00 + učení uživatelem, registr kanálů console/window/alarm/email, okno Reminder), Topos v1 (gazetteer, kontejnment, slovesná třída pohybu — „Kde putoval Ježíš?"), Mnemos jako brána zápisu pro subsystémy. | Fáze po jedné, každá měřená; S0 mechanická. STAV: S0+S1+S2-jádro HOTOVO (subsystems balíček, day_parts, reminder-default+přeplánování, okno Reminder+kanály, tvrdý filtr); zbývá S2-dočištění (formální claim() v parseru), S4 brána E, S5 učení definic (částečně: Topos se už učí za pochodu). S3 Topos jádro HOTOVO (viz archiv). Směr S2 (dialog 2026-07-19): karty vstřebávají QL jako VZOROVÝ JAZYK nad tokeny (vzory = data, matching = mechanismus), pravidlo po pravidle s parity gate — NE interpret v JSONu. | **1** |
| 38 | Provoz | **Triage telemetrie** — strukturovaná stopa tahu (otázka → evidence rozřešení → vystřelené karty → odpověď + assurance) do JSONL; `./jelly triage` shlukuje tahy s miss/nízkou assurance. Provoz pak plní #37 průběžně, místo ručního lovení pastí. | Rozšíření telemetrie karet (měří benefit) i na neúspěchy. Z dialogu 2026-07-19. | 2 |
| 40 | Infra | **Verzovací handshake služeb** — git SHA + čas startu v metadatech Iris REST; web při připojení loguje, při neshodě křičí. Řeší třídu deploy-bolesti „napojeno na starou instanci". | Malé (odpoledne práce). Z dialogu 2026-07-19. | 2 |
| 27 | Chronos | ✅ **HOTOVO**: správa plánu dialogem — „zruš všechno na zítra" (výběr intervalem/hodinou/textem, bez selektoru nutné „všechno"), „posuň všechny ze zítra na čtvrtek" (den v týdnu drží čas dne záznamu), „přeplánuj to ze 17 na 20"; poctivý miss. Karty reminder-cancel/move/manage-miss; weekday_forms v lang. | commit v main. | ✓ |
| 28 | Viz / Paměť | **Okno „Paměť"** místo „Aktivní dokumenty" (zadání): přehled dokumentů VČETNĚ záznamů subsystémů — pojmenování `sub_chronos_<název>`, `sub_topos_<název>`, `sub_mnemos_<název>`; statická znalost `notitia_<název>` (kurátorský korpus). Sklady subsystémů (reminders.jsonl, memory.jsonl, budoucí gazetteer) se v okně ukazují jako dokumenty paměti s aktivací. | Přejmenování okna + doc-id konvence v buildu/skladech; souvisí s #17 (konsolidace data/). | 3 |
| 9 | Viz / detail | Detail uzlu po rozkliknutí: **tvary/aliasy** (`graph.aliases`), kmen; vysvětlit řádky obj/subj (role, ve kterých se uzel účastní). | `viz/detail.py`; čistě prezentační — rychlá výhra, dobrý první úkol. | 4 |
| 25 | Answerer | **Ranking identit** (etalon gap „Kdo je jezis?"→Kristus): šumová spona `být(Ježíš, Bůh)` ze „Syn Boha" přebíjí `jmenovat(Kristus)`/`druh(Mesiáš)`. | Preferuj DATOVOU cestu: „syn Boha" je vztah (genitiv), ne identita — guard v extrakci spony; případně karta (nabídka pater), NE natvrdo v kódu. | 2 |
| 39 | Data | **Provenience faktů** — zdroj + hladina důvěry na každém faktu, hromadný retract podle zdroje („zapomeň všechno z toho hovoru"). Zavést konvencí TEĎ, dokud jsou zapisovatelé dva (korpus, uživatel) — než přibudou #30 Ollama, #7 učení dialogem a STT audio (#42). | Zobecnění dvou bází + provenience-arbitráže homonym; souvisí #17. Z dialogu 2026-07-19. | 4 |
| 5 | Iris karty | **Zbytek**: `clarify-period`/`clarify-relation` karty (potřebují své eventy v turn()); glow-dominantní řazení výčtu po volbě oblasti. | Vzor: jak turn() hlásí `data.overflow` s `area_lit` guardem. | 3 |
| 24 | Mnemos | ✅ **HOTOVO** (2026-07-19): **Negace dějů** — „Prší?" s faktem `neprší(T)` → „Ne, od T neprší." Mechanismus: `_existence` sbírá evidenci OBOU polarit (`negation_prefix` z cs.json), vyhrává NEJNOVĚJŠÍ (datovaná > nedatovaný korpus; remíza → pozdější zápis, deník je chronologický); text šablonou `negative_existence_answer` (jazyk jako data). Parser: negační pár v `_verb_match` s jádrem ≥ 3 znaky — POUČENÍ: šumový predikát „nes" udělal z předložky „s" sloveso a rozbil „Válku s mloky" (etalon regresi chytil okamžitě). Částice (Už) hotové dřív. Scénář negace-deje v dialog benchmarku. | ODBLOKOVÁNO #41 (výjimky dědění lze vyslovit). Zbývá drobnost: inverzní „Neprší?" nad jen-pozitivní evidencí → nenašel (nepáruje zpětně) — řešit s #41. | ✓ |
| 11 | Metron | „Kolikrát letos pršelo?" = díra typu počet-výskytů; zavře gap „Kolik měla dětí BN?". | Tázací tabulka cs.json + počítání faktů (s filtrem #10). | 6 |
| 7 | Mnemos | **Učení pojmů dialogem** — „Co jsou závody aut?" → karta `data-empty` → vysvětlení rozloží extrakční pipeline → fakty do deníku. | KRITICKY PROMYSLET: verzování deníku, zdroj=uživatel, nikdy tiše nepřepsat korpus. | 7 |
| 41 | Graf — koncept | **Oceán vrstev (dědění po druh-hranách)** — aktivace a odpovědi dědí po is-a řetězu s útlumem: „Psi mají rádi maso" → Ronik(pes) dědí slaběji; přímý fakt VŽDY poráží zděděný (defeasible inheritance). Hloubka = vzdálenost v řetězu druh-hran (EMERGENTNÍ, ne kurátorovaná patra). Zárodky už stojí: `druh`/`_is_a`, `_typed_match`, Topos kontejnment (= tatáž mechanika pro prostor), sharpener #13. Dává datový model bodu #7 (učení pojmů). | ZÁVISÍ na #24 (výjimky) a #37 (guardrail). Pasti: strmý útlum, doména/provenience hlídá homonymii (kohout zvíře × vodovodní), nedědit přes homonymní uzly. Koncept user (dialog 2026-07-19, Z1b). | po #24 |
| 8 | Graf — koncept | **Jméno není entita — fáze 2** (spec `2026-07-18-jmenny-uzel-instance.md`): instance per odstavec, rozpuštění dvou-osobových slepenců („Áronovi Mojžíš"), jmenovka jako uzel typu jméno; Toyota s #7. | VELKÉ. Čti spec + měření (otisk identitu nerozliší!); pozor na vakuovou kompatibilitu (případ „Le"). Pozn. (user): extrakční parser by měl mít mírný kontext okolí. | 8 |
| 13 | Sharpener | Cross-distribuce + vyzařování focusu po hranách (kontext hrany slabší); váhy v configu; K-křivka run_focus. | — | 9 |
| 42 | Vize | **Audio kanály** — STT (Vosk/Whisper) jako VSTUPNÍ kanál, TTS (Piper) jako VÝSTUPNÍ — oba přes registr kanálů (vzor console/window/alarm/email), NE jako subsystém. Zásada (dialog 2026-07-19, S1): doménoví experti (Chronos/Topos/Mnemos — srostlí s grafem) ≠ kanálové adaptéry (hrana, nezávislé). STT = zátěžový test tvarosloví (bez interpunkce, chyby rozpoznávání) → otevřít až po #37. | park s tématem audio; fakta ze STT dostanou provenienci (#39). | park |
| 29 | Topos-geo | **Polohové připomínky + mobil** (spec `2026-07-18-topos-geo.md`): „až budu v Kauflandu" — geofence nad skladem připomínek; poloha přes POST /geo (PWA z mobilu — Geolocation API + Web Push), lokace místa učením za pochodu („Jsi teď v X?"), explicitně („X je tady") nebo geokódovacím adaptérem; WhatsApp jako druhý kanál (bez geo). Mobil = senzor + displej, mozek na serveru. | ZAPARKOVÁNO s celým tématem Topos — otevřít až po odzkoušení a potvrzení stávajícího. | park |
| 12 | Topos | Hierarchie míst (Praha ⊂ Čechy), containment, „tady/poblíž". | Jádro HOTOVO v S3 (archiv); zbývá: „poblíž/sousedí" dotazy (near záznamy jsou), místo jako filtr i pro díry, gazetteer editor. | 9 |
| 30 | Topos | **Kartografický adaptér (Ollama)**: lokální LLM přes API plní gazetteer (pseudo-mapu) — generuje kontejnment/sousedství JSONL řádky pro zadaná místa; VŽDY přes validaci (kurátorský filtr / potvrzení dialogem — halucinace). K tomu dotazy „Kde jsou Domažlice?" (odpověď z gazetteeru — rodičovský řetěz) a „Bydlí pan Tetříček v Domažlicích?" (Mnemos fakt + místní filtr, už funguje principem). | Adaptér jako zásuvný modul (vzor: registr kanálů); po dokončení konceptu Topos. | 12 |
| 14 | Čistý řez | UDPipe pryč z query (gate splněn: etalon 28/28 v `--mode templates`), answerer → Iris pluginy, pohrobci → `conserved_`. | Mechanické; po něm zmizí závislost dotazů na ÚFAL službách. POVÝŠENO (dialog 2026-07-19): KARANTÉNA ÚFAL — po řezu zbude závislost jen v anotaci korpusu + Mnemos fallbacku, „vachrlatost" se stane dvěma ohraničenými body. | **2** |
| 2 | Hygiena dat | **Zbytek**: uzel „mle" (NOUN mangle — lemma↔form konzistence), kapitalizované slovesné predikáty (Chvalte), „dovoleno" v pred/attr. | Vzor: hlasování v hygiene.py. | 12 |
| 15 | Coverage | Anaforický kbelík (~2 100 vět se zájmenným podmětem). | — | 13 |
| 16 | Etalon gapy | BN copula-profese; Kolik dětí (→ #11); Kde působila; „Jaka babicka?". | Průběžně s příslušnými body. | 14 |
| 17 | Infra | Konsolidace `data/` (mapa statické × uživatelské znalosti) + README sekce. | Malý úklid. | 15 |
| 21 | Infra | viewBase python testy: chybí httpx2 (6 souborů nekolektuje). | Doinstalovat/upravit testclient. | 16 |
| 19 | Experiment | Hybridní aktivace uzel × hrana. | Metrika → prototyp za flagem. | 17 |
| 20 | Vize | Osobnost/hlas databáze (persona nad Echo). | Závisí na Echo (kompozice). | 18 |

## Otevřené z runtime Mnemos — nevyřešené (2026-07-18, deploy jelly.ithosaudio.eu)

Body, které se za běhu (Mnemos, ne korpus) NEPODAŘILO zlomit. Kořen je společný:
morfologie na IZOLOVANÉM tvaru (mimo větný kontext) je nespolehlivá, a náš
lehký Mnemos parser (`_l_form`/`_finite_verb`) je na hraně.

**Od 2026-07-19 sekci MĚŘÍ zápisový etalon (#37, `benchmark/run_mnemos.py`).**
Baseline odhalil 4 NOVÉ nálezy („Potkal jsem Karla." ztráta; „Sněží." ztráta;
„Můj email" → subjekt „Můj"; „Minulý" v objektech) — **vše opraveno týž den
spolu s #31, #35, katalogovou částí #32 a částicí „Už" z #24: GAP 12/12
opraveno** (řádek #35 se měří jen s `--nom` — bez služeb benchmark poctivě
hlásí 11/1). Jádro #24 (párování predikát↔negace, „Ne, od T neprší") a
OBECNÉ řešení #32 zůstávají otevřené v hlavní tabulce.

| # | Oblast | Problém | Co jsme zkusili / proč to nešlo | Priorita |
|---|--------|---------|--------------------------------|----------|
| 31 | Mnemos parser | ✅ **VYŘEŠENO** (2026-07-19, měřeno #37): výběr l-predikátu preferuje tvary MALÝMI písmeny (skutečné příčestí; kapitalizovaný kandidát jen když jiný není — „Pršelo v Praze"); kapitalizované l-lookalike tvary („Emil", „Karla", „Marcela") zůstávají v objektech (`exclude_l` jen na malá písmena; vylučuje se povrchový ZDROJ predikátu, ne jen tvar po katalogu). Platí i pro ženská jména (Marcela) a bezdiakritické kombinace („Karel bydli" → bydlet). Netknuto: `_l_form` sám — žádné „vlastní pravidlo" tvaru (poučení respektováno). | 5 gap řádků #37 GAP-FIXED; k tomu nálezy „Potkal jsem Karla" (ztráta), „Minulý" v objektech, „Sněží." (karta allow_no_objects), „Můj email" (possessive_words v cs.json) — vše opraveno a měřeno. | ✓ |
| 32 | Mnemos sloveso | **Bezdiakritické tvary sloves** (zmršený/nerozpoznaný predikát). | ŘEŠENO KATALOGY pro známé tvary: `predicate_catalog` (bydl→bydlet, prsi→prší, snezi→sněží) + `finite_verb_forms` (rozpoznání „prsi"/„snezi" jako prézens — koncovka „i" jako pravidlo by viděla sloveso v každém plurálu „psi"). Gap řádky #37 FIXED. OBECNÉ řešení (libovolné bezdiakritické sloveso) zůstává → vlastní lematika (doporučení níže). | 3 |
| 33 | Mnemos typy | **Jména jako „pojem", ne „osoba"** (Honza/Pavla). Mnemos při dějovém výroku typuje objekty natvrdo concept (nedělá NER jako korpus s NameTagem). | Zkusili PROPN z UDPipe → nespolehlivé (Honza mis-tagnut jako ADP, Pavla ok). Vráceno. | 2 |
| 34 | Answerer | **„Kdo bydlí s Karlem?" → uživatel** (theme drift): `theme=uživatel` je na KAŽDÉM faktu 1. osoby → chronicky přehřátý → přebíjí „kdo". | Na čistém memory OK; při nasbírané paměti drift. Souvisí s rankingem (#25) a s tím, že user je theme. | 2 |
| 35 | Mnemos místa | ✅ **VYŘEŠENO** (2026-07-19, měřeno #37 `--nom`): délková pojistka dostala VÝJIMKU `place_nominative_endings` (ice/any) — kratší nominativ se známou toponymní koncovkou projde (Petrovicích→Petrovice, Vodňanech→Vodňany); zmršení (Lhotě→Lhot) dál blokováno. E2E: „Marcela bydlí v Petrovicích." → „Kde bydlí Marcela?" → Petrovice. | Kompromis MÍSTA přes UDPipe / JMÉNA přes morpho trvá; koncovky rozšiřovat katalogem. | ✓ |

### 💡 Doporučení (user): **vlastní lematika / kořenový parser + identifikátor tvaru a slovních druhů**

Morpho (izolovaně) i UDPipe (kontext, ale slévá rody) i naše `_l_form` selhávají
každý jinak. Body #31–#35 mají společnou příčinu: **chybí spolehlivý určovač
tvaru** za běhu. Zvážit VLASTNÍ lehký modul (bez ÚFAL služeb pro Mnemos zápis):

- **kořenový parser / stemmer** české flexe (kmen + koncovka), navázaný na
  `cluster_key` (už existuje pro korpus) — ale s pojistkou rodu/životnosti,
  aby neslil Pavel≠Pavla (kde cluster_key sám selhává);
- **identifikátor slovního druhu a tvaru** (osoba vs. sloveso vs. místo) z
  koncovek + kapitalizace + malé tabulky výjimek (`predicate_catalog` je zárodek);
- explicitní **katalog výjimek** pro známé zmršené tvary (rozšiřovat, ne
  univerzalizovat — univerzální `_l_form` fix rozbil klasifikaci).

Cíl: nominativ jmen i míst, správný typ (osoba/místo/pojem) a čistý predikát
(bydlet) BEZ nespolehlivé morfo/UDPipe magie na izolovaných tvarech.

**Guardrail pro celou tuto sekci: #37 zápisový etalon — nejdřív měření, pak
lematika** (dialog 2026-07-19). NEdělat teď (tamtéž): #19 hybridní aktivace,
#20 Echo/persona, vytrhávání subsystémů do knihoven, S2 velkým třeskem.

| # | Oblast | Bod | Poznámka | Priorita |
|---|--------|-----|----------|----------|
| 46 | Architektura | **Vzorová gramatika** (spec `2026-07-19-vzorova-gramatika.md`, zadání user): JEDEN lexer druhů slov (mechanismus + existující tabulky; token = MNOŽINA hypotéz — „byt"={SPONA?,SUBST}), sekvenční VZORY na kartách (regulární, žádný interpret v JSONu), výběr přes deck.best (týž jako dialog), rekurze POUZE klauzulová (čárky/spojky → táž analýza per klauzule, skládání podmětu). Sjednocuje zápis i dotaz → predikáty se potkají konstrukcí; karty i query.py se ZJEDNODUŠÍ (šablona = karta, ne Python). Migrace 5 fází, každá s guardrails; fáze 2 zavře #44, fáze 2–4 většinu #45, lexer dá typy pro #33. | Inkrementální cesta k V4 bráně [1] uvnitř V3. Po fázích, začít lexerem (extrakce bez změny chování). | **2** |
| 44 | Query šablony | **Chybějící tázací vzory** (živý dialog 2026-07-19): „Kdo <sloveso> v <místě>?" (`build_query` → None; UDPipe fallback fakty paměti nenajde — dějové zápisy nemají subj, osoba je obj/concept — souvisí #33/#34); „Kdo DALŠÍ bydlí…?"; „Čím krmí X Y?" (instrumentálová díra). GAP scénáře kdo-bydli-v-miste + kratke-sloveso-ji v dialog benchmarku (runner umí gap). | Šablona kdo+místo do query.py; role/typ díry u faktů paměti promyslet s #33 (osoba jako obj/concept) a #34 (theme=uživatel se nesmí nabízet). | **2** |
| 45 | Mnemos parser | **Krátká slovesa a homografy** (živý dialog 2026-07-19): dvojznakové „jí/sní" pod délkovým guardem `_finite_verb` (≥3) → „Roník jí stravu." se ZTRATÍ a spadne do dotazu (GUI odpovědělo „pes"!); „byt" ≡ spona „být" po deakcentaci → „V Plzni má pronajatý byt." se zmrší na observation bez bytu; souvětí s čárkou se ztratí celé; vsuvka „To co X jí je Y" nechá „co/jí" v objektech. | 4 gap řádky v zápisovém etalonu (kategorie krátká-slovesa/homograf/souvětí/vsuvka). Krátká slovesa možná katalogem (finite_verb_forms vzor — ALE „jí" je i zájmeno, chce kontext); homograf byt/být chce rozlišit substantivum od spony (pozice ve větě?). Rodina pro vlastní lematiku. | **2** |
| 43 | Mnemos/Iris | ✅ **HOTOVO** (2026-07-19): **clarify-identity** — jméno rozřešené na osobu grafu jen ČÁSTEČNĚ („Emil"→„Emil Filla") se nepřipisuje mlčky: karta `clarify-identity` (event `statement.subject`, rys `inexact_person`) nabídne existující osobu i založení nové. Volba jménem / „ano" (první) / slovo z `new_person_words` (nová osoba); otazník ruší. Mechanismus: `PendingIdentity` ve stavu, `_subject_or_clarify` sdílený oběma cestami zápisu, `_resume_identity` před `_resume_pick`. Přesná shoda a elidovaný podmět (kontext) se nedoptávají — scénář „ano, měl rád knedlíky" nezměněn. E2E přes GUI inbox ✓; scénář identita-podmetu v dialog benchmarku. | Bez karty platí původní chování (rozhodnutí nese karta). Souvisí #8, #33. | ✓ |
| 36 | Viz / infra | **Zabalit font lokálně** do `viewbase/static` — troika-three-text tahá glyfy z `cdn.jsdelivr.net` (unicode-font-resolver) přes `fetch()`; za striktní CSP (`connect-src 'self'`) jsou titulky uzlů neviditelné. Self-host = žádná externí závislost ani nutnost povolovat CDN (důležité pro air-gapped/offline). | Předgenerovat/uložit font(y) do static a resolver přesměrovat na lokální cestu; pak CSP nechat přísnou. | 3 |

## Hotovo (archiv — detaily v git logu)

| # | Bod | Výsledek |
|---|-----|----------|
| 1 | Kanonizace v1+v2 | Pádové clustery + poziční merge; **nominativizace id** (`propn_lemma_votes` + `nominativize`, 343 id: Betlémě→Betlém, Boha+Bůh=1572, Šimona→Šimon); guard vakuové kompatibility („Le"). |
| 2 | NER slepence osob | Pádová shoda jmen (`name_case_agreement`): korpus první, ≥2 jednomyslné hlasy — „Ježíš Martu" (827!), „Masaryka Svatopluk Beneš" pryč. |
| 3 | Sémantické guardy | Životnost (osoba pod neživotným druhem), vztah bez protistrany, uvozovky (66→0), pádový práh. |
| 4 | Iris benefit-výběr | `deck.best` (těsnost → priorita) + telemetrie karet (použití + měřený zisk). |
| 5 | Data-overflow | „Co řekl Ježíš?" → nabídka oblastí, volba → aktivační zúžení. |
| 6 | Mnemos připsané fakty | „ano, měl rád knedlíky." → subjekt z těžiště; rozřešení je zaostření; „Pršelo dnes?"; momentový čas dějů; únik z nabídky otázkou. |
| 8f1 | Instance fáze 1 | Srůst JEN z textového tvrzení „X řečený Y" (+ otisk ≥3): „Ježíše Krista"→„Ježíš"; name_families (169); spec s měřeními. |
| 22 | Assurance v2 | Afinita filtruje soupeře — bez faktu predikátu není alternativa. |
| 23 | Mnemos události | „Venku prší." → fakt; „Prší venku?" → Ano. |
| 27+ | Chronos plánování | Kompletní okruh: zadání s chytrými defaulty (událost v pět → předstih 2 h, dnes → +3 h, zítra → ráno 7:00, příští týden → neděle večer; vždy nabídka změny, přeplánování odkazem „to"), správa (zruš/posuň/přeplánuj, dny v týdnu), výpis plánu intervalem, měsíce a „příští/minulý rok", vlastní vlákno hodin, kanály console/window, okno ⏰ Reminder. |
| S3 | Topos jádro | Gazetteer (seed + diktát + učení za pochodu z „na Barrandově v Praze"), kontejnment s pády/palatalizací/epentezí, místní filtr („Pršelo v Čechách?" → Ano přes Praha ⊂ Čechy), slovesná třída přesunu („Kde putoval Ježíš?" → Kafarnaum), místa v Mnemos výrocích s rolí loc, near záznamy. |
| M+ | Mnemos příkazy | „zapamatuj si / pamatuj / ulož si / zapiš si za uši / nezapomeň" — strukturované → fakty, přísloví → doslovné poznámky; vzpomínání „Co jsem ti řekl včera/dnes/minulý týden?" (memory-recall karty). |
| 10 | Chronos tvrdý filtr | ✅ **HOTOVO** (S2): tvrdý časový filtr — interval z otázky (v 19. století, letos, 21.1.1900) vyřadí fakty s časem mimo něj (`_match`/`_existence`); nedatované fakty zůstávají; explicitní datum je primitivum `resolve_temporal`; answerer má injektovatelný clock synchronizovaný automatem. E2E kovářova kobyla (1900 JE 19. století) v `tests/test_time_filter.py`. |

Trvalé zásady: automat i znalostní báze se rozšiřují **JSON kartami/soubory**,
logika nikdy fixně v kódu; každá změna měřena (etalon/focus/dialog/coverage);
dialog > figly. Podrobně `docs/HANDOVER.md`.
