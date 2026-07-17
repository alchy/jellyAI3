# BACKLOG — otevřené body (živý dokument)

> Aktualizuj při každém uzavření/přidání bodu. Stav ke commitu: viz git log.
> Metriky teď: **442 testů, etalon 28/28 (+5 gap), focus 12/12, dialog 21/21 — 100 %.**
>
> **➡️ PŘEDÁNÍ PRÁCE: čti nejdřív `docs/HANDOVER.md`** — zákony projektu,
> testovací smyčka, implementační tipy ke každému otevřenému bodu, pasti.

## Otevřené (v pořadí priority)

| # | Oblast | Bod | Řešení / poznámka | Priorita |
|---|--------|-----|-------------------|----------|
| 26 | Subsystémy | **Refaktor na společný půdorys** — spec `2026-07-18-subsystemy-iris.md`: port Subsystem (poznání/kanonizace/aktivace/záznamy), tři brány (E extrakce, Q nárok v parseru, A reflektor), fáze S0–S5. Obsahuje Chronos v2 (reminder-default „za čtvrt hodiny" + přeplánování, day_parts ráno=7:00 + učení uživatelem, registr kanálů console/window/alarm/email, okno Reminder), Topos v1 (gazetteer, kontejnment, slovesná třída pohybu — „Kde putoval Ježíš?"), Mnemos jako brána zápisu pro subsystémy. | Fáze po jedné, každá měřená; S0 mechanická. STAV: S0+S1+S2-jádro HOTOVO (subsystems balíček, day_parts, reminder-default+přeplánování, okno Reminder+kanály, tvrdý filtr); zbývá S2-dočištění (formální claim() v parseru), S3 Topos, S4 brána E, S5 učení definic. | **1** |
| 9 | Viz / detail | Detail uzlu po rozkliknutí: **tvary/aliasy** (`graph.aliases`), kmen; vysvětlit řádky obj/subj (role, ve kterých se uzel účastní). | `viz/detail.py`; čistě prezentační — rychlá výhra, dobrý první úkol. | 2 |
| 25 | Answerer | **Ranking identit** (etalon gap „Kdo je jezis?"→Kristus): šumová spona `být(Ježíš, Bůh)` ze „Syn Boha" přebíjí `jmenovat(Kristus)`/`druh(Mesiáš)`. | Preferuj DATOVOU cestu: „syn Boha" je vztah (genitiv), ne identita — guard v extrakci spony; případně karta (nabídka pater), NE natvrdo v kódu. | 2 |
| 5 | Iris karty | **Zbytek**: `clarify-period`/`clarify-relation` karty (potřebují své eventy v turn()); glow-dominantní řazení výčtu po volbě oblasti. | Vzor: jak turn() hlásí `data.overflow` s `area_lit` guardem. | 3 |
| 24 | Mnemos | **Negace dějů** — „Prší?" s faktem `neprší(čas T)` → „Ne, od T neprší" (negovaný fakt je evidence opaku); „Už" se do objektů nemá ukládat. | Negační prefix do cs.json; párování predikát↔negace mechanismem, text kartou; promyslet s #10. | 5 |
| 11 | Metron | „Kolikrát letos pršelo?" = díra typu počet-výskytů; zavře gap „Kolik měla dětí BN?". | Tázací tabulka cs.json + počítání faktů (s filtrem #10). | 6 |
| 7 | Mnemos | **Učení pojmů dialogem** — „Co jsou závody aut?" → karta `data-empty` → vysvětlení rozloží extrakční pipeline → fakty do deníku. | KRITICKY PROMYSLET: verzování deníku, zdroj=uživatel, nikdy tiše nepřepsat korpus. | 7 |
| 8 | Graf — koncept | **Jméno není entita — fáze 2** (spec `2026-07-18-jmenny-uzel-instance.md`): instance per odstavec, rozpuštění dvou-osobových slepenců („Áronovi Mojžíš"), jmenovka jako uzel typu jméno; Toyota s #7. | VELKÉ. Čti spec + měření (otisk identitu nerozliší!); pozor na vakuovou kompatibilitu (případ „Le"). Pozn. (user): extrakční parser by měl mít mírný kontext okolí. | 8 |
| 13 | Sharpener | Cross-distribuce + vyzařování focusu po hranách (kontext hrany slabší); váhy v configu; K-křivka run_focus. | — | 9 |
| 12 | Topos | Hierarchie míst (Praha ⊂ Čechy), containment, „tady/poblíž". | Paralela Chronosu; tabulky v cs.json. | 10 |
| 14 | Čistý řez | UDPipe pryč z query (gate splněn: etalon 28/28 v `--mode templates`), answerer → Iris pluginy, pohrobci → `conserved_`. | Mechanické; po něm zmizí závislost dotazů na ÚFAL službách. | 11 |
| 2 | Hygiena dat | **Zbytek**: uzel „mle" (NOUN mangle — lemma↔form konzistence), kapitalizované slovesné predikáty (Chvalte), „dovoleno" v pred/attr. | Vzor: hlasování v hygiene.py. | 12 |
| 15 | Coverage | Anaforický kbelík (~2 100 vět se zájmenným podmětem). | — | 13 |
| 16 | Etalon gapy | BN copula-profese; Kolik dětí (→ #11); Kde působila; „Jaka babicka?". | Průběžně s příslušnými body. | 14 |
| 17 | Infra | Konsolidace `data/` (mapa statické × uživatelské znalosti) + README sekce. | Malý úklid. | 15 |
| 21 | Infra | viewBase python testy: chybí httpx2 (6 souborů nekolektuje). | Doinstalovat/upravit testclient. | 16 |
| 19 | Experiment | Hybridní aktivace uzel × hrana. | Metrika → prototyp za flagem. | 17 |
| 20 | Vize | Osobnost/hlas databáze (persona nad Echo). | Závisí na Echo (kompozice). | 18 |

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
| 10 | Chronos tvrdý filtr | ✅ **HOTOVO** (S2): tvrdý časový filtr — interval z otázky (v 19. století, letos, 21.1.1900) vyřadí fakty s časem mimo něj (`_match`/`_existence`); nedatované fakty zůstávají; explicitní datum je primitivum `resolve_temporal`; answerer má injektovatelný clock synchronizovaný automatem. E2E kovářova kobyla (1900 JE 19. století) v `tests/test_time_filter.py`. |

Trvalé zásady: automat i znalostní báze se rozšiřují **JSON kartami/soubory**,
logika nikdy fixně v kódu; každá změna měřena (etalon/focus/dialog/coverage);
dialog > figly. Podrobně `docs/HANDOVER.md`.
