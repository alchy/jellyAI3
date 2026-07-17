# BACKLOG — otevřené body (živý dokument)

> Aktualizuj při každém uzavření/přidání bodu. Stav ke commitu: viz git log.
> Metriky teď: 397 testů, etalon 27/27 (5 gap), focus 12/12, dialog 9/9 (100 %).

| # | Oblast | Bod | Řešení / poznámka | Fáze | Priorita |
|---|--------|-----|-------------------|------|----------|
| 1 | Kanonizace | ✅ **v1 HOTOVO** (commit 25a6055): kapitalizovaný koncept → person cluster (Ježíš person + aliasy Ježíše/Ježíšova/…), koncovky ého/ova, kanon = nejkratší člen (nominativ), aliasy v exact/ins patrech, plné pokrytí > slepenec. Ježíš/Ježíše/ježíšovo/Jezis → jeden uzel. **Zbývá v2**: morfologická nominativizace id skloněných geo/dílo uzlů („Betlémě"→„Betlém" jako id, ne jen popisek) + jednoslovné↔víceslovné jméno (→ #8: „Ježíše Krista"↔„Ježíš" — gap řádek etalonu). | — | F5 | **1→v2** |
| 2 | Hygiena dat | ✅ **z většiny HOTOVO** (hygiene.py — hlasování upos: −394 účastníků, −353 faktů; hodit/dovoleno-jako-obj/Izaiáš pryč, strukturní predikáty whitelist). **Zbytek**: uzel „mle" (NOUN mangle — potřebuje lemma↔form konzistenci), kapitalizované slovesné predikáty (Chvalte/Chce — kosmetika /schema), „dovoleno" jako pred/attr role (tam ADJ patří — projeví se jen u vadných faktů), NER slepence („Ježíš Martu") → #8. | commit v main. | F5 | 2→zbytek |
| 3 | Hygiena dat | Zbylé guardy z auditu: `_apposition_identities` (114), `_relation_person` (95), `flat` (77, nízký dopad) | Stejné vzory jako `_appos_belongs`/verbální filtr. | F5 | 5 |
| 4 | Iris karty | Benefit-výběr karet (`deck.best`): skórování kandidátek + **měřený nárůst aktivace** po akci; telemetrie karet (použití/zisk, časové razítko Chronos) | run_focus nad dialogovými scénáři jako metrika zisku. | F2 | **3** |
| 5 | Iris karty | Karty `data-overflow` („Co řekl Ježíš?" → nabídka oblastí aktivace), `clarify-period`, `clarify-relation` | Mechanismus hotový (deck + eventy), jen karty + eventy z answereru. | F2 | **3** |
| 6 | Mnemos | **Fakta ke korpusovým entitám z potvrzení** — „Měl KČ rád knedlíky?" → nenašel → „ano, měl rád knedlíky" → subjekt z konverzačního těžiště (pro-drop v Mnemos), fakt `měl-rád(Karel Čapek, knedlíky)` do deníku | Statement karta bez 1. osoby s elipsou subjektu + `_fill_subject` mechanika; deník memory.jsonl UŽ je oddělený od statické báze a merguje se po startu (transparentně) — splňuje požadavek extended memory. | F2/F3 | **4** |
| 7 | Mnemos | **Učení pojmů dialogem** — „Co jsou závody aut?" → karta `data-empty`: „můžeš vysvětlit?" → vysvětlení se rozloží extrakční pipeline (extract_facts nad větou — build-side, UDPipe povoleno) → fakty s tématem do deníku | KRITICKY PROMYSLET (kvalita, zneužití, konflikty se statickou bází — verzování deníku, zdroj=uživatel). Koncept jasný, realizace po 6. | F3 | 6 |
| 8 | Graf — koncept | **Homonymní instance (dva Jan Novákové; Moje vs. Karlova Toyota)** — ZAPARKOVÁNO s konceptem: *jméno není entita*. Jmenný uzel (jmenovka) + instanční uzly spojené hranou „jmenuje se"; instance per provenienční kontext (dokument/odstavec), srůstání jen při překryvu kontextových otisků (sdílené okolí — doc_links/kontext fakty). Toyota = druhový uzel, moje/Karlova = instance přes druh-hranu (`_is_a` už existuje); vlastnosti (modrá/bílá) se váží VŽDY na instanci, nikdy na druh. Žádné vícerozměrné rozšíření není potřeba. | Dotaz na jmenovku → focus-offer instancí (mechanismus hotový). Realizace = build-side změna vzniku uzlů; promyslet migraci. | F5+ | 7 (park) |
| 9 | Viz / detail | Detail uzlu po rozkliknutí: doplnit **tvary/aliasy a kmen (ocas)**; popsat řádky obj/subj (= vazby uzlu ve faktech: u „nenastat" obj: zemětřesení, subj: konec — role, ve kterých se uzel účastní) | `viz/detail.py` — přidat řádky „tvary" (graph.aliases) a „kmen"; přejmenovat/vysvětlit popisky rolí. | teď | **2b** (rychlá výhra) |
| 10 | Chronos | Interval jako tvrdý filtr odpovědi + E2E kovářova kobyla do etalonu; run_focus s časem | — | F2 | 8 |
| 11 | Metron | „Kolikrát letos pršelo?" = počítání výskytů faktů; zavře gap „Kolik měla dětí BN?" | Nová díra typu počet-výskytů + jazyková tabulka „kolikrát". | F2/F4 | 9 |
| 12 | Topos | Hierarchie míst (Praha ⊂ Čechy), containment jako intervaly, „tady/poblíž" | Paralela Chronosu. | F4 | 10 |
| 13 | Sharpener | Cross-distribuce + vyzařování focusu po hranách; váhy v configu; K-křivka run_focus | — | F4 | 11 |
| 14 | Čistý řez | UDPipe pryč z query (gate splněn), `graph_answerer.py` → Iris pluginy, pohrobci → `conserved_` | — | F5 | 12 |
| 15 | Coverage | Anaforický kbelík (2 136 vět se zájmenným podmětem) | — | F5+ | 13 |
| 16 | Etalon gapy | BN copula-profese; Kolik dětí (→ Metron); Kde působila; „Jaka babicka?" | — | průběžně | 14 |
| 17 | Infra | Konsolidace souborové struktury dat: `data/` (graph.pkl, annotations.pkl, memory.jsonl, budoucí extended knowledge) — jasná mapa statické × uživatelské znalosti | Malý úklid + README sekce. | F3 | 15 |
| 18 | Dokumentace | **Revize dokumentace agenty** — porovnat docs vs. stav, přizpůsobit, findings jako review | Spuštěno (agent). | teď | běží |
| 19 | Experiment | Hybridní aktivace uzel × hrana | Metrika → prototyp za flagem. | F6 | 16 |
| 20 | Vize | Osobnost/hlas databáze (persona nad Echo) | Far-away; závisí na Echo. | pozdější | 17 |
| 21 | Infra | viewBase python testy: chybí httpx2 (6 souborů nekolektuje) | Doinstalovat/upravit testclient. | údržba | 18 |

| 22 | Assurance v2 | ✅ HOTOVO: afinita filtruje soupeře (soupeř bez faktu predikátu není alternativa) — „Kde se narodil Jezis?" odpovídá rovnou, identitní dialogy zůstávají. | commit v main. | F2 | ✓ |
| 23 | Mnemos | ✅ HOTOVO: rys `finite_verb` + karta `statement-event` — „Venku prší." se ukládá, „Prší venku?" → Ano; veto sloves přes doslovná slova uzlů. | commit v main. | F2 | ✓ |

Trvalé zásady: stavový automat i znalostní báze se rozšiřují **json kartami/soubory**, logika nikdy fixně v kódu; každá změna měřena (etalon/focus/dialog/coverage).
