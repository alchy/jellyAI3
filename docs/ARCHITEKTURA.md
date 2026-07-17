# ARCHITEKTURA jellyAI3 — nalejvárna (stav 2026-07-18)

Orientační mapa celku pro sladění. Detaily: HANDOVER.md (jak pracovat),
BACKLOG.md (co dál), specs/ (proč to tak je).

## 1. Celek na jeden pohled

```
                 ┌──────────────── BUILD (offline) ────────────────┐
 data/raw/*.txt ─► index → annotate (UDPipe+NameTag) → extrakce faktů
                   → HYGIENA (korpusová hlasování) → kanonizace
                   → nominativizace → instanční vrstva → data/graph.pkl
                 └─────────────────────────────────────────────────┘
                                        │
                              FAKTOVÝ GRAF (jeden!)
                     korpusová fakta + uživatelská fakta (s časem)
                                        ▲
 ┌────────── DIALOG (online) ───────────┼──────────────────────────┐
 │ uživatel ► web(:8080, pasivní) ► REST Iris(:8084) ► IrisAutomaton │
 │                                                                  │
 │  turn(): hodiny? → volba? → konstatování (Mnemos) → připomínka   │
 │          (Chronos) → dotaz na plán → pseudo-QL → match → KARTY   │
 │                                                                  │
 │  ChronosTicker (vlastní vlákno) ──► kanály: konzole / okno ⏰    │
 └──────────────────────────────────────────────────────────────────┘
```

## 2. Vrstvy a jejich zákony

1. **Graf** (`jellyai/graph/`) — jediná znalostní báze. Fakt = uzel
   (predikát + váha) s role-hranami (subj/obj/loc/time/num/pred/attr/
   theme). Korpusový i uživatelský obsah žijí SPOLU; uživatelská fakta
   mají navíc timestamp dialogu. Kvalitu drží **korpusová hlasování**
   (hygiene.py): lokální tag jedné věty nikdy nerozhoduje.
2. **Answerer** (`jellyai/answerer/`) — mechanika odpovědi: pseudo-QL
   šablony (query.py, bez UDPipe) → match nad grafem → ranking
   vahou + AKTIVACÍ. `ActivationField` (jas uzlů) je jediné médium,
   kterým cokoli ovlivňuje odpověď. Tvrdý časový filtr (time_filter)
   vyřazuje fakty mimo interval otázky.
3. **Iris** (`jellyai/iris/`) — dirigent dialogu. Stavový automat, jehož
   CHOVÁNÍ nesou JSON karty (`patterns/cs/` — trigger→dialog→akce→teach):
   nabídky zaostření, upřímné terminály, paměť, připomínky. Kód = jen
   mechanismy; nový vzor chování = nová karta. QueryAssurance měří
   jistotu; pod prahem karty se automat PTÁ (dialog > figly).
4. **Subsystémy** (`jellyai/iris/subsystems/`) — doménoví experti nad
   osami reality, společný půdorys (spec 2026-07-18-subsystemy-iris.md):
   POZNÁNÍ (vzory) / KANONIZACE (jazyk↔záznam) / AKTIVACE (reflektor
   do pole) / ZÁZNAMY (JSONL sklady). Brány: E (extrakce), Q (nárok na
   tokeny otázky), A (osvětlení grafu).
   - **Chronos** (čas): intervaly („letos", „v 19. století", „21.1.1900"),
     hodinové odpovědi, day_parts (ráno=7:00), PŘIPOMÍNKY (sklad
     reminders.jsonl, vlastní vlákno hodin, kanály konzole/okno,
     výpis plánu zúžený intervalem), tvrdý filtr odpovědí.
   - **Mnemos** (paměť uživatele): konstatování → fakta do TÉHOŽ grafu
     s časovou kotvou (deník memory.jsonl, replay po startu); připsané
     fakty korpusovým osobám z potvrzení.
   - **Topos** (prostor, plán S3): kontejnment míst (Praha ⊂ Čechy) —
     zrcadlo Chronos intervalů na ose prostoru.
5. **Jazyk** (`jellyai/lang/cs.json`) — VŠECHNA čeština jako data:
   tázací slova, spony, časové tabulky, fráze připomínek… Nový jazyk =
   nový JSON.
6. **Prezentace** — REST služba (`services/iris_service.py`, :8084) je
   jediný vstup; web (`./jelly web`, :8080, viewBase) je PASIVNÍ displej:
   konzole, ⚡ aktivační okno, 📄 dokumenty, ⏰ okna Reminder (push od
   Chronosu přes REST event).

## 3. Tok jednoho tahu (co se stane s otázkou)

1. `turn()`: odpálí dozrálé připomínky (předřadí odpovědi).
2. Hodinová otázka? → Chronos odpoví sám. Volba z nabídky? → přehraje
   zaostřenou otázku. Konstatování? → Mnemos (karty určí druh) → fakt.
   Žádost o připomenutí / dotaz na plán? → Chronos.
3. Jinak: Chronos rozsvítí časové uzly intervalu otázky (soft) a nastaví
   tvrdý filtr → pseudo-QL přeloží otázku na pattern (entity vs. slovesa
   rozhoduje slovník grafu) → `_resolve_topic` rozřeší jména (patra
   evidence, aliasy, domény) → match vyplní díru → ranking jasem.
4. KARTY rozhodnou výsledek: jistá odpověď / nabídka kandidátů
   (homonyma, přetékající výčty) / upřímné „nepodařilo se zaostřit".
5. Vše se vrací s metadaty (assurance, použité karty, aktivační okno) —
   web jen zobrazuje.

## 4. Kde co leží (soubory dat)

| Co | Kde | Povaha |
|---|---|---|
| korpus | `data/raw/*.txt` → `data/processed` | statická znalost |
| anotace | `data/annotations.pkl` | build meziprodukt |
| graf | `data/graph.pkl` | jediná báze (korpus+uživatel) |
| deník Mnemos | `data/memory.jsonl` | trvalá paměť uživatele |
| sklad Chronos | `data/reminders.jsonl` | krátkodobá (odpálené mizí) |
| karty chování | `jellyai/iris/patterns/cs/*.json` | chování Iris |
| jazyk | `jellyai/lang/cs.json` | veškerá čeština |

Plán (BACKLOG #28): okno „Paměť" místo „Aktivní dokumenty" — sklady
subsystémů jako dokumenty `sub_chronos_*`, `sub_topos_*`, `sub_mnemos_*`,
statická znalost `notitia_*`.

## 5. Neporušitelné principy (proč to drží pohromadě)

- **Karty, ne kód** — chování se rozšiřuje JSONem; kód jen mechanismy.
- **Dialog > figly** — nejistota se řeší otázkou, ne heuristikou.
- **Jeden graf, jedno pole** — subsystémy ovlivňují odpovědi VÝHRADNĚ
  aktivací (reflektory) a filtry z bran; žádné boční kanály.
- **Korpusová evidence** — hlasování přes celek poráží lokální tag.
- **Vše měřeno** — 445 testů + 4 benchmarky (etalon 28/28, focus 12/12,
  dialog 21/21, coverage) jsou brány každé změny.
