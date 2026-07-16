# Fáze B1 — Vzdálenostní jádro (návrh)

**Datum:** 2026-07-16 · **Větev:** `feature/vb1-distance-kernel`
**Status:** Schváleno (design), čeká na spec-review uživatelem

## 1. Cíl a kontext

V4a ukázala, že pravidlový (sponový) answerer umí složit čistou odpověď, **jen když
retrieval vytáhne správnou větu do horní pasáže**. Dnešní retrieval hledá v pevných
blocích (`chunker`: okna po `size` větách s překryvem). Když definiční věta padne
nešikovně na hranici bloku nebo se „utopí" ve víceté pasáži, propadne to na
extraktivní.

B1 zavádí **retrieval na úrovni vět s vzdálenostním útlumem**: každá věta shodná
s dotazem vyzařuje své skóre do okolí, útlum klesá se vzdáleností (sever i jih),
**soubor je tvrdá hranice**. Vrchol aktivace určí **ostřicí okno** — dynamicky
sestavenou pasáž kolem sémanticky nejrelevantnější věty. Nezávisí to na kvalitě
odstavců.

**Poctivé omezení:** retrieval nevymyslí fakt, který v korpusu není (korpus nemá
„Rossum je/byl…"). B1 zlepšuje **robustnost a zaostření** tam, kde definice existuje
(např. „Josef Čapek byl český malíř, grafik a spisovatel").

## 2. Klíčová rozhodnutí (schválená)

- **Exponenciální útlum** `decay(d) = exp(−d/τ)`. („Geometrický `γ^d`" je jen
  reparametrizace téže křivky; gaussovský/trojúhelníkový tvar zamítnut — exponenciála
  má jeden parametr a hladce se ladí.)
- **Samostatná třída `SentenceRetriever`.** V1 `Retriever` zůstává čitelný jako
  učební blok. Nová třída ho **znovu použije jako vnitřní skórovač** (viz §4), takže
  se BM25 neduplikuje a chování V1 se nemění.
- **Přepis anotací na větnou granularitu.** `annotate.py` bude klíčovat anotace
  `(doc_id, index věty)` místo `(doc_id, index pasáže)`. Tím se anotace odpojí od
  granularity retrievalu a `TemplateAnswerer` si složí anotaci pro libovolnou pasáž
  (chunkerové i ostřicí okno) z rozsahu jejích vět.

## 3. Architektura a datový tok

```
documents ──► SentenceRetriever.build
                 │  (rozdělí na věty, postaví vnitřní Retriever nad 1větnými pasážemi)
                 ▼
dotaz ──► search: base = vnitřní Retriever.score_all(dotaz)
                 │  final(s) = Σ okolí téhož souboru · exp(−d/τ)   [§4]
                 │  vrchol → ostřicí okno (Passage) [§5]
                 ▼
          [(Passage, score)] ──► Answerer (beze změny rozhraní)
                                    │
                 TemplateAnswerer: složí anotaci okna z (doc_id, idx věty) [§6]
```

## 4. `SentenceRetriever` (nový soubor `jellyai/sentence_retriever.py`)

**Stav po `build(documents)`:**
- Pro každý dokument `split_sentences(doc.text)` → věty s **lokálním indexem v rámci
  dokumentu** (0…). Věty se ukládají v pořadí dokument po dokumentu, takže věty téhož
  souboru jsou souvislé.
- Paralelní pole: `sent_doc[k]` (doc_id), `sent_local[k]` (index věty v dokumentu),
  `sent_text[k]`.
- **Vnitřní `Retriever`** postavený nad 1větnými pasážemi
  `Passage(doc_id, local_idx, věta, local_idx, local_idx+1)` — poskytuje BM25 skóre.

**`Retriever.score_all(query) -> np.ndarray`** (aditivní metoda na `Retriever`):
vrátí surové skóre pro **všechny** pasáže podle nakonfigurované metody (bez ořezu
`top_k`). Existující `search` se nemění.

**`search(query, top_k=None) -> list[tuple[Passage, float]]`:**
1. `base = internal.score_all(query)` (délka = počet vět).
2. `final[k] = Σ_{k' : sent_doc[k']==sent_doc[k]} base[k'] · exp(−|sent_local[k]−sent_local[k']|/τ)`.
   Napříč soubory příspěvek 0 (tvrdá hranice).
3. Seřaď věty podle `final` sestupně; ber vrcholy hladově a **přeskoč ty, které už
   leží v dřívějším okně** (aby `top_k` okna byla různá), dokud není `top_k` oken.
4. Pro každý vrchol `p` (lokální index `L`, dokument s lokálním rozsahem `[0..d]`):
   okno = věty `[max(0, L−r) … min(d, L+r)]`, `Passage(doc_id, index=L, text=join,
   start=první lok. index, end=poslední+1)`. Skóre okna = `final` vrcholu.

**Parametry z configu (§8):** `τ = decay_tau`, `r = focus_radius`.

## 5. Ostřicí okno jako `Passage`

Okno je běžná `Passage`, takže **rozhraní `search()` zůstává `[(Passage, score)]`** a
žádný answerer se nemění. Důležité: `start`/`end` okna jsou **lokální indexy vět
v dokumentu** — stejná soustava jako u `chunker` (ten také dělí per dokument). Díky
tomu §6 funguje pro chunkerová i ostřicí okna identicky.

## 6. Větné anotace (`jellyai/annotate.py` — přepis)

**`annotate_documents(documents, client) -> dict`:**
- Klíč: `(doc_id, index věty v dokumentu)`.
- Hodnota (stejný **tvar** jako dnes, ať `selection.py` funguje beze změny):
  `{"entities": [...], "sentences": [[token, …]]}` — právě jedna věta.
- **Offsety v rámci dokumentu:** při anotaci se každé větě přičte `base` = běžící součet
  délek předchozích vět (+1 mezera). Token i entity dané věty se posunou o týž `base`.
  Tím jsou offsety napříč větami **disjunktní a konzistentní** → po složení okna
  nedojde k záměně entity z jedné věty za token v druhé.
- `save_annotations` / `load_annotations` (pickle) — beze změny formátu volání.

**Regenerace dat:** formát se mění, takže `./jelly annotate` je nutné spustit znovu
(na rozdíl od V4a). Zdokumentovat v README.

## 7. Napojení `TemplateAnswerer`

Dnes: `annotations.get((doc_id, passage.index))` → jedna anotace. Nově answerer
**složí** anotaci z větného úložiště přes rozsah pasáže:

```
def _annotation_for(passage):
    sents, ents = [], []
    for i in range(passage.start, passage.end):
        a = annotations.get((passage.doc_id, i))
        if not a: continue
        sents += a["sentences"]          # každá věta = 1 seznam tokenů
        ents  += a["entities"]           # offsety už disjunktní (§6)
    return {"entities": ents, "sentences": sents} if sents else None
```

Zbytek (`analyze_question`, `select_answer`, `_render`) beze změny. Funguje pro
chunkerová okna (mode template dnes) i pro ostřicí okna (B1).

## 8. Konfigurace (`config.py`)

`RetrieverConfig` získá:
- `granularity: str = "passage"` — `"passage"` (V1) nebo `"sentence"` (B1).
- `decay_tau: float = 1.6` — dosah útlumu.
- `focus_radius: int = 2` — poloměr ostřicího okna (počet vět na každou stranu).

## 9. Integrace do pipeline a CLI

- `pipeline._build_retriever(config)`: podle `granularity` postaví `Retriever`
  (passage) nebo `SentenceRetriever` (sentence). `from_corpus`/`from_index` ho použijí.
- `SentenceRetriever.save/load` (pickle celého stavu) — aby `index`/`repl` fungovaly
  i pro sentence režim.
- CLI `annotate` → `annotate_documents` (načte dokumenty, ne pasáže).
- CLI `index` uloží retriever podle `granularity`.

## 10. Testy

**Hermetické (bez modelů):**
- `Retriever.score_all`: délka = počet pasáží; pořadí shodné se `search`.
- Vzdálenostní jádro: nad vnucenými `base` skóre ověř exp útlum, **vrchol**, **tvrdou
  hranici souboru** (věta v jiném souboru nepřispěje), ořez okna na hranice dokumentu.
- `annotate_documents` (FakeUfalClient): klíče per věta; offsety posunuté a disjunktní.
- `TemplateAnswerer._annotation_for`: okno přes 2 věty se složí; spona v jedné z nich
  → `select_predicate` vrátí přísudek.

**End-to-end (FakeUfalClient / malý korpus):**
- `SentenceRetriever` nad 2 dokumenty: dotaz zaostří na správnou větu; okno ji obsahuje.
- Pipeline `granularity="sentence"` + template: sponová otázka nad zanořenou definicí
  → přísudek (ne fallback).

## 11. Ošetření chyb a hraniční případy

- Prázdný dokument / žádná věta → do indexu nepřidat; `search` na prázdném indexu → `[]`.
- `top_k` větší než počet různých vrcholů → vrať, kolik jde.
- Vrchol u kraje dokumentu → okno se ořízne na `[0..d]` (žádné přetečení do sousedního
  souboru).
- Ostřicí okno bez jakékoli anotace (např. neanotovaný dokument) → `_annotation_for`
  vrátí `None` → template spadne na extraktivní (dnešní chování).
- `decay_tau ≤ 0` → pojistně `max(τ, 1e-6)`.

## 12. Mimo rozsah (YAGNI)

- **B2 konverzační paměť** (stav napříč dotazy v `repl`, zahřívání/dohasínání témat).
  Sdílí metaforu aktivace, ale je to samostatný spec.
- Předpočítané agregované anotace oken (větné anotace stačí; agregace se skládá levně).
- Jiné tvary jádra (gaussovský/trojúhelníkový) — zamítnuto v §2.
