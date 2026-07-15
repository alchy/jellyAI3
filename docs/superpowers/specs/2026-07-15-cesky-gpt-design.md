# Design: jellyAI3 → výuková knihovna pro QA nad českými texty

**Datum:** 2026-07-15
**Autor:** Jindřich Němec + Claude
**Status:** Schváleno (V1 detailně, V2 roadmapa)

## 1. Cíl a kontext

Původní jellyAI3 je slovní LSTM model pro predikci dalšího slova (trénovaný na Čapkově
R.U.R.), který dává nekvalitní výstup a zahazuje diakritiku, interpunkci i velikost písmen.

Skutečný cíl uživatele: **odpovídat na jednoduché dotazy o obsahu textu** — např.
„kdo vyráběl roboty?", „co znamená R.U.R.?". To je úloha *question answering (QA)*,
ne generování textu. Klíčové zjištění brainstormingu: malý generativní model trénovaný
od nuly na pár MB textu tuhle úlohu neřeší — fakta se musí brát **z textu přes retrieval**,
ne z „paměti" modelu.

Projekt proto stavíme jako **výukovou knihovnu skládatelných bloků** pro retrieval-based QA
nad načtenými českými texty. Uživatel skládá a parametrizuje řešení z bloků jako stavebnici,
každý blok má jasné rozhraní a vysvětlení „co dělá a proč", bez nutnosti vidět nejhlubší
matematické detaily uvnitř.

**Rozsah dotazů:** nejdřív výhradně o načtených textech (closed-domain). Rozšíření na obecné
znalosti je „pak uvidíme" (mimo rozsah teď).

## 2. Klíčová rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Typ úlohy | Retrieval-based QA nad načtenými texty (closed-domain) |
| Forma | Skládatelná výuková knihovna bloků s jasnými rozhraními |
| Retrieval | TF-IDF / BM25, **psané od nuly v numpy, žádný stažený model** |
| Answerer | Vyměnitelný blok; V1 extraktivní + šablona, V2 generátor od nuly |
| Data | Kurátorovaný korpus českých public-domain knih (jednotky až nižší desítky MB) |
| Čeština | Zachovat diakritiku, interpunkci i velikost písmen; jen lehké čištění balastu |
| Závislosti | **Bez externích služeb** (žádná Ollama, žádné API). Minimum: numpy, requests, pytest (V1); torch, sentencepiece (V2) |
| Běh | Plně lokálně a offline (po stažení dat); trénink V2 na MPS (M4 Pro) |
| Prostředí | venv, **Python 3.11 nebo 3.12** (novější verze mají problém s torch) |
| Repo | Rozvinout přímo v repu `jellyAI3`; LSTM verze zůstává dohledatelná v git historii |
| Postup | Ve vrstvách: V1 (retrieval + extraktivní QA) → V2 (generátor od nuly) |

## 3. Architektura: skládatelné bloky

Pipeline od dotazu k odpovědi. Každý blok = jeden účel, jasné rozhraní, `explain()` popis.

```
Loader → Chunker → Retriever → Answerer → (Answer)
                       ▲            ▲
                   (index)     vyměnitelný blok
```

| Blok | Co dělá | Parametry | Vrstva |
|---|---|---|---|
| Loader | načte vyčištěné texty do `Document` | zdroje | V1 |
| Chunker | rozseká `Document` na `Passage` | velikost, překryv, jednotka | V1 |
| Retriever | index + hledání top-k pasáží (TF-IDF/BM25) | metoda, top-k | V1 |
| Answerer (extraktivní) | z pasáží složí odpověď (úsek + šablona) | typ šablony | V1 |
| Pipeline | pospojuje bloky: `otázka → Answer` | složení | V1 |
| QA-gen | syntetické `(otázka, kontext, odpověď)` páry z korpusu | strategie maskování | V2 |
| Answerer (generativní) | malý seq2seq trénovaný na QA párech | model, sampling | V2 |

Oba `Answerer`y sdílejí jedno rozhraní, takže jsou zaměnitelné a lze přidat další (např.
pro obecné znalosti) beze změny zbytku pipeline.

## 4. Struktura repozitáře

```
config.py                 # dataclasses s konfigurací všech bloků
data/
  download.py             # stáhne kurátorovaný seznam public-domain českých knih
  clean.py                # lehké čištění (patičky, kódování, bílé znaky); zachová diakritiku
  raw/  processed/        # gitignorováno
jellyai/                  # KNIHOVNA (skládatelné bloky)
  loader.py               # load processed texty → Document
  chunker.py              # Document → list[Passage]
  retriever.py            # TF-IDF / BM25 index + search (od nuly, numpy)
  pipeline.py             # QAPipeline: otázka → Answer
  explain.py              # výuková vrstva: bloky popisují, co dělají
  answerer/
    base.py               # rozhraní Answerer + datové typy Answer
    extractive.py         # extraktivní + šablonová odpověď (V1)
    generative.py         # obal seq2seq generátoru (V2)
qagen/  synth.py          # V2: syntetické QA páry
model/                    # V2: generátor od nuly
  tokenizer.py seq2seq.py train.py generate.py
tests/                    # pytest
cli.py                    # prepare-data | build-index | ask | explain | (V2: gen-qa | train-gen)
requirements.txt          # (requirements-v2.txt pro torch/sentencepiece)
README.md                 # setup venv (py 3.11/3.12) + postup krok za krokem
```

## 5. V1 — komponenty (detailně)

### 5.1 Data (`data/download.py`, `data/clean.py`)
- `download.py`: stáhne knihy podle seznamu `(titul, url)` v configu (Project Gutenberg,
  Wikizdroje). Seznam je kurátorovaný a rozšiřitelný. Přesné URL se ověří při implementaci
  (dostupnost + public-domain licence). Cíl: jednotky až nižší desítky MB čistého textu.
- `clean.py`: odstraní licenční hlavičky/patičky (Gutenberg boilerplate), sjednotí kódování
  na UTF-8, normalizuje bílé znaky a konce řádků. **Zachová diakritiku, interpunkci
  a velikost písmen.** Výstup do `data/processed/`.

### 5.2 Loader (`jellyai/loader.py`)
- `load_documents(dir) -> list[Document]`, kde `Document(id, title, text)`.
- Načítá už vyčištěné texty z `data/processed/`.

### 5.3 Chunker (`jellyai/chunker.py`)
- `chunk(document, config) -> list[Passage]`; `Passage(doc_id, index, text, start, end)`.
- Sliding window s překryvem; jednotka = věty (výchozí, věta-aware) nebo znaky.
- Parametry: `size`, `overlap`, `unit`. `explain()` popíše princip a vliv parametrů.

### 5.4 Retriever (`jellyai/retriever.py`) — jádro učení
- Psaný **od nuly v numpy**, žádný stažený model.
- Jednoduchá česká tokenizace pro retrieval: lowercasing, rozdělení na slova, zachování
  diakritiky (volitelně lehká normalizace).
- `build(passages)`: postaví slovník + matici vah (TF-IDF) nebo statistiky (BM25).
- `search(query, top_k) -> list[(Passage, score)]`.
- Dvě metody na výběr (`method="tfidf"|"bm25"`) — BM25 jako názorný kontrast k TF-IDF.
- `explain()` vysvětlí skórování (proč vzácná slova váží víc, role délky dokumentu).

### 5.5 Answerer — extraktivní (`jellyai/answerer/extractive.py`)
- Rozhraní (`base.py`): `answer(question, passages) -> Answer`;
  `Answer(text, sources, score)`.
- Vybere z top pasáží nejrelevantnější větu/úsek (skórování překryvu s dotazem).
- Volitelné šablonování podle typu otázky (kdo/co/kde/kdy) pro čitelnější odpověď.
- Vždy vrací **zdroj** (z které pasáže/dokumentu), aby byla odpověď dohledatelná.

### 5.6 Pipeline (`jellyai/pipeline.py`)
- `QAPipeline(retriever, answerer)`; `ask(question) -> Answer`.
- Index se staví z vyčištěného korpusu (malý → stavba v paměti při startu je dostačující;
  volitelné cachování indexu na disk).

### 5.7 Výuková vrstva (`jellyai/explain.py`)
- Každý blok umí `explain()` — lidsky čitelný popis účelu, vstupů/výstupů a parametrů.
- CLI `explain <blok>` to zobrazí. Cíl: uživatel chápe stavebnici, aniž řeší vnitřní matiku.

### 5.8 CLI (`cli.py`)
- `prepare-data` → download + clean
- `build-index` → load + chunk + postavení retrieveru (volitelně uložení)
- `ask "otázka"` → retrieve + answer, vypíše odpověď + zdroj
- `explain <blok>` → popis bloku

## 6. Tok dat (V1)

```
public-domain URL → data/raw → clean → data/processed
   → Loader → Chunker → Retriever.build(passages)
   → [dotaz] → Retriever.search → Answerer.answer → Answer(text, source)
```

## 7. Ošetření chyb a hraniční případy (V1)

- Prázdný/neexistující `data/processed` → srozumitelná chyba s návodem spustit `prepare-data`.
- Dotaz bez shody (žádná relevantní pasáž nad prahem) → poctivé „nenašel jsem v textu",
  ne vymyšlená odpověď.
- Prázdný dotaz → validace vstupu.
- Neznámá slova v dotazu → retrieval je ignoruje (nejsou ve slovníku), skóre klesne.
- Download selže / zdroj nedostupný → přeskočí s upozorněním, pokračuje ostatními.

## 8. Testování (V1, pytest)

- Čištění zachová diakritiku a interpunkci (vzorek s „ěščřžý").
- Chunker: správná velikost a překryv; pokrytí celého textu; hraniční krátký dokument.
- Retriever: na známý dotaz vrátí očekávanou pasáž mezi top-k (na malém fixture textu);
  TF-IDF i BM25.
- Extraktivní Answerer: na „kdo vyráběl roboty?" nad fixture vrátí správný úsek a zdroj.
- Pipeline end-to-end: `otázka → Answer` na R.U.R. pro sadu ověřených dotazů.
- Poctivé „nenašel jsem" při dotazu mimo obsah.

## 9. V2 — roadmapa (další spec + plán)

- **QA-gen (`qagen/synth.py`):** ze syntetického maskování entit vyrobí
  `(otázka, kontext, odpověď)` páry z korpusu (cloze styl: věta → vymaskovaná entita → dotaz).
- **Tokenizer (`model/tokenizer.py`):** SentencePiece BPE, `character_coverage=1.0`,
  `byte_fallback`.
- **Generátor (`model/seq2seq.py`, `train.py`, `generate.py`):** malý encoder-decoder
  transformer trénovaný od nuly na QA párech; sampling generování; trénink na MPS.
- **Generativní Answerer (`jellyai/answerer/generative.py`):** obal generátoru do rozhraní
  `Answerer`, tedy plná zaměnitelnost s extraktivním.
- **Realistický strop:** model se naučí „gramaticky přeskládat nalezenou pasáž" do tvaru
  odpovědi; nejslabší místo = česká shoda (skloňování/časování). Nebude uvažovat ani skládat
  odpovědi z více faktů.

## 10. Mimo rozsah (YAGNI)

- Obecné znalosti mimo načtené texty (velký předtrénovaný model / API) — „pak uvidíme".
- Externí služby/frameworky (Ollama, API).
- Webové UI (zůstává CLI).
- Distribuovaný / multi-GPU trénink, RLHF, instrukční ladění.
