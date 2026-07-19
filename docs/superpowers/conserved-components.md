# Zakonzervované komponenty

Rozhodnutí (2026-07-16): jdeme cestou **faktového grafu**. Části, které se
neosvědčily, **nemažeme — konzervujeme** (kód i testy zůstávají, jen mimo hlavní
cestu a jasně označené). Plně vratné.

| Komponenta | Stav | Proč konzervováno | Jak zapnout |
|---|---|---|---|
| **B1 větný retrieval** (`SentenceRetriever`, vzdálenostní jádro) | zakonzervováno | živě smíšený výsledek, ne jasná výhra (`2026-07-16-vb1-results.md`) | `RetrieverConfig.granularity="sentence"` |
| **gen-qa / qagen** (syntetická QA data) | zakonzervováno | vzniklo pro generátor V2b (jiná větev, datově limitovaný); dataset nikdo nekonzumuje | `./jelly gen-qa` (běží dál) |
| **`question_pattern` — UDPipe rozbor otázky** (`answerer/pattern.py`, funkce + helpery; dataclassy Pattern/SubQuery zůstávají nosné) | zakonzervováno (řez #14, 2026-07-19) | šablony (vzorové karty + pseudo-QL) jsou jediná dotazová cesta — etalon 29/29 bez UDPipe; `query_mode` zrušen | přímý import `from jellyai.answerer.pattern import question_pattern` (testy `tests/test_pattern.py` běží dál) |

**Nezakonzervováno (zůstává v hlavní cestě):**
- Retrieval V1 (BM25/TF-IDF) + extraktivní answerer — robustní default.
- Větné anotace (`annotate_documents`) — nosné pro graf i template.
- Template answerer (V3/V4a) — čisté sponové odpovědi, fallback.
- `analyze_question`, ÚFAL služby — sdílené.
- **Faktový graf** — hlavní směr.

**Konsolidace:** až se faktový graf plně osvědčí, lze zakonzervované části odstranit
úplně (nebo je nechat jako výukové ukázky). Zatím se nemaže.
