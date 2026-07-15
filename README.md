# jellyAI3 — český QA nad texty (V1)

Výuková knihovna skládatelných bloků pro odpovídání na dotazy o obsahu českých
textů. Fakta se berou **přímo z textu přes retrieval** (ne z „paměti" modelu),
takže odpovědi jsou dohledatelné a model si nevymýšlí. Vše běží **lokálně, bez
externích služeb** (žádná Ollama, žádné API) a s minimem závislostí (numpy).

> Dřívější verze byla slovní LSTM model pro predikci dalšího slova — zůstává
> dohledatelná v git historii. Proč a jak vznikla tahle podoba, viz
> `docs/superpowers/specs/2026-07-15-cesky-gpt-design.md`.

## Setup

Vyžaduje **Python 3.11 nebo 3.12** (novější verze mají problém s torch, který
přijde až ve V2).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Použití

```bash
# 1) připravit data (naseeduje R.U.R. z training_text/, vyčistí do data/processed)
python cli.py prepare-data

# 2) zeptat se
python cli.py ask "kdo vynalezl roboty?"
python cli.py ask "co znamená R.U.R.?"

# kolik pasáží je v indexu
python cli.py build-index

# jak funguje který blok (výuková vrstva)
python cli.py explain retriever
python cli.py explain            # vypíše seznam bloků
```

Příklad výstupu:

```
$ python cli.py ask "kdo vynalezl roboty?"
Podle textu: Stojí tam například, že Roboty vynalezl starý pán.
(zdroj: rur#85)
```

## Jak to funguje — bloky

Pipeline vede dotaz řadou malých, nezávislých bloků:

```
Loader → Chunker → Retriever → ExtractiveAnswerer → Pipeline
```

| Blok | Co dělá |
|---|---|
| **Loader** | načte vyčištěné texty do objektů Document |
| **Chunker** | rozseká dokumenty na překrývající se pasáže (po větách) |
| **Retriever** | najde k dotazu nejrelevantnější pasáže — TF-IDF / BM25 psané od nuly v numpy |
| **ExtractiveAnswerer** | vybere z pasáží nejrelevantnější větu + uvede zdroj |
| **Pipeline** | pospojuje bloky do `ask(otázka) → odpověď` |

Veškerá konfigurace (velikost pasáží, metoda vyhledávání, top-k, …) je v
`config.py`. Každý blok umí `explain()` popsat, co dělá.

## Testy

```bash
python -m pytest -v
```

## Roadmapa

- **V2:** syntetické QA páry + malý seq2seq generátor od nuly → plynulejší,
  „učesané" odpovědi (dnešní V1 vrací větu doslova z textu). Realistický strop:
  gramatický přeskládávač nalezené pasáže, slabina = česká shoda.
- **Později:** obecné znalosti mimo načtené texty (větší předtrénovaný model).

Detaily v `docs/superpowers/specs/2026-07-15-cesky-gpt-design.md` a
`docs/superpowers/plans/2026-07-15-cesky-qa-v1.md`.
