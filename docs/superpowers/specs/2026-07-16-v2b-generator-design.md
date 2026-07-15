# Design V2b: generátor odpovědí (decoder-only transformer)

**Datum:** 2026-07-16
**Autor:** Jindřich Němec + Claude (autonomní běh)
**Status:** Návrh (autonomně, k revizi v 9:00)

## 1. Cíl a kontext

V1 odpovídá extraktivně (věta doslova z textu), V2a vyrobil QA dataset. V2b je
poslední článek: malý **decoder-only transformer** natrénovaný od nuly, který
z nalezené pasáže + otázky **vygeneruje krátkou odpověď** — plynulejší než holá
věta. Zapojí se jako vyměnitelný blok `GenerativeAnswerer` vedle extraktivního.

**Realistický strop (na rovinu):** „gramatický přeskládávač" nalezené pasáže —
při malém modelu a ~2 tis. párech spíš memoruje a lehce přeohýbá; česká shoda
bude občas ustřelovat, uvažovat neumí. Hlavní hodnota = kompletní od-nuly
příklad podmíněné generace.

## 2. Klíčová rozhodnutí (navazují na brainstorming V2)

| Oblast | Rozhodnutí |
|---|---|
| Architektura | Decoder-only transformer (GPT styl), psaný od nuly v PyTorch |
| Formát | `Kontext: <pasáž>\nOtázka: <otázka>\nOdpověď: <odpověď><eos>` |
| Tokenizace | SentencePiece BPE nad korpusem, `character_coverage=1.0`, `byte_fallback` |
| Loss | Cross-entropy jen na tokenech odpovědi (kontext+otázka maskovány) |
| Velikost | malý: `n_layer=4, n_head=4, n_embd=256, block_size=256` (~3–6M param.) |
| Trénink | AdamW, cosine LR + warmup, grad clip, MPS (fallback CPU), checkpointy |
| Generování | autoregresivní sampling (temperature / top-k / top-p), stop na `<eos>` |
| Integrace | `GenerativeAnswerer` do rozhraní Answerer; `config.answerer.mode` |
| Data | trénink na `data/qa/qapairs.jsonl` — **přegenerovat z wiki-only** (drama pryč) |
| Závislosti | `torch`, `sentencepiece` (do `requirements-v2.txt`) |

## 3. Struktura (nové soubory)

```
model/
  tokenizer.py     # SentencePiece BPE: train/load, encode/decode, special tokeny
  gpt.py           # decoder-only transformer (blok, attention, GPT) — komentované
  dataset.py       # JSONL QA → tokenizované sekvence + maska loss na odpověď
  train.py         # trénovací smyčka (AdamW, cosine, MPS, checkpoint)
  generate.py      # autoregresivní sampling odpovědi
jellyai/answerer/
  generative.py    # GenerativeAnswerer (obal modelu do rozhraní Answerer)
config.py          # GeneratorConfig + answerer.mode
cli.py / jelly     # train-gen, gen (přepínač generativního answereru)
```

## 4. Komponenty

### 4.1 Tokenizer (`model/tokenizer.py`)
SentencePiece BPE nad vyčištěným korpusem (`data/processed`). Vocab ~8k,
`character_coverage=1.0`, `byte_fallback=True`. Speciální tokeny: `<eos>` (konec
odpovědi), případně `<pad>`. Obal: `train(corpus, prefix, vocab_size)`,
`load(path)`, `encode(text) -> list[int]`, `decode(ids) -> str`, `eos_id`.

### 4.2 Model (`model/gpt.py`)
Decoder-only transformer, GPT-2 styl (pre-LayerNorm, kauzální multi-head
self-attention, MLP s GELU, residual, weight tying vstup/výstup). Konfigurovatelný
`n_layer, n_head, n_embd, block_size, dropout, vocab_size`. `forward(idx, targets=None)`
vrací logits a (volitelně) loss; `generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id)`.

### 4.3 Dataset (`model/dataset.py`)
Načte `qapairs.jsonl`, každý pár složí do formátu (viz 2), tokenizuje. Loss maska:
labels = -100 na tokenech promptu (`Kontext: … Otázka: … Odpověď: `), skutečné id
na tokenech odpovědi + `<eos>`. Ořez na `block_size` (uřízne se **kontext**, ne
odpověď). Vrací `(input_ids, labels)`.

### 4.4 Trénink (`model/train.py`)
AdamW, cosine LR s warmupem, gradient clipping, cross-entropy (ignore_index=-100).
MPS s fallbackem na CPU. Sleduje train loss, průběžně ukazuje vzorek generování.
Ukládá checkpoint (`model/ckpt.pt`).

### 4.5 Generování (`model/generate.py`)
Sestaví prompt `Kontext: … Otázka: … Odpověď: `, autoregresivně samplinguje
(temperature/top-k/top-p), zastaví na `<eos>` nebo max délce, vrátí text odpovědi.

### 4.6 GenerativeAnswerer (`jellyai/answerer/generative.py`)
Stejné rozhraní jako `ExtractiveAnswerer`: `answer(question, retrieved) -> Answer`.
Vezme top pasáž jako kontext, vygeneruje odpověď, vrátí ji + zdroj. Načítá model
+ tokenizer líně. Pluggable přes `config.answerer.mode = "extractive"|"generative"`.

## 5. Tok dat

```
data/processed → SentencePiece → tokenizer
qapairs.jsonl (wiki) → dataset (prompt+odpověď, maska) → train → model/ckpt.pt
[dotaz] → retriever → GenerativeAnswerer(prompt → sampling) → Answer
```

## 6. Ošetření chyb a hraniční případy

- Chybějící checkpoint/tokenizer při generování → jasná hláška („spusť train-gen").
- Sekvence delší než `block_size` → ořízne se kontext, odpověď zůstane.
- MPS nedostupné → CPU fallback s upozorněním.
- Prázdná/žádná pasáž z retrieveru → answerer vrátí poctivé „nenašel jsem".

## 7. Testování (pytest)

- **Tokenizer:** roundtrip `decode(encode(x))` ≈ x (na malém vzorku); `<eos>` má id.
- **Model:** `forward` vrací logits tvaru `[B,T,vocab]`; s targets vrací skalární loss;
  `generate` vrátí ≤ max_new_tokens tokenů a zastaví na `<eos>`.
- **Dataset:** prompt tokeny mají label -100, odpověď skutečné id; ořez respektuje odpověď.
- **GenerativeAnswerer:** s maličkým modelem vrátí `Answer` se zdrojem (smoke).
- **Smoke trénink:** pár kroků na maličkém modelu sníží loss (bez závislosti na GPU).
- Těžký reálný trénink NENÍ v testech (běží přes `train-gen`).

## 8. Mimo rozsah (YAGNI)

- Velký model / dlouhý trénink na obřích datech.
- Instrukční ladění, RLHF, beam search.
- Obecné znalosti mimo korpus.
