# V2b — generátor odpovědí — Implementation Plan

> Autonomní běh (uživatel spí, revize v 9:00). TDD, commit po každém úkolu, na
> větvi `feature/v2b-generator`. Trénink lokálně na MPS.

**Goal:** Malý decoder-only transformer trénovaný od nuly, který z pasáže + otázky
vygeneruje odpověď; zapojený jako pluggable `GenerativeAnswerer`.

**Tech Stack:** Python 3.11, torch 2.13 (MPS), sentencepiece. Navazuje na V1/V2a.

## Global Constraints

- Python 3.11 venv; commity končí trailerem `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Hermetické testy: maličký model / pár kroků, žádný těžký trénink v testech.
- Model a tokenizer artefakty (`model/*.model`, `model/*.pt`) gitignorovat.
- **Nemergovat do main** — nechat na větvi k revizi.

## Úkoly

### T1 — GeneratorConfig + answerer.mode + gitignore
- `config.py`: `GeneratorConfig(vocab_size, n_layer=4, n_head=4, n_embd=256, block_size=256, dropout=0.1, lr, warmup, epochs, batch_size, sp_prefix, ckpt_path, temperature, top_k, top_p, max_new_tokens)`; `AnswererConfig.mode = "extractive"|"generative"`; `Config.generator`.
- `.gitignore`: `model/*.model`, `model/*.vocab`, `model/*.pt`.
- Test: defaulty.

### T2 — Tokenizer (`model/tokenizer.py`)
- SentencePiece BPE. `train_tokenizer(corpus_path, prefix, vocab_size)`, `SPTokenizer.load(prefix)`, `.encode(text)`, `.decode(ids)`, `.eos_id`, `.pad_id`.
- `<eos>` jako user-defined symbol.
- Test: natrénuj mini-tokenizer na fixture textu (tmp), ověř roundtrip a že `eos_id` existuje.

### T3 — Model (`model/gpt.py`)
- Decoder-only transformer: token+pos embedding → N× blok (pre-LN, kauzální MHA, MLP+GELU, residual) → LN → lineární hlava (weight tying).
- `GPT(config).forward(idx, targets=None) -> (logits, loss)`; `generate(idx, max_new_tokens, temperature, top_k, top_p, eos_id)`.
- Test: forward tvar `[B,T,V]`; s targets skalární loss; generate ≤ max_new_tokens a zastaví na eos.

### T4 — Dataset (`model/dataset.py`)
- `QADataset(jsonl_path, tokenizer, block_size)`: každý pár → `Kontext: {c}\nOtázka: {q}\nOdpověď: ` (prompt) + `{a}` + `<eos>` (odpověď). `input_ids`, `labels` (prompt = -100). Ořez kontextu zleva na `block_size`, odpověď zachována. Collate s paddingem.
- Test: labels na promptu = -100, na odpovědi skutečné id; ořez respektuje odpověď.

### T5 — Trénink (`model/train.py`)
- `train(config)`: tokenizer + QADataset + DataLoader; AdamW, cosine LR + warmup, grad clip; cross-entropy ignore_index=-100; MPS/CPU; checkpoint; průběžný vzorek generování.
- Test: smoke — 20 kroků na maličkém modelu + malém fixture sníží loss (CPU).

### T6 — Generování (`model/generate.py`)
- `generate_answer(model, tokenizer, context, question, config) -> str`: sestaví prompt, sampling, stop na eos, vrátí text odpovědi.
- Test: s maličkým modelem vrátí string (smoke).

### T7 — GenerativeAnswerer (`jellyai/answerer/generative.py`)
- `GenerativeAnswerer(config)` s líným načtením modelu+tokenizeru; `answer(question, retrieved) -> Answer` (top pasáž = kontext; zdroj z pasáže). Prázdné retrieved → poctivé „nenašel jsem".
- `explain()`.
- Test: smoke s maličkým modelem/tokenizerem → `Answer` se zdrojem.

### T8 — Integrace (pipeline + CLI + jelly)
- `QAPipeline.from_corpus`/`from_index` volí answerer podle `config.answerer.mode`.
- CLI `train-gen` (natrénuje), `ask/repl` respektují `--mode generative`. `./jelly train-gen`, `./jelly ask --gen`.
- Test: pipeline s mode="generative" a maličkým modelem vrátí Answer.

### T9 — Příprava dat + reálný trénink + eval (autonomně)
- Přegenerovat `qapairs.jsonl` z **wiki-only** (odstranit `data/raw/*rur*`, `./jelly index`, `./jelly qa`).
- Natrénovat tokenizer + model na MPS (background). Vyhodnotit: vygenerovat odpovědi na pár otázek, porovnat s extraktivním.
- Report do `docs/` / závěrečné shrnutí pro revizi v 9:00.

## Self-Review
Pokrývá spec: tokenizer (T2), model (T3), dataset+maska (T4), trénink (T5),
generování (T6), GenerativeAnswerer+integrace (T7,T8), reálný trénink (T9),
config (T1). Testy hermetické, těžký trénink mimo testy. ✔
