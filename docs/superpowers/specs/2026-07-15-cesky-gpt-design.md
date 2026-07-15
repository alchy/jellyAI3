# Design: jellyAI3 → malý český GPT

**Datum:** 2026-07-15
**Autor:** Jindřich Němec + Claude
**Status:** Návrh ke schválení

## 1. Cíl a kontext

Původní jellyAI3 je slovní LSTM model pro predikci dalšího slova, trénovaný na jediném
textu (Čapkovo R.U.R.). Generuje nekvalitní výstup a zahazuje diakritiku, interpunkci
i velikost písmen.

Cílem je posunout projekt na moderní úroveň se **dvěma rovnocennými prioritami**:

1. **Naučit se moderní architekturu** — postavit decoder-only transformer (GPT styl)
   od základu, srozumitelně a komentovaně.
2. **Znatelně lepší výstup** — generovat souvislou, gramaticky správnou češtinu
   včetně diakritiky.

Bereme to jako vážnější projekt, kterému věnujeme čas.

## 2. Klíčová rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Architektura | Decoder-only transformer (nanoGPT styl), psaný od základu |
| Tokenizace | SentencePiece BPE, vocab 8–16k, `character_coverage=1.0`, `byte_fallback` |
| Data | Kurátorovaný korpus českých public-domain knih (jednotky až nižší desítky MB) |
| Čeština | Zachovat diakritiku, interpunkci i velikost písmen; jen lehké čištění balastu |
| Velikost modelu | ~30–50M parametrů (výchozí: `n_layer=8, n_head=8, n_embd=512, block_size=512`) |
| Generování | Autoregresivní sampling (temperature / top-k / top-p), ne beam search |
| Hardware | Apple M4 Pro, 20 GPU jader, 24 GB RAM → trénink na **MPS**, fallback CPU |
| Prostředí | venv, **Python 3.11 nebo 3.12** (novější verze mají problém s torch) |
| Repo | Rozvinout přímo v repu `jellyAI3`; LSTM verze zůstává dohledatelná v git historii |

## 3. Struktura repozitáře

```
jellyai3/
├── config.py            # jeden dataclass s veškerou konfigurací
├── data/
│   ├── download.py      # stáhne kurátorovaný seznam public-domain českých knih
│   ├── clean.py         # čištění: licenční patičky, kódování, bílé znaky (zachová diakritiku)
│   ├── raw/             # stažené syrové texty (gitignorováno)
│   └── processed/       # vyčištěný korpus (gitignorováno)
├── tokenizer/
│   └── train_bpe.py     # natrénuje SentencePiece BPE na vyčištěném korpusu
├── src/
│   ├── model.py         # decoder-only transformer (attention, blok, GPT) — komentované
│   ├── dataset.py       # tokenizace korpusu, bloky délky block_size, train/val split
│   ├── train.py         # trénovací smyčka: AdamW, cosine LR + warmup, grad clip, checkpointy
│   ├── generate.py      # autoregresivní sampling (temperature / top-k / top-p)
│   └── device.py        # výběr MPS / CPU
├── tests/               # pytest
├── cli.py               # jednotné rozhraní: prepare-data | train-tokenizer | train | generate
├── requirements.txt
└── README.md            # setup venv + postup krok za krokem
```

Každý modul má jeden jasný účel a dá se testovat samostatně.

## 4. Komponenty

### 4.1 Model (`src/model.py`)
Decoder-only transformer, moderní GPT-2 styl:
- Token embedding + poziční embedding (naučené).
- N× transformer blok: **pre-LayerNorm** → kauzální multi-head self-attention → residual;
  pre-LayerNorm → MLP (2 lineární vrstvy, GELU) → residual.
- Závěrečný LayerNorm → lineární hlava na velikost slovníku.
- **Weight tying** vstupního embeddingu a výstupní hlavy.
- Kauzální maska zajistí, že pozice `t` vidí jen pozice `≤ t`.
- Konfigurovatelné `n_layer, n_head, n_embd, block_size, dropout, vocab_size`.

### 4.2 Data (`data/download.py`, `data/clean.py`)
- `download.py`: stáhne knihy podle seznamu URL v configu (Project Gutenberg, Wikizdroje).
  Seznam je kurátorovaný a rozšiřitelný. Cíl: jednotky až nižší desítky MB čistého textu.
- `clean.py`: odstraní licenční hlavičky/patičky (např. Gutenberg boilerplate), sjednotí
  kódování na UTF-8, normalizuje bílé znaky a konce řádků. **Zachová diakritiku,
  interpunkci a velikost písmen.**

### 4.3 Tokenizer (`tokenizer/train_bpe.py`)
- SentencePiece BPE nad vyčištěným korpusem.
- `vocab_size` 8–16k (výchozí 16k), `character_coverage=1.0`, `byte_fallback=True`.
- Výstup: `sp_model.model` + `sp_model.vocab`.

### 4.4 Dataset (`src/dataset.py`)
- Načte vyčištěný korpus, tokenizuje ho na jeden dlouhý proud id.
- Vytváří tréninkové vzorky: vstup = `tokens[i : i+block_size]`,
  cíl = `tokens[i+1 : i+block_size+1]` (posun o 1, predikce dalšího tokenu na každé pozici).
- Train/val split (výchozí 90/10).

### 4.5 Trénink (`src/train.py`)
- Ztráta: cross-entropy přes všechny pozice.
- Optimalizátor: **AdamW** s weight decay; **cosine LR schedule** s lineárním warmupem.
- Gradient clipping, dropout — kvůli riziku přeučení na malém korpusu.
- Sledování train/val loss + **perplexity**; průběžné ukázky generování během tréninku.
- Checkpoint nejlepšího modelu (dle val loss) + možnost resume.
- Early stopping podle val loss.
- Běh na MPS, fallback CPU.

### 4.6 Generování (`src/generate.py`)
- Autoregresivní sampling: `temperature`, `top-k`, `top-p` (nucleus).
- Interaktivní režim: uživatel zadá začátek, model dopisuje N tokenů.

### 4.7 CLI (`cli.py`)
Jedno rozhraní se subpříkazy:
- `prepare-data` → download + clean
- `train-tokenizer` → SentencePiece BPE
- `train` → trénink modelu
- `generate` → interaktivní / jednorázové generování

## 5. Tok dat

```
public-domain URL  →  data/raw/*.txt  →  clean  →  data/processed/corpus.txt
        →  train BPE  →  sp_model.model
        →  dataset (id proud + bloky)  →  train  →  checkpoints/best.pt
        →  generate (prompt → text)
```

## 6. Ošetření chyb a hraniční případy

- Chybějící `sp_model.model` při tréninku/generování → jasná chybová hláška s návodem
  spustit `train-tokenizer` (původní kód na tomhle tiše padal).
- Prázdný / neexistující datový adresář → srozumitelná chyba.
- Prompt delší než `block_size` → ořízne se na posledních `block_size` tokenů.
- Neznámé znaky v promptu → řeší `byte_fallback` tokenizeru.
- MPS nedostupné → automatický fallback na CPU s upozorněním.

## 7. Testování (`tests/`, pytest)

- **Tokenizer roundtrip:** `decode(encode(text)) == text` (na vzorku s diakritikou).
- **Tvary datasetu:** vstup i cíl mají tvar `[block_size]`, cíl je vstup posunutý o 1.
- **Forward modelu:** výstup má tvar `[B, T, vocab_size]`.
- **Kauzální maska:** změna budoucího tokenu neovlivní výstup na dřívější pozici.
- **Generování:** vrátí přesně požadovaný počet nových tokenů; respektuje `top-k`/`top-p`.
- **Smoke trénink:** pár kroků na malém vzorku proběhne bez chyby a loss klesá.

## 8. Známé kompromisy a rizika

- **Přeučení:** 30–50M model na jednotkách–desítkách MB textu se bude chtít přeučovat.
  Mitigace: dropout, weight decay, early stopping, průběžné sledování val loss;
  případně rozšíření korpusu.
- **Dostupnost a kvalita dat:** public-domain české texty vyžadují čištění balastu;
  množství a čistota přímo ovlivní kvalitu výstupu. Seznam zdrojů je proto rozšiřitelný.
- **MPS omezení:** některé operace/přesnosti (bf16) mohou mít na MPS omezení; v případě
  potíží se sáhne po fp32.

## 9. Mimo rozsah (YAGNI)

- Distribuovaný / multi-GPU trénink.
- Fine-tuning předtrénovaných modelů.
- Webové UI (zůstává CLI).
- Pokročilé zarovnání (RLHF, instrukční ladění).
