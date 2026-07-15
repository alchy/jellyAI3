# Design V2a: syntetická QA pipeline (qagen)

**Datum:** 2026-07-15
**Autor:** Jindřich Němec + Claude
**Status:** Schváleno (V2a detailně, V2b roadmapa)

## 1. Cíl a kontext

V1 odpovídá extraktivně (věta doslova z textu). V2 má naučit malý decoder-only
transformer skládat plynulejší odpovědi. K tréninku ale potřebuje **dvojice
otázka→odpověď**, které holý korpus neobsahuje. V2a je vyrobí synteticky.

Klíčové rozhodnutí (z brainstormingu): odpovědní spány a typy otázek určí
**ÚFAL nástroje** — **NameTag** (rozpoznávání pojmenovaných entit) a **MorphoDiTa**
(POS tagging, lemmatizace). Výstupem V2a je **QA dataset**, který jde prohlédnout
a zkontrolovat dřív, než se na něm začne trénovat (V2b).

**Realistický strop:** syntetické otázky budou občas gramaticky kostrbaté (česká
morfologie). Cíl není dokonalá čeština, ale dost dobrá data, aby se model naučil
vzor „z kontextu najdi a vrať odpověď".

## 2. Klíčová rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Výběr odpovědi | NameTag entity (osoby/místa/instituce/čas) + MorphoDiTa (čísla, POS) |
| Typy otázek | Šablony podle typu entity: Kdo / Co / Kde / Kdy / Kolik |
| Formát páru | trojice `(question, context, answer)` + metadata (typ, zdroj) |
| Výstup | JSONL soubor `data/qa/qapairs.jsonl` |
| Tagger | schovaný za rozhraní `Tagger` → testovatelné s FakeTagger, hermetické testy |
| Závislosti | `ufal.nametag`, `ufal.morphodita` + modely z LINDAT (CC BY-NC-SA, nekomerční) |
| Běh | plně lokálně, offline (po stažení modelů) |

## 3. Architektura

```
korpus (data/processed) → věty → Tagger(NER+POS) → výběr odpovědi + typ
   → šablona otázky → (question, context, answer) → data/qa/qapairs.jsonl
```

### 3.1 Rozhraní Tagger (`qagen/tagger.py`)
Izoluje závislost na ÚFAL za jednoduchý kontrakt, aby zbytek pipeline nevěděl nic
o konkrétní knihovně a šel testovat bez stažených modelů.

```
@dataclass Entity(text: str, type: str, start: int, end: int)   # type: P/G/I/T/…
@dataclass Token(text: str, lemma: str, pos: str, start: int, end: int)

class Tagger(Protocol):
    def entities(self, text: str) -> list[Entity]: ...   # z NameTag
    def tokens(self, text: str) -> list[Token]: ...      # z MorphoDiTa
```

- `UfalTagger` — implementace přes `ufal.nametag` + `ufal.morphodita` (načte modely).
- `FakeTagger` — pro testy vrací nakonzervované entity/tokeny pro fixture větu.

### 3.2 Výběr odpovědi a typu (`qagen/answers.py`)
Z věty vybere kandidátní „odpovědi" a přiřadí typ otázky:

| Zdroj (NameTag typ / POS) | Typ otázky |
|---|---|
| osoba (P) | Kdo |
| místo/geografie (G) | Kde |
| instituce/organizace (I) | Co / Která |
| čas, datum (T) | Kdy |
| číslovka (MorphoDiTa POS = C) | Kolik |

Pravidla: z jedné věty se vezme nejvýš `max_answers_per_sentence` kandidátů;
věty kratší než `min_tokens` se přeskočí (málo kontextu).

### 3.3 Šablona otázky (`qagen/questions.py`)
Otázka vznikne nahrazením odpovědního spánu tázacím slovem a lehkou úpravou:

- Vezme se věta, odpovědní spán se odstraní/nahradí wh-slovem podle typu.
- Příklad (public domain, R.U.R.):
  - Věta: „Roboty vynalezl starý Rossum." · odpověď „starý Rossum" (osoba)
  - → Otázka: „Kdo vynalezl roboty?" · Odpověď: „starý Rossum"
- Lemmatizace z MorphoDiTa pomůže očistit tvar (např. sjednotit koncovky).
- **Poctivě:** u složitějších vět bude otázka někdy kostrbatá (např. „Roboty
  vynalezl kdo?"). To je přijatelné — pár zůstává koherentní a učí správný vzor.

### 3.4 Sestavení datasetu (`qagen/build.py`)
Projde pasáže korpusu (Chunker z V1 nebo věty), na každou pustí Tagger, vygeneruje
páry a zapíše je do JSONL. Kontext = pasáž, ze které odpověď pochází.

Řádek datasetu:
```json
{"question": "Kdo vynalezl roboty?", "context": "…pasáž…",
 "answer": "starý Rossum", "type": "Kdo", "doc_id": "rur", "passage_index": 85}
```

### 3.5 Stažení modelů (`qagen/download_models.py`)
Stáhne české modely z LINDAT (MorphoDiTa: czech-morfflex; NameTag: czech-cnec).
Best-effort s jasnou hláškou; modely jsou CC BY-NC-SA (nekomerční, pro osobní OK).

## 4. Konfigurace (`config.py`, nový `QagenConfig`)

```
@dataclass QagenConfig:
    qa_path: str = "data/qa/qapairs.jsonl"
    morphodita_model: str = "data/models/czech-morfflex.tagger"
    nametag_model: str = "data/models/czech-cnec.ner"
    min_tokens: int = 5
    max_answers_per_sentence: int = 2
    types: tuple = ("Kdo", "Co", "Kde", "Kdy", "Kolik")
```

## 5. Tok dat

```
data/processed/*.txt → věty/pasáže → Tagger → answers.py (spán+typ)
   → questions.py (šablona) → build.py → data/qa/qapairs.jsonl
```

## 6. Ošetření chyb a hraniční případy

- Věta bez použitelné entity/čísla → nevznikne žádný pár (v pořádku).
- Chybějící model taggeru → srozumitelná chyba s návodem na `download_models`.
- Prázdný korpus / processed → jasná chyba (spusť `prepare-data`).
- Odpovědní spán = celá věta → přeskočit (otázka by neměla kontext).
- Duplicitní páry → deduplikace podle (question, answer, doc_id).

## 7. Testování (pytest, hermetické)

- **FakeTagger fixture:** pro větu „Roboty vynalezl starý Rossum." vrátí entitu
  osoby „starý Rossum" a tokeny s POS → `answers.py` vybere spán + typ „Kdo".
- **questions.py:** z (věta, spán, typ) složí očekávanou otázku.
- **build.py:** nad malým fixture korpusem + FakeTagger vytvoří validní JSONL
  se správnými poli; deduplikace funguje; věta bez entit nepřidá pár.
- **UfalTagger:** integrační test **přeskočený** (`skipif`), když modely nejsou
  stažené — aby CI/hermetické běhy nezávisely na síti.

## 8. Závislosti a prostředí

- Nové: `ufal.nametag`, `ufal.morphodita` (pip), + modely ~desítky MB z LINDAT.
- Prostředí: venv, Python 3.11. Modely a `data/qa/` gitignorovat.
- Vše lokální, offline (po stažení modelů). Žádná externí služba/API.

## 9. V2b — roadmapa (další spec + plán)

- **Tokenizer:** SentencePiece BPE nad korpusem (+ QA texty).
- **Model:** decoder-only transformer, formát `Kontext: … Otázka: … Odpověď: …`.
- **Trénink:** na `qapairs.jsonl`, AdamW + cosine LR, MPS, checkpointy.
- **Generování:** autoregresivní sampling (temperature/top-k/top-p).
- **GenerativeAnswerer:** obal modelu do rozhraní Answerer (pluggable);
  přepínač `answerer.mode` v CLI/`./jelly`.

## 10. Mimo rozsah (YAGNI)

- Samotný trénink/model (to je V2b).
- Dokonalá gramatická správnost syntetických otázek.
- Lemmatizace ve V1 retrieveru (volitelný samostatný krok, ne součást V2a).
- Neuronové generování otázek (drží se šablon + ÚFAL analýzy).
