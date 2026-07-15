# Design: V2a-improvements (kvalita QA dat)

**Datum:** 2026-07-16
**Autor:** Jindřich Němec + Claude
**Status:** Schváleno

## 1. Cíl a kontext

V2a generuje QA dataset, ale na reálné próze (česká Wikipedie) je část otázek
kostrbatá. Analýza výstupu ukázala čtyři konkrétní příčiny; tento design je řeší.
Cílem je **kvalitnější a hojnější dataset** pro budoucí trénink generátoru (V2b),
beze změny architektury.

## 2. Čtyři vylepšení

### 2.1 Lepší dělení vět (`jellyai/text.py`)
`split_sentences` sekal na tečce u zkratek („MUDr.") a pořadových čísel/dat
(„9. ledna 1890"), čímž tříštil věty a mrzačil otázky.

Nové pravidlo: na hranici `[.!?…] + mezera` se **neseká**, když:
- před tečkou je **samé číslo** A následující slovo začíná **malým písmenem**
  (ordinál/datum uprostřed věty — „9. ledna"); u čísla následovaného velkým
  písmenem se seká normálně („…číslo 0. Věta…"),
- před tečkou je **známá zkratka** (`MUDr.`, `tzv.`, `atd.`, `apod.`, `např.`,
  `tj.`, `mj.`, `č.`, `r.`, `st.`, `sv.`, `ing.`, `prof.`…) — pak nikdy.

Ovlivní i V1 (chunker/retriever) — k lepšímu; existující testy zůstávají v platnosti.

### 2.2 Filtr kvality otázek (`qagen/quality.py` — nový)
`is_acceptable(question, answer) -> bool` zahodí zjevně rozbité páry:
- odpověď kratší než 2 znaky a není číslo,
- za tázacím slovem hned interpunkce (`Kdo , rodným…`),
- zbytek otázky má míň než 3 slovná (písmenná) slova,
- podíl slovných slov ve zbytku < 50 % (samé číslice/interpunkce).

`build_dataset` každý pár tímto proţene; neprošlé zahodí.

### 2.3 Konzervativní čištění datových závorek (`dataprep/clean.py`)
Odstraní jen **jednoznačné rozsahové biografické závorky** — `(… rok … – … rok …)`
(narození–úmrtí), tj. závorka obsahující rok (15xx–20xx) i pomlčku. Běžné závorky
(bez roku a pomlčky) zůstanou. Cílené, ať se nesmaže užitečný obsah.

### 2.4 Víc wiki článků (`dataprep/wiki.py` — nový + CLI)
Blok pro stažení českých wiki článků přes `cs.wikipedia.org/w/api.php`
(explaintext). Kurátorovaný seznam titulů v configu. Nový příkaz `./jelly wiki`
je stáhne do `data/raw/` jako `wiki_<slug>.txt`. Fetcher je injektovatelný
(testovatelnost bez sítě).

## 3. Konfigurace

- `DataConfig.wiki_titles: tuple` — kurátorovaný seznam českých článků
  (Čapkové + pár autorů/děl).

## 4. Rozhraní (nové)

- `jellyai.text.split_sentences(text) -> list[str]` — přepsané chování (viz 2.1).
- `qagen.quality.is_acceptable(question: str, answer: str) -> bool`.
- `dataprep.wiki.fetch_extract(title: str) -> str`,
  `dataprep.wiki.fetch_articles(titles, dest_dir, fetch=fetch_extract) -> list[str]`,
  `dataprep.wiki._slug(title: str) -> str`.
- `cli.cmd_wiki(config) -> list[str]`.

## 5. Tok dat (beze změny principu)

```
wiki API → data/raw/wiki_*.txt → clean (odřez referencí + datových závorek)
   → data/processed → chunker (lepší věty) → qagen → filtr kvality → JSONL
```

## 6. Ošetření chyb a hraniční případy

- Dělení vět: číslo na konci reálné věty (velké písmeno dál) se stále dělí; prázdný
  vstup → `[]`.
- Filtr: extrémně krátká/prázdná otázka → zamítnuto.
- Wiki: neexistující/prázdný článek → přeskočí s hláškou; síťová chyba → přeskočí.
- Čištění závorek: závorka bez roku+pomlčky se nedotkne.

## 7. Testování (pytest, hermetické)

- **Dělení vět:** neseká „9. ledna 1890" ani „MUDr. Čapek"; **seká** „…číslo 0. Věta…";
  prázdný vstup → `[]`. (A stávající testy chunkeru/text zůstávají zelené.)
- **Filtr:** dobrá otázka projde; vedoucí interpunkce, krátký zbytek, fragment
  odpovědi → zamítnuto.
- **Čištění:** rozsahová závorka `(… 1890 … – … 1938 …)` zmizí; běžná závorka zůstane.
- **Wiki:** `_slug` mapuje název na jméno souboru; `fetch_articles` s injektovaným
  fetcherem zapíše soubory (bez sítě).

## 8. Mimo rozsah (YAGNI)

- Dokonalá segmentace vět (statistický model) — držíme se pravidel.
- NLP-based čištění závorek — jen cílený rozsahový vzor.
- Generátor V2b (samostatný spec/plán).
