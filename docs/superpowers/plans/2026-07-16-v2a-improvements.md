# V2a-improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development). Steps use checkbox (`- [ ]`) syntax.
> **Realizace:** veškerý kód píše Claude sám (preference uživatele).

**Goal:** Zkvalitnit a rozšířit syntetická QA data — lepší dělení vět, filtr kvality otázek, čištění datových závorek, víc wiki článků.

**Architecture:** Úpravy stávajících bloků (`jellyai/text.py`, `dataprep/clean.py`, `qagen/build.py`) + nové moduly (`qagen/quality.py`, `dataprep/wiki.py`) + CLI příkaz `wiki`.

**Tech Stack:** Python 3.11, stdlib (re, json, urllib). Bez nových třetích stran.

## Global Constraints

- Python **3.11** ve venv; hermetické testy (žádná síť); UTF-8 + diakritika.
- Pracovní větev: `feature/v2a-improvements`.
- **Každá commit zpráva končí trailerem:**
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```

---

### Task 1: Lepší dělení vět (text.py)

**Files:**
- Modify: `jellyai/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Produces: přepsané `split_sentences(text) -> list[str]` — neseká na zkratkách a na ordinálech/datech (číslo + následující malé písmeno).

- [ ] **Step 1: Přidat failing testy**

Do `tests/test_text.py` přidej:
```python
def test_split_sentences_keeps_dates_and_abbrev():
    from jellyai.text import split_sentences
    assert split_sentences("Narodil se 9. ledna 1890 v Úpici.") == \
        ["Narodil se 9. ledna 1890 v Úpici."]
    assert split_sentences("Ošetřil ho MUDr. Čapek osobně.") == \
        ["Ošetřil ho MUDr. Čapek osobně."]


def test_split_sentences_splits_number_before_capital():
    from jellyai.text import split_sentences
    assert split_sentences("Věta číslo 0. Věta číslo 1.") == \
        ["Věta číslo 0.", "Věta číslo 1."]
```

- [ ] **Step 2: Spustit — musí selhat**

Run: `python -m pytest tests/test_text.py -v`
Expected: FAIL (věty se tříští na „9." a „MUDr.")

- [ ] **Step 3: Přepsat split_sentences v jellyai/text.py**

Nahraď stávající `_SENT_RE` a `split_sentences` tímto:
```python
_BOUNDARY_RE = re.compile(r"[.!?…]\s+")

# České zkratky, po nichž se nemá dělit věta.
_ABBREV = {
    "mudr", "judr", "phdr", "rndr", "ing", "prof", "doc", "csc", "mgr", "bc",
    "tzv", "atd", "apod", "např", "tj", "mj", "tzn", "aj", "resp", "popř", "cca",
    "č", "r", "st", "sv", "kap", "obr", "tab", "roč", "s", "str", "stol",
}


def split_sentences(text):
    """Rozdělí text na věty a nenaletí na zkratky ani na data/ordinály.

    Věta je základní jednotka odpovědi, takže na jejím správném vyseknutí hodně
    záleží. Dělíme na koncové interpunkci s mezerou, ale hranici zahodíme, když
    tečka patří ke zkratce („MUDr.", „tzv.") nebo k pořadovému číslu/datu, které
    pokračuje malým písmenem („9. ledna"). Číslo následované velkým písmenem
    („…číslo 0. Věta…") je naopak normální konec věty.

    Args:
        text (str): Vstupní text (odstavec, pasáž, dokument).

    Returns:
        list[str]: Věty bez okolních bílých znaků; prázdné vynechány.
    """
    text = text.strip()
    if not text:
        return []
    sentences = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        punct_pos = match.start()
        # slovo těsně před interpunkcí (souvislý alnum běh)
        j = punct_pos
        while j > start and text[j - 1].isalnum():
            j -= 1
        preceding = text[j:punct_pos]
        next_char = text[match.end()] if match.end() < len(text) else ""
        if preceding.lower() in _ABBREV:
            continue                                   # zkratka → nesekat
        if preceding.isdigit() and next_char.islower():
            continue                                   # ordinál/datum uprostřed
        sentence = text[start:punct_pos + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
```

- [ ] **Step 4: Spustit celý test_text — musí projít**

Run: `python -m pytest tests/test_text.py -v`
Expected: PASS (nové i původní testy)

- [ ] **Step 5: Ověřit, že chunker/answerer/pipeline testy stále projdou**

Run: `python -m pytest tests/test_chunker.py tests/test_answerer.py tests/test_pipeline.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add jellyai/text.py tests/test_text.py
git commit -m "feat: dělení vět respektuje zkratky a data/ordinály"
```

---

### Task 2: Filtr kvality otázek (quality.py + build)

**Files:**
- Create: `qagen/quality.py`
- Modify: `qagen/build.py`
- Test: `tests/test_qagen_quality.py`

**Interfaces:**
- Produces: `is_acceptable(question: str, answer: str) -> bool`.
- Consumes v build: filtruje páry před zápisem.

- [ ] **Step 1: Přidat failing test**

`tests/test_qagen_quality.py`:
```python
from qagen.quality import is_acceptable


def test_good_question_accepted():
    assert is_acceptable("Kdo původně chtěl roboty nazvat laboři?", "Karel Čapek")


def test_leading_punctuation_rejected():
    assert not is_acceptable("Kdo , rodným jménem Karel?", "Karel")


def test_too_few_words_rejected():
    assert not is_acceptable("Kdy 1890 Malé Svatoňovice – 25?", "ledna")


def test_fragment_answer_rejected():
    assert not is_acceptable("Kdo je nejlepší?", "R")


def test_numeric_answer_ok():
    assert is_acceptable("Kolik planet má sluneční soustava?", "8")
```

- [ ] **Step 2: Spustit — musí selhat**

Run: `python -m pytest tests/test_qagen_quality.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Napsat qagen/quality.py**

```python
"""Filtr kvality vygenerovaných otázek.

Šablona z věty občas vyrobí paskvil — otázku, co začíná čárkou, skoro bez slov,
nebo odpověď, co je jen fragment. Tenhle filtr takové páry zahodí, aby se do
datasetu dostaly jen ty aspoň trochu smysluplné. Je to poslední síto před zápisem.
"""

# Znaky, kterými by rozumná otázka po tázacím slově začínat neměla.
_LEADING_BAD = set(",.;:)-–—…!?")


def is_acceptable(question, answer):
    """Rozhodne, zda je pár (otázka, odpověď) dost dobrý na zařazení do datasetu.

    Args:
        question (str): Vygenerovaná otázka (začíná tázacím slovem, končí „?").
        answer (str): Odpovědní spán.

    Returns:
        bool: True, když pár projde všemi kritérii kvality.
    """
    a = answer.strip()
    if len(a) < 2 and not a.isdigit():
        return False

    parts = question.split(" ", 1)
    body = parts[1].rstrip("?").strip() if len(parts) > 1 else ""
    if not body or body[0] in _LEADING_BAD:
        return False

    words = body.split()
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if len(alpha_words) < 3:
        return False
    if len(alpha_words) / len(words) < 0.5:
        return False
    return True
```

- [ ] **Step 4: Spustit — musí projít**

Run: `python -m pytest tests/test_qagen_quality.py -v`
Expected: PASS

- [ ] **Step 5: Zapojit filtr do build.py**

V `qagen/build.py` uprav vnitřek smyčky přes kandidáty:
```python
                for cand in candidates(sentence, tagger, config.qagen):
                    question = build_question(sentence, cand)
                    if not is_acceptable(question, cand.answer):
                        continue
                    key = (question, cand.answer, doc.doc_id)
```
A přidej import na začátek souboru:
```python
from qagen.quality import is_acceptable
```

- [ ] **Step 6: Spustit build test — musí projít**

Run: `python -m pytest tests/test_qagen_build.py -v`
Expected: PASS (fixture „Kdo roboty vynalezl?" filtr projde)

- [ ] **Step 7: Commit**

```bash
git add qagen/quality.py qagen/build.py tests/test_qagen_quality.py
git commit -m "feat: filtr kvality otázek + zapojení do build"
```

---

### Task 3: Čištění rozsahových datových závorek (clean.py)

**Files:**
- Modify: `dataprep/clean.py`
- Test: `tests/test_clean.py`

**Interfaces:**
- Produces: `clean_text` navíc odstraní `(… rok … – … rok …)` závorky.

- [ ] **Step 1: Přidat failing test**

Do `tests/test_clean.py` přidej:
```python
def test_clean_removes_date_range_parentheses():
    raw = "Karel Čapek (9. ledna 1890 Úpice – 25. prosince 1938 Praha) byl spisovatel."
    out = clean_text(raw)
    assert out == "Karel Čapek byl spisovatel."


def test_clean_keeps_normal_parentheses():
    raw = "Slovo robot (z českého robota) zdomácnělo."
    out = clean_text(raw)
    assert "(z českého robota)" in out
```

- [ ] **Step 2: Spustit — musí selhat**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL (datová závorka zůstává)

- [ ] **Step 3: Doplnit dataprep/clean.py**

Přidej regex a funkci (nad `clean_text`):
```python
# Rozsahová biografická závorka: obsahuje rok (15xx–20xx) i pomlčku (narození–úmrtí).
_DATE_RANGE_PAREN = re.compile(
    r"\s*\([^()]*\b(?:1[5-9]\d\d|20\d\d)\b[^()]*[–—-][^()]*\)"
)


def _strip_date_range_parens(text):
    """Odstraní jednoznačné rozsahové datové závorky (narození–úmrtí).

    Cílí jen na závorky, které mají rok i pomlčku — ty v úvodních větách dělají
    otázky nečitelnými. Běžné závorky (bez roku a pomlčky) nechá být.

    Args:
        text (str): Vstupní text.

    Returns:
        str: Text bez rozsahových datových závorek.
    """
    return _DATE_RANGE_PAREN.sub("", text)
```
A v `clean_text` zavolej po odstranění wiki aparátu:
```python
    text = _strip_wiki_apparatus(text)
    text = _strip_date_range_parens(text)
```

- [ ] **Step 4: Spustit — musí projít**

Run: `python -m pytest tests/test_clean.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataprep/clean.py tests/test_clean.py
git commit -m "feat: čištění odstraní rozsahové datové závorky (narození–úmrtí)"
```

---

### Task 4: Wiki fetch modul + config + CLI

**Files:**
- Create: `dataprep/wiki.py`
- Modify: `config.py` (přidat `wiki_titles`)
- Modify: `cli.py` (příkaz `wiki`)
- Modify: `jelly` (subcommand `wiki`)
- Test: `tests/test_wiki.py`

**Interfaces:**
- Produces: `fetch_extract(title)`, `fetch_articles(titles, dest_dir, fetch=fetch_extract)`, `_slug(title)`; `cli.cmd_wiki(config)`.

- [ ] **Step 1: Přidat failing test**

`tests/test_wiki.py`:
```python
from dataprep.wiki import _slug, fetch_articles


def test_slug():
    assert _slug("Karel Čapek") == "karel_čapek"
    assert _slug("R.U.R.") == "r.u.r."


def test_fetch_articles_writes_files(tmp_path):
    fake = {"Karel Čapek": "Text o Čapkovi.", "Prázdný": "  "}
    written = fetch_articles(
        list(fake), str(tmp_path), fetch=lambda t: fake[t]
    )
    assert len(written) == 1                       # prázdný přeskočen
    assert (tmp_path / "wiki_karel_čapek.txt").read_text(encoding="utf-8") == "Text o Čapkovi."
```

- [ ] **Step 2: Spustit — musí selhat**

Run: `python -m pytest tests/test_wiki.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Napsat dataprep/wiki.py**

```python
"""Stažení českých článků z Wikipedie jako korpusu.

Wikipedia je čistá, dobře psaná próza — ideální zdroj pro syntetická QA data.
Stahujeme přes oficiální API v režimu „explaintext" (holý text bez značek).
Fetcher jde injektovat, aby šla logika testovat bez sítě.
"""

import json
import os
import urllib.parse
import urllib.request

_API = "https://cs.wikipedia.org/w/api.php"


def _slug(title):
    """Převede název článku na bezpečné jméno souboru (malá písmena, _ místo mezer).

    Args:
        title (str): Název článku.

    Returns:
        str: Slug pro název souboru (diakritika zachována).
    """
    return title.lower().replace(" ", "_").replace("/", "_")


def fetch_extract(title):
    """Stáhne holý text jednoho článku z české Wikipedie.

    Args:
        title (str): Název článku (např. „Karel Čapek").

    Returns:
        str: Prostý text článku (může být prázdný, když článek neexistuje).
    """
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": title,
    }
    url = _API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "jellyAI3/edu (local)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def fetch_articles(titles, dest_dir, fetch=fetch_extract):
    """Stáhne seznam článků do dest_dir jako wiki_<slug>.txt.

    Best-effort: prázdné/chybové články přeskočí a nahlásí.

    Args:
        titles (list[str]): Názvy článků.
        dest_dir (str): Cílový adresář (vytvoří se).
        fetch (callable): Funkce title→text (injektovatelná kvůli testům).

    Returns:
        list[str]: Cesty k zapsaným souborům.
    """
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for title in titles:
        try:
            text = fetch(title)
        except Exception as exc:  # noqa: BLE001
            print(f"Přeskočeno {title}: {exc}")
            continue
        if not text.strip():
            print(f"Přeskočeno (prázdné): {title}")
            continue
        path = os.path.join(dest_dir, f"wiki_{_slug(title)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Staženo: {path}")
        written.append(path)
    return written
```

- [ ] **Step 4: Spustit — musí projít**

Run: `python -m pytest tests/test_wiki.py -v`
Expected: PASS

- [ ] **Step 5: Přidat wiki_titles do config.py**

V `DataConfig` přidej pole (za `books`):
```python
    # kurátorované české wiki články jako zdroj čisté prózy pro QA data
    wiki_titles: tuple = (
        "Karel Čapek", "Josef Čapek", "R.U.R.", "Válka s mloky",
        "Bílá nemoc", "Božena Němcová", "Jan Neruda",
    )
```
A doplň řádek do docstringu `DataConfig`:
```python
        wiki_titles (tuple): Názvy českých wiki článků ke stažení jako korpus.
```

- [ ] **Step 6: Přidat příkaz do cli.py**

Přidej funkci (za `cmd_qa_models`):
```python
def cmd_wiki(config):
    """Stáhne kurátorované wiki články do raw adresáře.

    Args:
        config (Config): Konfigurace (wiki_titles, raw_dir).

    Returns:
        list[str]: Cesty ke staženým souborům.
    """
    from dataprep.wiki import fetch_articles
    return fetch_articles(list(config.data.wiki_titles), config.data.raw_dir)
```
Do `_build_parser` přidej:
```python
    sub.add_parser("wiki", parents=[common], help="stáhne české wiki články do data/raw")
```
Do `main` větvení přidej:
```python
    elif args.command == "wiki":
        cmd_wiki(config)
```

- [ ] **Step 7: Přidat subcommand do wrapperu jelly**

V `jelly` do horního `case` pattern doplň `|wiki` a do vnitřního `case`:
```bash
      wiki)      "$PY" "$ROOT/cli.py" wiki ;;
```
A do řádku nápovědy doplň `wiki`.

- [ ] **Step 8: Spustit celou sadu**

Run: `python -m pytest -q`
Expected: PASS (vše hermetické; ÚFAL integrační test podle přítomnosti modelu)

- [ ] **Step 9: Commit**

```bash
git add dataprep/wiki.py config.py cli.py jelly tests/test_wiki.py
git commit -m "feat: stahování českých wiki článků (dataprep/wiki + ./jelly wiki)"
```

---

### Task 5: Ověření end-to-end + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Reálné ověření zlepšení**

```bash
python cli.py wiki           # stáhne kurátorované články
python cli.py reindex        # vyčistí (reference, datové závorky) → processed
python cli.py gen-qa         # vygeneruje dataset s lepšími větami + filtrem
```
Expected: dataset vznikne; namátkou zkontrola, že otázky jsou čitelnější než dřív
(míň fragmentů, žádné vedoucí interpunkce).

- [ ] **Step 2: Doplnit README (sekce V2a)**

V sekci „V2a" doplň za příkazy:
```markdown
Víc dat čisté prózy z Wikipedie:

```bash
./jelly wiki           # stáhne kurátorované české wiki články do data/raw
./jelly index          # vyčistí (reference, datové závorky) do data/processed
./jelly qa             # vygeneruje QA dataset (lepší dělení vět + filtr kvality)
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README — stahování wiki článků pro QA data"
```

---

## Self-Review (kontrola plánu proti specu)

**Spec coverage:** dělení vět (T1, spec 2.1), filtr kvality (T2, spec 2.2), datové
závorky (T3, spec 2.3), wiki fetch + config + CLI (T4, spec 2.4/3/4), ověření +
README (T5). Hermetické testy ke všem; síťová část wiki injektovaná (spec 7). ✔

**Placeholder scan:** kód všech tasků je kompletní a konkrétní; žádné TBD ani
„handle edge cases". ✔

**Type consistency:** `split_sentences -> list[str]`, `is_acceptable -> bool`,
`fetch_articles -> list[str]`, `cmd_wiki -> list[str]` — používány konzistentně;
`is_acceptable` volané v build se signaturou (question, answer) sedí na T2. ✔
