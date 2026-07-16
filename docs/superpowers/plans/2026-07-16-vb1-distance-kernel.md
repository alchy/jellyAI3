# Fáze B1 — Vzdálenostní jádro — Implementační plán

> **Pro agentní workery:** POVINNÁ SUB-SKILL: použij superpowers:executing-plans
> (nebo subagent-driven-development) k implementaci úkol po úkolu. Kroky mají
> checkboxy (`- [ ]`).

**Goal:** Přidat větný retrieval s exponenciálním vzdálenostním útlumem
(`SentenceRetriever`), který zaostří odpověď na sémanticky nejrelevantnější větu a
její okolí, a napojit ho na existující answerery přes přepis anotací na větnou
granularitu.

**Architecture:** Nová třída `SentenceRetriever` (soubor `jellyai/sentence_retriever.py`)
znovu použije V1 `Retriever` jako vnitřní BM25 skórovač (nová aditivní metoda
`score_all`), navrch aplikuje čistou funkci `distance_activation` (útlum uvnitř
souboru, napříč soubory 0), najde vrchol a vyrobí ostřicí okno jako `Passage`.
Anotace se překlíčují na `(doc_id, index věty)`; `TemplateAnswerer` si složí anotaci
libovolné pasáže z rozsahu jejích vět. Volba granularity přes `RetrieverConfig`.

**Tech Stack:** Python 3.11/3.12, numpy, stdlib. ÚFAL jen přes existující služby
(`FakeUfalClient` v testech). Žádné nové závislosti.

## Global Constraints

- Bez nových závislostí (numpy + stdlib; ÚFAL jen přes stávající `UfalClient`/služby).
- V1 `Retriever` — **chování beze změny**; `score_all` je jen aditivní metoda.
- Každá funkce/metoda má **bohatý český docstring** (proč/co, Args, Returns) — styl projektu.
- Testy **hermetické** (bez modelů/sítě): `FakeUfalClient`, umělá skóre.
- Rozhraní `search()` zůstává `list[tuple[Passage, float]]`.
- Sentence index i anotace používají **lokální index věty v dokumentu** (shodně s `chunker`).
- Spouštění testů: `.venv/bin/python -m pytest`.
- TDD, časté commity (jeden na úkol). Commit message končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `Retriever.score_all` — surové skóre všech pasáží

**Files:**
- Modify: `jellyai/retriever.py` (přidat metodu do třídy `Retriever`)
- Test: `tests/test_retriever.py`

**Interfaces:**
- Produces: `Retriever.score_all(query: str) -> np.ndarray` — skóre délky `len(passages)`
  podle `config.method`, bez ořezu `top_k`. Používá ji `SentenceRetriever` (Task 4).

- [ ] **Step 1: Napiš padající test**

V `tests/test_retriever.py` přidej:

```python
def test_score_all_covers_all_passages_and_matches_search():
    from config import RetrieverConfig
    from jellyai.chunker import Passage
    from jellyai.retriever import Retriever
    passages = [
        Passage("d", 0, "roboti pracují v továrně", 0, 1),
        Passage("d", 1, "Helena přišla do továrny", 1, 2),
        Passage("d", 2, "moře je modré", 2, 3),
    ]
    r = Retriever(RetrieverConfig()).build(passages)
    scores = r.score_all("roboti v továrně")
    assert scores.shape == (3,)
    # nejvýš skórující pasáž je i první ve search
    import numpy as np
    top = int(np.argmax(scores))
    assert r.search("roboti v továrně")[0][0].index == passages[top].index
```

- [ ] **Step 2: Spusť test — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_retriever.py::test_score_all_covers_all_passages_and_matches_search -q`
Expected: FAIL (`AttributeError: 'Retriever' object has no attribute 'score_all'`).

- [ ] **Step 3: Implementuj `score_all`**

V `jellyai/retriever.py` přidej metodu do třídy `Retriever` (za `search`):

```python
    def score_all(self, query):
        """Vrátí skóre **všech** pasáží k dotazu — bez ořezu na top_k.

        Stejný výpočet jako `search`, ale vrací surový vektor skóre v pořadí
        `self.passages` (nic se neřadí ani nezahazuje). Slouží nadstavbám, které
        potřebují skóre každé pasáže zvlášť (např. větný retriever pro vzdálenostní
        útlum). Chování `search` to nijak nemění.

        Args:
            query (str): Dotaz v češtině.

        Returns:
            numpy.ndarray: Skóre pro každou pasáž (v pořadí `self.passages`);
                prázdné pole pro prázdný index.
        """
        if not self.passages:
            return np.zeros(0)
        tokens = self._tok(query)
        if self.config.method == "tfidf":
            return self._tfidf_scores(tokens)
        return self._bm25_scores(tokens)
```

- [ ] **Step 4: Spusť test — musí projít**

Run: `.venv/bin/python -m pytest tests/test_retriever.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/retriever.py tests/test_retriever.py
git commit -m "feat: Retriever.score_all — surové skóre všech pasáží"
```

---

### Task 2: `RetrieverConfig` — pole pro větný režim

**Files:**
- Modify: `config.py` (dataclass `RetrieverConfig`)
- Test: `tests/test_config.py` (vytvoř, pokud chybí)

**Interfaces:**
- Produces: `RetrieverConfig.granularity: str = "passage"`, `decay_tau: float = 1.6`,
  `focus_radius: int = 2`. Konzumují Task 3–4 a pipeline (Task 8).

- [ ] **Step 1: Napiš padající test**

Vytvoř/rozšiř `tests/test_config.py`:

```python
def test_retriever_config_distance_defaults():
    from config import RetrieverConfig
    c = RetrieverConfig()
    assert c.granularity == "passage"
    assert c.decay_tau == 1.6
    assert c.focus_radius == 2
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_retriever_config_distance_defaults -q`
Expected: FAIL (`AttributeError`/chybějící pole).

- [ ] **Step 3: Přidej pole**

V `config.py`, v `RetrieverConfig`, do docstringu doplň popis a za `use_stopwords`
přidej:

```python
    granularity: str = "passage"   # "passage" (V1) nebo "sentence" (B1)
    decay_tau: float = 1.6         # dosah exponenciálního útlumu (věty)
    focus_radius: int = 2          # poloměr ostřicího okna (vět na každou stranu)
```

A do docstringu třídy přidej řádky:
```
        granularity (str): "passage" (V1) nebo "sentence" (větný retrieval B1).
        decay_tau (float): Dosah útlumu τ pro větný režim (exp(−d/τ)).
        focus_radius (int): Poloměr ostřicího okna ve větách (na každou stranu).
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: RetrieverConfig — granularity/decay_tau/focus_radius pro větný režim"
```

---

### Task 3: `distance_activation` + `SentenceRetriever.build`

**Files:**
- Create: `jellyai/sentence_retriever.py`
- Test: `tests/test_sentence_retriever.py`

**Interfaces:**
- Produces:
  - `distance_activation(base, sent_doc, sent_local, tau) -> np.ndarray` — čistá funkce.
  - `SentenceRetriever(config).build(documents) -> SentenceRetriever` s poli
    `sent_doc: list[str]`, `sent_local: list[int]`, `sent_text: list[str]`,
    `_bounds: dict[str, tuple[int,int]]`, `_retriever: Retriever`.
- Consumes: `Document` (doc_id, text), `Passage`, `Retriever`, `split_sentences`.

- [ ] **Step 1: Napiš padající testy**

Vytvoř `tests/test_sentence_retriever.py`:

```python
import numpy as np
from config import RetrieverConfig
from jellyai.loader import Document
from jellyai.sentence_retriever import distance_activation, SentenceRetriever


def test_distance_activation_decays_and_respects_file_boundary():
    base = [0.0, 1.0, 0.0, 5.0]
    sent_doc = ["a", "a", "a", "b"]
    sent_local = [0, 1, 2, 0]
    finals = distance_activation(base, sent_doc, sent_local, tau=1.0)
    assert finals[1] == 1.0                          # vrchol si drží své
    assert finals[0] == finals[2]                    # symetrie sever/jih
    assert abs(finals[0] - np.exp(-1.0)) < 1e-9      # útlum o 1 krok
    assert finals[3] == 5.0                           # jiný soubor: 5 nikam nezasáhne
    # a vrchol z 'b' nezasáhl do 'a'
    assert finals[1] == 1.0


def test_build_indexes_sentences_per_document():
    docs = [Document("da", "da", "Alfa jedna. Klíč je tady. Gama tři."),
            Document("db", "db", "Delta prší. Epsilon svítí.")]
    sr = SentenceRetriever(RetrieverConfig()).build(docs)
    assert len(sr.sent_text) == 5
    assert sr.sent_doc == ["da", "da", "da", "db", "db"]
    assert sr.sent_local == [0, 1, 2, 0, 1]          # lokální index se resetuje per dokument
    assert sr._bounds["da"] == (0, 3) and sr._bounds["db"] == (3, 5)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py -q`
Expected: FAIL (`ModuleNotFoundError: jellyai.sentence_retriever`).

- [ ] **Step 3: Implementuj modul (build + čistá funkce)**

Vytvoř `jellyai/sentence_retriever.py`:

```python
"""Větný retriever se vzdálenostním útlumem (B1).

Místo pevných bloků skóruje na úrovni vět: nalezená věta vyzařuje své BM25 skóre
do okolí s exponenciálním útlumem podle vzdálenosti (sever i jih), **soubor je
tvrdá hranice**. Vrchol téhle aktivace je sémantický střed odpovědi; kolem něj se
vyrobí ostřicí okno jako běžná `Passage`, takže se answerer nemění.

Znovu používá V1 `Retriever` jako vnitřní BM25 skórovač (přes `score_all`), takže
se matematika neduplikuje a chování V1 zůstává nedotčené.
"""

from collections import defaultdict
import pickle

import numpy as np

from jellyai.text import split_sentences
from jellyai.chunker import Passage
from jellyai.retriever import Retriever


def distance_activation(base, sent_doc, sent_local, tau):
    """Rozlije větná skóre do okolí s exponenciálním útlumem uvnitř souboru.

    Pro každou větu s sečte příspěvky všech vět t **téhož souboru** vážené
    `exp(−|pozice_s − pozice_t| / τ)`. Věta obklopená relevantními větami tak
    vyskočí; osamocená shoda zůstane skromná. Napříč soubory je příspěvek nulový —
    soubor je tvrdá hranice, ať systém nezávisí na formátování odstavců.

    Args:
        base (Sequence[float]): Základní (BM25) skóre každé věty.
        sent_doc (Sequence[str]): doc_id každé věty (pro seskupení do souborů).
        sent_local (Sequence[int]): Lokální index věty v jejím dokumentu.
        tau (float): Dosah útlumu; pojistně zdola omezen na 1e-6.

    Returns:
        numpy.ndarray: Aktivované (finální) skóre v pořadí vstupu.
    """
    base = np.asarray(base, dtype=float)
    n = len(base)
    finals = np.zeros(n)
    tau = max(float(tau), 1e-6)
    groups = defaultdict(list)
    for k in range(n):
        groups[sent_doc[k]].append(k)
    for idxs in groups.values():
        idxs = np.array(idxs)
        local = np.array([sent_local[k] for k in idxs], dtype=float)
        dist = np.abs(local[:, None] - local[None, :])
        weight = np.exp(-dist / tau)
        finals[idxs] = weight @ base[idxs]
    return finals


class SentenceRetriever:
    """Retriever nad větami se vzdálenostním útlumem a ostřicím oknem."""

    def __init__(self, config):
        """Vytvoří prázdný větný retriever.

        Args:
            config (RetrieverConfig): Metoda/BM25 parametry + `decay_tau`,
                `focus_radius`, `top_k`. Index vznikne až `build`.
        """
        self.config = config
        self.sent_doc = []
        self.sent_local = []
        self.sent_text = []
        self._bounds = {}
        self._retriever = None

    def build(self, documents):
        """Rozdělí dokumenty na věty a postaví nad nimi vnitřní BM25 index.

        Každý dokument se rozseká `split_sentences` na věty s **lokálním indexem**
        (od 0). Věty se ukládají dokument po dokumentu (souvisle), takže hranice
        souboru je prostě rozsah indexů. Vnitřní `Retriever` skóruje jednotlivé
        věty jako 1větné pasáže.

        Args:
            documents (list[Document]): Dokumenty korpusu.

        Returns:
            SentenceRetriever: `self` (pro řetězení).
        """
        passages = []
        for doc in documents:
            sentences = split_sentences(doc.text)
            start = len(self.sent_text)
            for local, sent in enumerate(sentences):
                self.sent_doc.append(doc.doc_id)
                self.sent_local.append(local)
                self.sent_text.append(sent)
                passages.append(Passage(doc.doc_id, local, sent, local, local + 1))
            if len(self.sent_text) > start:
                self._bounds[doc.doc_id] = (start, len(self.sent_text))
        self._retriever = Retriever(self.config).build(passages)
        return self
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py -q`
Expected: PASS (2 testy).

- [ ] **Step 5: Commit**

```bash
git add jellyai/sentence_retriever.py tests/test_sentence_retriever.py
git commit -m "feat: distance_activation + SentenceRetriever.build (větný index)"
```

---

### Task 4: `SentenceRetriever.search` — vrchol → ostřicí okno

**Files:**
- Modify: `jellyai/sentence_retriever.py`
- Test: `tests/test_sentence_retriever.py`

**Interfaces:**
- Produces: `SentenceRetriever.search(query, top_k=None) -> list[tuple[Passage, float]]`
  — ostřicí okna kolem vrcholů; `Passage.start/end` = lokální indexy vět.

- [ ] **Step 1: Napiš padající test**

Přidej do `tests/test_sentence_retriever.py`:

```python
def test_search_focuses_on_matching_sentence():
    docs = [Document("da", "da", "Alfa jedna. Klíč leží tady. Gama tři."),
            Document("db", "db", "Delta prší. Epsilon svítí.")]
    cfg = RetrieverConfig(granularity="sentence", focus_radius=1, decay_tau=1.5)
    sr = SentenceRetriever(cfg).build(docs)
    results = sr.search("klíč", top_k=2)
    assert results, "něco se má najít"
    top_passage, top_score = results[0]
    assert top_passage.doc_id == "da"
    assert "Klíč leží tady" in top_passage.text        # vrchol
    assert top_passage.start <= 1 < top_passage.end     # okno obsahuje větu 1
    assert top_score > 0
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py::test_search_focuses_on_matching_sentence -q`
Expected: FAIL (`AttributeError: ... 'search'`).

- [ ] **Step 3: Implementuj `search`**

Do třídy `SentenceRetriever` přidej:

```python
    def search(self, query, top_k=None):
        """Najde ostřicí okna kolem vrcholů aktivace k dotazu.

        Základní BM25 skóre vět (`score_all`) se rozlije vzdálenostním útlumem
        (`distance_activation`). Vrcholy se berou hladově odshora; věty, které už
        leží v dřívějším okně, se přeskočí (aby okna byla různá). Kolem každého
        vrcholu se vyrobí okno ± `focus_radius` vět (ořezané na hranice dokumentu)
        jako běžná `Passage` — rozhraní se tak neliší od V1.

        Args:
            query (str): Dotaz v češtině.
            top_k (int | None): Kolik oken vrátit; None = z konfigurace.

        Returns:
            list[tuple[Passage, float]]: (ostřicí okno, skóre vrcholu) sestupně;
                prázdný seznam pro prázdný index nebo nulové skóre.
        """
        if top_k is None:
            top_k = self.config.top_k
        n = len(self.sent_text)
        if n == 0:
            return []
        base = self._retriever.score_all(query)
        finals = distance_activation(base, self.sent_doc, self.sent_local,
                                     self.config.decay_tau)
        radius = self.config.focus_radius
        results = []
        covered = set()
        for k in np.argsort(-finals):
            if finals[k] <= 0:
                break
            if k in covered:
                continue
            doc_id = self.sent_doc[k]
            g0, g1 = self._bounds[doc_id]
            lo = max(g0, k - radius)
            hi = min(g1 - 1, k + radius)
            window = Passage(
                doc_id=doc_id,
                index=self.sent_local[k],
                text=" ".join(self.sent_text[lo:hi + 1]),
                start=self.sent_local[lo],
                end=self.sent_local[hi] + 1,
            )
            results.append((window, float(finals[k])))
            covered.update(range(lo, hi + 1))
            if len(results) >= top_k:
                break
        return results
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py -q`
Expected: PASS (3 testy).

- [ ] **Step 5: Commit**

```bash
git add jellyai/sentence_retriever.py tests/test_sentence_retriever.py
git commit -m "feat: SentenceRetriever.search — vrchol → ostřicí okno"
```

---

### Task 5: `SentenceRetriever.save/load` — perzistence indexu

**Files:**
- Modify: `jellyai/sentence_retriever.py`
- Test: `tests/test_sentence_retriever.py`

**Interfaces:**
- Produces: `SentenceRetriever.save(path) -> str`, `SentenceRetriever.load(path) -> SentenceRetriever`.

- [ ] **Step 1: Napiš padající test**

Přidej do `tests/test_sentence_retriever.py`:

```python
def test_save_load_roundtrip(tmp_path):
    docs = [Document("da", "da", "Alfa jedna. Klíč leží tady. Gama tři.")]
    sr = SentenceRetriever(RetrieverConfig(granularity="sentence")).build(docs)
    path = str(tmp_path / "sent_index.pkl")
    sr.save(path)
    loaded = SentenceRetriever.load(path)
    assert loaded.sent_text == sr.sent_text
    assert loaded.search("klíč")[0][0].text == sr.search("klíč")[0][0].text
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py::test_save_load_roundtrip -q`
Expected: FAIL (`AttributeError: ... 'save'`).

- [ ] **Step 3: Implementuj `save`/`load`**

Přidej `import os` na začátek modulu (k ostatním importům) a do třídy:

```python
    def save(self, path):
        """Uloží postavený větný index na disk (pickle).

        Args:
            path (str): Cílová cesta.

        Returns:
            str: Cesta, kam byl index uložen.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        state = {
            "config": self.config,
            "sent_doc": self.sent_doc,
            "sent_local": self.sent_local,
            "sent_text": self.sent_text,
            "bounds": self._bounds,
            "retriever": self._retriever,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return path

    @classmethod
    def load(cls, path):
        """Načte dříve uložený větný index z disku.

        Args:
            path (str): Cesta k souboru s indexem.

        Returns:
            SentenceRetriever: Připravený k `search`.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        sr = cls(state["config"])
        sr.sent_doc = state["sent_doc"]
        sr.sent_local = state["sent_local"]
        sr.sent_text = state["sent_text"]
        sr._bounds = state["bounds"]
        sr._retriever = state["retriever"]
        return sr
```

(Přidej `import os` nahoru, pokud tam ještě není.)

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_sentence_retriever.py -q`
Expected: PASS (4 testy).

- [ ] **Step 5: Commit**

```bash
git add jellyai/sentence_retriever.py tests/test_sentence_retriever.py
git commit -m "feat: SentenceRetriever.save/load"
```

---

### Task 6: `annotate_documents` — větné anotace s posunem offsetů

**Files:**
- Modify: `jellyai/annotate.py`
- Test: `tests/test_annotate.py` (vytvoř, pokud chybí)

**Interfaces:**
- Produces: `annotate_documents(documents, client) -> dict` klíčovaný `(doc_id, index věty)`,
  hodnota `{"entities": [...], "sentences": [[token,...],...]}` s offsety v rámci
  dokumentu. `save_annotations`/`load_annotations` beze změny.
- Consumes: `Document`, `split_sentences`, `client.parse`, `client.entities`.

- [ ] **Step 1: Napiš padající test**

Vytvoř/rozšiř `tests/test_annotate.py`:

```python
from jellyai.loader import Document
from jellyai.annotate import annotate_documents
from jellyai.ufal_client import FakeUfalClient


def test_annotate_documents_keys_per_sentence_and_shifts_offsets():
    text = "Anna spí. Bere klobouk."
    docs = [Document("d", "d", text)]
    client = FakeUfalClient(
        parse={
            "Anna spí.": [[
                {"form": "Anna", "lemma": "Anna", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 4},
                {"form": "spí", "lemma": "spát", "upos": "VERB", "head": 0, "deprel": "root", "start": 5, "end": 8},
            ]],
            "Bere klobouk.": [[
                {"form": "Bere", "lemma": "brát", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 4},
                {"form": "klobouk", "lemma": "klobouk", "upos": "NOUN", "head": 1, "deprel": "obj", "start": 5, "end": 12},
            ]],
        },
        entities={"Anna spí.": [{"text": "Anna", "type": "P", "start": 0, "end": 4}]},
    )
    ann = annotate_documents(docs, client)
    assert set(ann.keys()) == {("d", 0), ("d", 1)}
    # věta 0 beze změny (base 0)
    assert ann[("d", 0)]["sentences"][0][0]["start"] == 0
    assert ann[("d", 0)]["entities"][0]["start"] == 0
    # věta 1 posunutá o base = len("Anna spí.") + 1 = 10 → disjunktní od věty 0
    assert ann[("d", 1)]["sentences"][0][0]["start"] == 10
    assert ann[("d", 1)]["sentences"][0][0]["form"] == "Bere"
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_annotate.py -q`
Expected: FAIL (`ImportError: cannot import name 'annotate_documents'`).

- [ ] **Step 3: Implementuj `annotate_documents`**

V `jellyai/annotate.py` přidej import nahoru a novou funkci (starou
`annotate_passages` ponech kvůli zpětné kompatibilitě testů, nebo ji smaž — Task 7/8
ji přestanou používat; zde ji necháme, ať nic nerozbijeme):

```python
from jellyai.text import split_sentences


def _shift(item, base):
    """Vrátí kopii tokenu/entity s offsety start/end posunutými o `base`.

    Posun do rámce celého dokumentu zajistí, že se offsety vět nepřekrývají —
    po složení ostřicího okna z více vět pak entita jedné věty nesedne na token
    jiné (viz `selection._tokens_in_span`).

    Args:
        item (dict): Token nebo entita s klíči start/end.
        base (int): O kolik posunout.

    Returns:
        dict: Kopie s posunutými start/end (None se nechá být).
    """
    out = dict(item)
    if out.get("start") is not None:
        out["start"] = out["start"] + base
    if out.get("end") is not None:
        out["end"] = out["end"] + base
    return out


def annotate_documents(documents, client):
    """Obohatí dokumenty o entity a rozbor **po větách** (klíč = index věty).

    Každý dokument se rozseká `split_sentences`; každá věta se zvlášť anotuje
    (entity + syntaktický rozbor) a její offsety se posunou do rámce dokumentu,
    takže jsou napříč větami disjunktní. Answerer si pak složí anotaci libovolné
    pasáže z rozsahu jejích vět (funguje pro chunkerová i ostřicí okna).

    Args:
        documents (list[Document]): Dokumenty korpusu.
        client: ÚFAL klient (`UfalClient` nebo `FakeUfalClient`).

    Returns:
        dict: (doc_id, index věty) → {"entities": [...], "sentences": [[token,...],...]}.
    """
    annotations = {}
    for doc in documents:
        base = 0
        for i, sent in enumerate(split_sentences(doc.text)):
            parsed = client.parse(sent)
            sentences = [[_shift(tok, base) for tok in s] for s in parsed]
            entities = [_shift(e, base) for e in client.entities(sent)]
            annotations[(doc.doc_id, i)] = {"entities": entities, "sentences": sentences}
            base += len(sent) + 1
    return annotations
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_annotate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jellyai/annotate.py tests/test_annotate.py
git commit -m "feat: annotate_documents — větné anotace s posunem offsetů do rámce dokumentu"
```

---

### Task 7: `TemplateAnswerer` skládá anotaci z rozsahu vět

**Files:**
- Modify: `jellyai/answerer/template.py`
- Test: `tests/test_template_answerer.py` (migrace klíčů na větné + nový test)

**Interfaces:**
- Consumes: `annotations` klíčované `(doc_id, index věty)` (Task 6).
- Produces: `TemplateAnswerer._annotation_for(passage) -> dict | None` složené přes
  `range(passage.start, passage.end)`.

- [ ] **Step 1: Uprav existující testy na větné klíče + přidej nový**

V `tests/test_template_answerer.py`:
- v `test_template_answerer_person` změň klíč anotace z `("wiki_bn", 5)` na `("wiki_bn", 0)`
  (pasáž má `start=0, end=1` → skládá se z věty 0).
- v `test_copula_definition_not_tautology` změň klíč z `("wiki", 1)` na `("wiki", 0)`.
- přidej test skládání okna přes dvě věty:

```python
def test_annotation_assembled_across_window():
    # okno pokrývá věty 0..1; spona je až ve větě 1 → přísudek se najde
    q = "kdo je Rossum?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 4, "end": 6},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 7, "end": 13},
    ]]})
    passage = Passage("wiki", 0, "Něco jiného. Rossum je vynálezce.", 0, 2)
    annotations = {
        ("wiki", 0): {"entities": [], "sentences": [[
            {"form": "Něco", "lemma": "něco", "upos": "PRON", "head": 0, "deprel": "root", "start": 0, "end": 4},
        ]]},
        ("wiki", 1): {"entities": [], "sentences": [[
            {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 13, "end": 19},
            {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 20, "end": 22},
            {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 23, "end": 32},
        ]]},
    }
    answerer = TemplateAnswerer(client, annotations, ExtractiveAnswerer(AnswererConfig()))
    result = answerer.answer(q, [(passage, 1.0)])
    assert result.text == "vynálezce"
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_template_answerer.py -q`
Expected: FAIL (migrace klíčů + nový test padnou, protože answerer ještě čte
`passage.index`).

- [ ] **Step 3: Implementuj skládání anotace**

V `jellyai/answerer/template.py` přidej metodu do `TemplateAnswerer` a uprav `answer`:

```python
    def _annotation_for(self, passage):
        """Složí anotaci pasáže z větného úložiště přes rozsah jejích vět.

        Anotace jsou klíčované (doc_id, index věty), takže pasáž libovolného původu
        (chunkerové okno i ostřicí okno větného retrieveru) se poskládá z vět
        `passage.start … passage.end`. Offsety jsou už v rámci dokumentu (disjunktní),
        takže spojení entit a vět je konzistentní.

        Args:
            passage (Passage): Pasáž s rozsahem vět (start/end = lokální indexy).

        Returns:
            dict | None: {"entities": [...], "sentences": [...]}, nebo None když
                žádná věta pasáže není anotovaná.
        """
        sentences, entities = [], []
        for i in range(passage.start, passage.end):
            annotation = self.annotations.get((passage.doc_id, i))
            if not annotation:
                continue
            sentences += annotation.get("sentences", [])
            entities += annotation.get("entities", [])
        if not sentences:
            return None
        return {"entities": entities, "sentences": sentences}
```

A v metodě `answer` nahraď řádek
`annotation = self.annotations.get((passage.doc_id, passage.index))`
za:
```python
            annotation = self._annotation_for(passage)
```

- [ ] **Step 4: Spusť — musí projít**

Run: `.venv/bin/python -m pytest tests/test_template_answerer.py -q`
Expected: PASS (všechny, včetně nového).

- [ ] **Step 5: Commit**

```bash
git add jellyai/answerer/template.py tests/test_template_answerer.py
git commit -m "feat: TemplateAnswerer skládá anotaci pasáže z rozsahu vět (větné anotace)"
```

---

### Task 8: Pipeline + CLI — volba granularity a větné anotace

**Files:**
- Modify: `jellyai/pipeline.py`
- Modify: `cli.py` (`cmd_annotate`)
- Test: `tests/test_pipeline.py` (rozšíř)

**Interfaces:**
- Produces: `pipeline._build_retriever(config, documents)`, `pipeline._load_retriever(index_path, config)`.
- Consumes: `SentenceRetriever`, `annotate_documents`, `load_documents`.

- [ ] **Step 1: Napiš padající test**

Přidej do `tests/test_pipeline.py`:

```python
def test_build_retriever_picks_sentence_for_sentence_granularity(tmp_path):
    from config import Config, RetrieverConfig
    from jellyai.loader import Document
    from jellyai.pipeline import _build_retriever
    from jellyai.sentence_retriever import SentenceRetriever
    from jellyai.retriever import Retriever
    docs = [Document("d", "d", "Alfa jedna. Klíč leží tady. Gama tři.")]

    cfg_sent = Config(); cfg_sent.retriever = RetrieverConfig(granularity="sentence")
    assert isinstance(_build_retriever(cfg_sent, docs), SentenceRetriever)

    cfg_pass = Config(); cfg_pass.retriever = RetrieverConfig(granularity="passage")
    assert isinstance(_build_retriever(cfg_pass, docs), Retriever)
```

- [ ] **Step 2: Spusť — musí spadnout**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::test_build_retriever_picks_sentence_for_sentence_granularity -q`
Expected: FAIL (`ImportError: cannot import name '_build_retriever'`).

- [ ] **Step 3: Implementuj `_build_retriever` / `_load_retriever` a přepoj `from_corpus`/`from_index`**

V `jellyai/pipeline.py` přidej helpery (za `_make_answerer`):

```python
def _build_retriever(config, documents):
    """Postaví retriever podle `config.retriever.granularity`.

    "sentence" → větný `SentenceRetriever` (B1, vzdálenostní útlum) nad dokumenty;
    jinak V1 `Retriever` nad chunkerovými pasážemi. Import větného retrieveru je
    líný, ať passage cesta nezávisí na modulu B1.

    Args:
        config (Config): Konfigurace (retriever + chunker).
        documents (list[Document]): Načtené dokumenty.

    Returns:
        Retriever | SentenceRetriever: Postavený index.
    """
    if config.retriever.granularity == "sentence":
        from jellyai.sentence_retriever import SentenceRetriever
        return SentenceRetriever(config.retriever).build(documents)
    passages = []
    for doc in documents:
        passages.extend(chunk(doc, config.chunker))
    return Retriever(config.retriever).build(passages)


def _load_retriever(index_path, config):
    """Načte uložený retriever podle `config.retriever.granularity`.

    Args:
        index_path (str): Cesta k uloženému indexu.
        config (Config): Konfigurace (rozhoduje o typu indexu).

    Returns:
        Retriever | SentenceRetriever: Načtený index.
    """
    if config.retriever.granularity == "sentence":
        from jellyai.sentence_retriever import SentenceRetriever
        return SentenceRetriever.load(index_path)
    return Retriever.load(index_path)
```

Uprav `from_corpus`, ať staví přes helper:
```python
        docs = load_documents(directory)
        retriever = _build_retriever(config, docs)
        answerer = _make_answerer(config)
        return cls(retriever, answerer)
```
Uprav `from_index`:
```python
        retriever = _load_retriever(index_path, config)
        answerer = _make_answerer(config)
        return cls(retriever, answerer)
```

- [ ] **Step 4: Přepoj CLI `cmd_annotate` na `annotate_documents`**

V `cli.py`, v `cmd_annotate`, nahraď tělo (načítání pasáží z indexu → načítání
dokumentů):

```python
    from jellyai.loader import load_documents
    from jellyai.annotate import annotate_documents, save_annotations
    documents = load_documents(config.data.processed_dir)
    own = client is None
    if own:
        from jellyai.ufal_client import UfalClient
        client = UfalClient(config.services)
    try:
        annotations = annotate_documents(documents, client)
    finally:
        if own:
            client.close()
    save_annotations(annotations, config.services.annotations_path)
    print(f"Anotováno {len(annotations)} vět → {config.services.annotations_path}")
    return len(annotations)
```
(Uprav i docstring `cmd_annotate`: nově anotuje **věty dokumentů**, ne pasáže indexu.)

- [ ] **Step 5: Spusť — musí projít celá sada**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (vč. nového pipeline testu; existující testy dál procházejí).

- [ ] **Step 6: Commit**

```bash
git add jellyai/pipeline.py cli.py tests/test_pipeline.py
git commit -m "feat: pipeline volí retriever podle granularity; CLI annotate anotuje věty dokumentů"
```

---

### Task 9: Živé ověření + dokumentace

**Files:**
- Modify: `config.py` (přepnout default `granularity` — viz níže, rozhodni)
- Create: `docs/superpowers/2026-07-16-vb1-results.md`
- Modify: `README.md`

**Interfaces:** žádné nové; end-to-end ověření + poctivé výsledky.

- [ ] **Step 1: Regeneruj větné anotace a index (potřebuje modely/služby)**

```bash
./jelly index                      # postaví index (passage default — viz Step 3 pro sentence)
./jelly annotate                   # nově: větné anotace → data/annotations.pkl
```

- [ ] **Step 2: Ověř větný režim naživo**

Dočasně zapni sentence režim (env nebo úprava configu) a postav sentence index:
```bash
# v config.py dočasně granularity="sentence", pak:
./jelly index
./jelly template "Jaký byl Josef Čapek?"
./jelly template "kdo napsal Babičku?"
./jelly template "kdy se narodil Karel Čapek?"
```
Zapiš skutečné odpovědi (i tam, kde spadne na extraktivní). Cíl: u definice, která
v korpusu **je** (Josef Čapek), zaostřit přísudek; jinde poctivě fallback.

- [ ] **Step 3: Rozhodni výchozí `granularity`**

Na základě výsledků: buď nech `"passage"` jako default a sentence jako opt-in
(bezpečnější), nebo přepni default na `"sentence"`. Zdůvodni v results dokumentu.

- [ ] **Step 4: Napiš results dokument**

Vytvoř `docs/superpowers/2026-07-16-vb1-results.md`: co B1 přidává, tabulka
před/po (stejný formát jako V4a results), poctivé omezení (korpus bez některých
definic), volba defaultu. (Bez reprodukce chráněného textu — jen krátké odpovědi.)

- [ ] **Step 5: Uprav README**

Přidej sekci „B1 — vzdálenostní jádro" (jak zapnout `granularity="sentence"`,
nutnost `./jelly annotate` po změně formátu) a doplň roadmapu (hotovo B1; další B2).

- [ ] **Step 6: Spusť celou sadu a commit**

```bash
.venv/bin/python -m pytest -q
git add -A
git commit -m "docs: B1 výsledky + README; volba výchozí granularity"
```

---

## Self-review (kontrola proti specu)

- **§4 SentenceRetriever** → Task 3 (build) + Task 4 (search) + Task 1 (score_all). ✓
- **§4 exp útlum / hranice souboru** → `distance_activation`, Task 3 test. ✓
- **§5 ostřicí okno jako Passage (start/end lok. indexy)** → Task 4. ✓
- **§6 větné anotace + posun offsetů** → Task 6. ✓
- **§7 _annotation_for přes rozsah vět** → Task 7. ✓
- **§8 konfigurace** → Task 2. ✓
- **§9 pipeline + CLI + save/load** → Task 5 (save/load) + Task 8. ✓
- **§10 testy** → hermetické v Task 1–8, e2e v Task 4/8 + živě Task 9. ✓
- **§11 hraniční případy** → prázdný index (Task 4 `if n==0`), τ≤0 (`distance_activation`),
  kraj dokumentu (ořez v `search`), okno bez anotace (Task 7 `None` → fallback). ✓
- **§12 mimo rozsah** → B2 se neimplementuje. ✓

Typová konzistence: `search()` vrací `list[tuple[Passage,float]]` (Task 4) shodně s V1;
`Passage.start/end` = lokální indexy vět použité v `_annotation_for` (Task 7) i v
anotačních klíčích (Task 6). ✓
