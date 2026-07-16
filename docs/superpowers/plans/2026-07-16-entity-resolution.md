# Entity Resolution (genitiv ↔ nominativ) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jeden kanonický uzel na osobu — post-build resolver sloučí pádové varianty (`Josefa Čapka` → `Josef Čapek`), čímž se spojí roztříštěné fakty a rozběhne se rekurze, relace i navazující dotazy (cesta A z handoff specu `docs/superpowers/specs/2026-07-16-entity-identification-blocker.md`).

**Architecture:** Build zůstává beze změny (pro-drop koreference nerušena — žádný whack-a-mole). Nový pass `resolve_entities(graph)` běží po buildu: shlukne person-uzly kmenovým klíčem (`canon.cluster_key`), zvolí kanonické id, přepíše fakty/uzly/index. Query-side dostane kmenový fallback v `_resolve_topic` — **týž mechanismus** (`canon._stem`) na obou stranách.

**Tech Stack:** Python (interpreter **`.venv/bin/python`** — systémový python nemá ufal ML služby), pytest, benchmark `run_etalon.py`.

## Klíčová rozhodnutí (z dat, sondou do `data/graph.pkl`)

1. **Kanonický tvar clusteru = lexikograficky nejmenší člen** — NE `subj_count` z handoffu. Sonda ukázala, že pro-drop koreference rozdává podměty i genitivním uzlům (`Karla Čapka` subj=14, `Van Tocha` subj=7, `Adolfa Hitlera` subj=5), takže subj_count i váha vybírají genitiv. Naopak česká pádová koncovka nominativ **prodlužuje** (`Josef Čapek` < `Josefa Čapka` < `Josefem Čapkem`), takže lexikografické minimum trefilo nominativ ve **všech 15 vícečlenných clusterech** reálného grafu. Bonus: volba nezávisí na vahách → resolver je **idempotentní**.
2. **Žádný subset-merge** (otevřená otázka 2 handoffu): slučuje se jen stejný `cluster_key` (= stejný počet slov, stejné kmeny). `Antonína Čapka` (otec, klíč `(antonín, čapk)`) se strukturálně NEMŮŽE slít se synem `Karel Antonín Čapek` (klíč `(karl, antonín, čapk)`), holé `Čapek` zůstává zvlášť. Délkové fragmenty dál řeší per-dokument `_canonical_persons`.
3. **Jen `person`** (otevřená otázka 3): geo (`Slezsku`↔`Slezsko`) až později stejným passem.
4. **Resolver běží 2×**: na konci `build_graph` (per spec) a znovu v `build_fact_graph` po `recover_entities` — recover bere podměty ze **surového povrchu** entit (`_sentence_subject` vrací `e["text"]` bez canon) a mohl by pádový uzel znovu zanést. Díky idempotenci je druhý běh bezpečný.
5. **Query-side = kmenový fallback ve skóre `_resolve_topic`**, ne mapa `stem_key → id` navěšená na graf: mapa by v `_solve` přebila víceslovnou preferenci (`bratr Karla Čapka` musí dál rozřešit `Karel Antonín Čapek`, ne 2-slovný cluster) a vyžadovala by persistenci v pickle. Fallback dimenze skóre nemění žádné dnešní pořadí (viz Task 4).

## Global Constraints

- **Etalon nikdy pod 9/11**; cíl **10/11** („Kdo byl bratr Josefa Čapka?" → obsahuje „Karel"). Měř `.venv/bin/python benchmark/run_etalon.py` po každém tasku. Pozn.: „Jaký je robot?" → „R." je **pre-existing FAIL** (nesouvisí), zůstává.
- **Determinismus**: žádné iterování `set`, tie-breaky lexikograficky.
- **Slučovat jen person↔person, jen stejný `cluster_key`** (žádné subset/hladové slučování).
- **Build i query týž mechanismus**: `canon._stem`/`cluster_key` (žádná externí morfologie).
- Vždy `.venv/bin/python` / `.venv/bin/python -m pytest`.
- Celá suita zelená (206 testů + nové).
- Práce na větvi `entity-resolution` (repo commituje na `main` až po dokončení; `data/*.pkl` je git-ignored — rebuild se necommituje).

---

### Task 1: Stemmer — dativ `-ovi`

Bez toho se `Josefu Čapkovi` (w=5) a `Karlu Čapkovi` (w=1) neshluknou se svými osobami (klíč `(josf, čapkov)` ≠ `(josf, čapk)`).

**Files:**
- Modify: `jellyai/graph/canon.py:25-27` (`_SUFFIXES`)
- Test: `tests/test_canon.py`

**Interfaces:**
- Produces: `cluster_key("Josefu Čapkovi") == cluster_key("Josef Čapek")` — spoléhá na to Task 2/3.

- [ ] **Step 1: Failing test** — do `tests/test_canon.py` přidat:

```python
def test_dative_ovi_variants_share_key():
    """Dativ „-ovi" (Čapkovi) musí dát týž kmen jako ostatní pády (mezera stemmeru
    z handoffu: klíč (josf, čapkov) ≠ (josf, čapk) držel Čapkovi mimo cluster)."""
    assert cluster_key("Josefu Čapkovi") == cluster_key("Josef Čapek")
    assert cluster_key("Karlu Čapkovi") == cluster_key("Karel Čapek")
```

- [ ] **Step 2: Ověřit FAIL**

Run: `.venv/bin/python -m pytest tests/test_canon.py -q`
Expected: `1 failed` (`('josf', 'čapkov') != ('josf', 'čapk')`), ostatní pass.

- [ ] **Step 3: Implementace** — v `jellyai/graph/canon.py` doplnit `"ovi"` do `_SUFFIXES` (do skupiny `ov*`, před kratší koncovky):

```python
_SUFFIXES = ("ovými", "ových", "ovém", "ovou", "ové", "ová", "ovi", "ovu", "ovy",
             "ými", "ých", "ém", "ům", "ách", "emi", "ami", "ou", "em", "ěm",
             "e", "ě", "y", "u", "a", "o", "i", "í", "é")
```

- [ ] **Step 4: Ověřit PASS**

Run: `.venv/bin/python -m pytest tests/test_canon.py -q`
Expected: vše PASS.

- [ ] **Step 5: Celá suita + etalon (baseline se nesmí hnout — stemmer zatím nikdo nevolá nad reálnými daty)**

Run: `.venv/bin/python -m pytest -q` → vše zelené; `.venv/bin/python benchmark/run_etalon.py` → `JÁDRO: 9/11`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_canon.py jellyai/graph/canon.py
git commit -m "fix(canon): stemmer zvládá dativ -ovi (Čapkovi → čapk)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Post-build resolver `resolve_entities(graph)`

**Files:**
- Modify: `jellyai/graph/graph.py` (nová funkce na konec modulu; import `cluster_key`)
- Create: `tests/test_resolve_entities.py`

**Interfaces:**
- Consumes: `cluster_key` z Task 1; `make_fact`, `Participant` (extract.py); `Node`, `FactNode` (graph.py).
- Produces: `resolve_entities(graph: FactGraph) -> FactGraph` — in-place přepis, vrací týž graf. Volá ho Task 3 z `build_graph` i `tasks.build_fact_graph`.

- [ ] **Step 1: Failing testy** — nový `tests/test_resolve_entities.py`:

```python
"""Post-build resolver entit — sloučení pádových variant osob (cesta A blockeru).

Kanonické id = lexikograficky nejmenší člen clusteru (pádová koncovka nominativ
prodlužuje, takže minimum je nominativ; subj_count nefunguje — pro-drop rozdává
podměty i genitivním uzlům). Konzervativně: jen person↔person, jen týž kmenový klíč.
"""

from jellyai.graph.graph import FactGraph, resolve_entities
from jellyai.graph.extract import make_fact, Participant


def _bratr(subj, obj):
    return make_fact("bratr", [Participant("subj", subj, "person"),
                               Participant("obj", obj, "person")])


def _narodit(subj, loc):
    return make_fact("narodit", [Participant("subj", subj, "person"),
                                 Participant("loc", loc, "geo")])


def test_case_variants_merge_to_nominative():
    # přesně mechanismus blockeru: bratr-fakt visí na genitivu, narodit na nominativu
    g = FactGraph()
    g.add_fact(_narodit("Josef Čapek", "Hronově"))
    g.add_fact(_bratr("Karel Antonín Čapek", "Josefa Čapka"))
    resolve_entities(g)
    assert "Josefa Čapka" not in g.nodes
    bratr = g.facts_of("Josef Čapek", role="obj", predicate="bratr")
    assert bratr and g.participants(bratr[0], "subj") == ["Karel Antonín Čapek"]


def test_canonical_is_lex_min_not_weight():
    # genitiv frekventovanější (reálná data: 'Karla Čapka' w=15 vs 'Karel Čapek' w=3)
    g = FactGraph()
    for _ in range(5):
        g.add_fact(_narodit("Karla Čapka", "Praze"))
    g.add_fact(_narodit("Karel Čapek", "Praze"))
    resolve_entities(g)
    assert "Karel Čapek" in g.nodes and "Karla Čapka" not in g.nodes


def test_merged_fact_weights_aggregate():
    g = FactGraph()
    g.add_fact(_narodit("Josefa Čapka", "Hronově"))
    g.add_fact(_narodit("Josefa Čapka", "Hronově"))
    g.add_fact(_narodit("Josef Čapek", "Hronově"))
    resolve_entities(g)
    facts = g.facts_of("Josef Čapek", role="subj", predicate="narodit")
    assert len(facts) == 1 and facts[0].weight == 3
    assert g.nodes["Josef Čapek"].weight == 3


def test_conservative_father_son_and_bare_surname_stay_apart():
    # mantinel 4: jiný počet slov = jiný klíč → otec/syn/holé příjmení se nedotknou
    g = FactGraph()
    g.add_fact(_narodit("Karel Antonín Čapek", "Malých Svatoňovicích"))
    g.add_fact(_narodit("Antonína Čapka", "Žernově"))
    g.add_fact(_narodit("Čapek", "Praze"))
    resolve_entities(g)
    assert {"Karel Antonín Čapek", "Antonína Čapka", "Čapek"} <= set(g.nodes)


def test_non_person_nodes_untouched():
    # mantinel 3: geo/time/číselné uzly se (zatím) neshlukují
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Praha", "geo"),
                                 Participant("pred", "město", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Prahy", "geo"),
                                 Participant("pred", "město", "concept")]))
    resolve_entities(g)
    assert "Praha" in g.nodes and "Prahy" in g.nodes


def test_resolve_is_idempotent_and_deterministic():
    def build():
        g = FactGraph()
        g.add_fact(_bratr("Karel Antonín Čapek", "Josefa Čapka"))
        g.add_fact(_narodit("Josef Čapek", "Hronově"))
        g.add_fact(_narodit("Josefu Čapkovi", "Hronově"))   # dativ (Task 1)
        return resolve_entities(g)
    g1, g2 = build(), build()
    assert list(g1.facts.keys()) == list(g2.facts.keys())
    assert list(resolve_entities(g1).facts.keys()) == list(g2.facts.keys())
```

- [ ] **Step 2: Ověřit FAIL**

Run: `.venv/bin/python -m pytest tests/test_resolve_entities.py -q`
Expected: `ImportError: cannot import name 'resolve_entities'`.

- [ ] **Step 3: Implementace** — na konec `jellyai/graph/graph.py`; nahoře k importům přidat `from jellyai.graph.canon import cluster_key`:

```python
def resolve_entities(graph):
    """Post-build entity resolution: sloučí pádové varianty osob do jednoho uzlu.

    Osoby se shluknou kmenovým klíčem (`cluster_key`); kanonické id clusteru je
    **lexikograficky nejmenší člen** — pádové koncovky nominativ prodlužují
    („Josef Čapek" < „Josefa Čapka" < „Josefem Čapkem"), takže minimum bývá
    nominativ. (Subj-count jako proxy nominativu nefunguje: pro-drop koreference
    rozdává podměty i genitivním uzlům.) Fakty se přepíšou s přemapovanými
    účastníky (kolize identity → součet vah), uzly a index `_by_node` se přestaví.
    Jen person↔person; fragmenty s jiným počtem slov mají jiný klíč, takže se
    nedotknou (žádné hladové subset-slučování). Idempotentní a deterministické —
    smí běžet i opakovaně (po `recover_entities`).

    Args:
        graph (FactGraph): Graf k přepsání (in-place).

    Returns:
        FactGraph: Týž graf (pro řetězení).
    """
    clusters = {}
    for node in graph.nodes.values():
        if node.type == "person":
            clusters.setdefault(cluster_key(node.id), []).append(node.id)
    node_map = {}
    for members in clusters.values():
        canonical = min(members)
        for name in members:
            if name != canonical:
                node_map[name] = canonical
    if not node_map:
        return graph
    remapped = {}
    for fact in graph.facts.values():
        moved = make_fact(fact.predicate,
                          [Participant(p.role, node_map.get(p.node, p.node), p.type)
                           for p in fact.participants])
        key = (moved.predicate, moved.participants)
        existing = remapped.get(key)
        if existing is None:
            remapped[key] = FactNode(key, moved.predicate, fact.weight,
                                     moved.participants)
        else:
            existing.weight += fact.weight
    graph.facts = remapped
    graph.nodes = {}
    graph._by_node = {}
    for key, fact in remapped.items():
        for p in fact.participants:
            node = graph.nodes.get(p.node)
            if node is None:
                graph.nodes[p.node] = Node(p.node, p.type, fact.weight)
            else:
                node.weight += fact.weight
            graph._by_node.setdefault(p.node, []).append((key, p.role))
    return graph
```

(Váha uzlu = Σ vah faktů, v nichž figuruje, za každou účast — přesně sémantika `_touch`.)

- [ ] **Step 4: Ověřit PASS**

Run: `.venv/bin/python -m pytest tests/test_resolve_entities.py -q`
Expected: 6 passed.

- [ ] **Step 5: Celá suita**

Run: `.venv/bin/python -m pytest -q`
Expected: vše zelené (funkce zatím není zapojená, nic se nemění).

- [ ] **Step 6: Commit**

```bash
git add tests/test_resolve_entities.py jellyai/graph/graph.py
git commit -m "feat(graph): resolve_entities — post-build sloučení pádových variant osob

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Zapojení do pipeline (build_graph + build_fact_graph)

**Files:**
- Modify: `jellyai/graph/graph.py:239-240` (`build_graph` — volání za `_decompose_dates`)
- Modify: `jellyai/tasks.py:12,49-53` (`build_fact_graph` — druhý běh po `recover_entities`)
- Test: `tests/test_fact_graph.py`, `tests/test_tasks.py`

**Interfaces:**
- Consumes: `resolve_entities` (Task 2).
- Produces: `build_graph`/`build_fact_graph` vracejí graf už s kanonizovanými osobami — na tom stojí Task 5 (rebuild + etalon).

- [ ] **Step 1: Failing testy** — do `tests/test_fact_graph.py` přidat:

```python
def test_build_graph_merges_case_variants_across_documents():
    # d1: nominativ (narodit); d2: genitivní zmínka (bratr-relace) → po buildu 1 uzel
    A = {
        ("d1", 0): {"entities": [{"text": "Josef Čapek", "type": "P", "start": 0, "end": 11}],
                    "sentences": [[
            {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "maloval", "lemma": "malovat", "upos": "VERB", "head": 0, "deprel": "root", "start": 12, "end": 19},
            {"form": "obrazy", "lemma": "obraz", "upos": "NOUN", "head": 3, "deprel": "obj", "start": 20, "end": 26},
        ]]},
        ("d2", 0): {"entities": [{"text": "Karel", "type": "P", "start": 0, "end": 5},
                                 {"text": "Josefa Čapka", "type": "P", "start": 16, "end": 28}],
                    "sentences": [[
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 6, "end": 9},
            {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 15},
            {"form": "Josefa", "lemma": "Josef", "upos": "PROPN", "head": 3, "deprel": "nmod", "start": 16, "end": 22},
            {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 23, "end": 28},
        ]]},
    }
    g = build_graph(A)
    assert "Josefa Čapka" not in g.nodes
    bratr = g.facts_of("Josef Čapek", role="obj", predicate="bratr")
    assert bratr and g.participants(bratr[0], "subj") == ["Karel"]
```

a do `tests/test_tasks.py` (celá pipeline vč. recover a save/load):

```python
def test_build_fact_graph_resolves_person_case_variants(tmp_path):
    from config import Config, ServicesConfig, GraphConfig
    from jellyai.tasks import build_fact_graph
    ann = {
        ("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                   "sentences": [[
            {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
            {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
            {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
            {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
        ]]},
        ("d", 1): {"entities": [{"text": "Josef", "type": "P", "start": 0, "end": 5},
                                {"text": "Boženu Němcovou", "type": "P", "start": 11, "end": 26}],
                   "sentences": [[
            {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "ctil", "lemma": "ctít", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 10},
            {"form": "Boženu", "lemma": "Božena", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 17},
            {"form": "Němcovou", "lemma": "Němcová", "upos": "PROPN", "head": 3, "deprel": "flat", "start": 18, "end": 26},
        ]]},
    }
    ann_path = str(tmp_path / "ann.pkl")
    with open(ann_path, "wb") as f:
        pickle.dump(ann, f)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=str(tmp_path / "graph.pkl"))
    graph = build_fact_graph(cfg)
    assert "Boženu Němcovou" not in graph.nodes
    assert graph.facts_of("Božena Němcová", role="obj", predicate="ctít")
```

- [ ] **Step 2: Ověřit FAIL**

Run: `.venv/bin/python -m pytest tests/test_fact_graph.py tests/test_tasks.py -q`
Expected: 2 failed (uzly zůstávají roztříštěné), ostatní pass.

- [ ] **Step 3: Implementace** — `jellyai/graph/graph.py`, konec `build_graph`:

```python
    _decompose_dates(graph)
    resolve_entities(graph)
    return graph
```

a `jellyai/tasks.py`: horní import rozšířit na
`from jellyai.graph.graph import build_graph, resolve_entities, FactGraph`
a v `build_fact_graph`:

```python
    graph = build_graph(annotations)
    from jellyai.graph.recover import recover_entities
    recover_entities(annotations, graph)      # role ②: doplnit tituly, co NER minul
    resolve_entities(graph)   # recover bere podměty ze surového povrchu → srovnat znovu
    graph.save(config.graph.graph_path)
    return graph
```

- [ ] **Step 4: Ověřit PASS + celá suita**

Run: `.venv/bin/python -m pytest -q`
Expected: vše zelené. Kdyby cokoli z existujících testů spadlo (fakta se přestěhovala), NEopravovat test — zastavit se a analyzovat (mantinel proti whack-a-mole).

- [ ] **Step 5: Etalon (resolver poprvé nad reálnými daty — přes rebuild v tmp? Ne: etalon čte uložený `data/graph.pkl`, ten se přestaví až v Tasku 5. Tady jen kontrola, že se nic nerozbilo se STARÝM grafem.)**

Run: `.venv/bin/python benchmark/run_etalon.py`
Expected: `JÁDRO: 9/11` (beze změny — graf na disku je starý).

- [ ] **Step 6: Commit**

```bash
git add jellyai/graph/graph.py jellyai/tasks.py tests/test_fact_graph.py tests/test_tasks.py
git commit -m "feat(graph): resolve_entities zapojen do build_graph i build_fact_graph (po recover)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Query-side kmenový fallback v `_resolve_topic`

Mantinel 5 (build i query týž mechanismus): termín dotazu, jehož lemma zůstane skloňované („Galéna"), musí najít kanonický uzel („Galén"). Kmenová shoda je **fallback dimenze skóre** — nikdy nepřebije přesnou/case-insensitive shodu, jen zachrání uzly, které by jinak vypadly.

**Files:**
- Modify: `jellyai/answerer/graph_answerer.py:59-90` (`_resolve_topic`; import `_stem`)
- Test: `tests/test_topic_resolve.py`

**Interfaces:**
- Consumes: `canon._stem` (import privátního jména napříč moduly je house style — viz `_clean_lemma`, `_SUBJ`).
- Produces: skóre `(exact_hits, ins_hits, stem_hits, len, weight)`; uzel projde, když `ins_hits > 0` **nebo** `stem_hits > 0`.

- [ ] **Step 1: Failing test** — do `tests/test_topic_resolve.py` přidat:

```python
def test_case_variant_term_resolves_by_stem():
    """Skloněný termín („Galéna" — lemma, které UDPipe nechá v pádu) najde
    kanonický uzel „Galén" kmenovým fallbackem (týž mechanismus jako build-side
    resolver: canon._stem)."""
    g = FactGraph()
    g.add_fact(make_fact("léčit", [Participant("subj", "Galén", "person"),
                                   Participant("obj", "malomocenství", "concept")]))
    q = "Co léčil Galéna?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "léčil", "lemma": "léčit", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Galéna", "lemma": "Galéna", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "malomocenství"
```

- [ ] **Step 2: Ověřit FAIL**

Run: `.venv/bin/python -m pytest tests/test_topic_resolve.py -q`
Expected: nový test FAIL (odpoví fallback „V textu…"), oba staré PASS.

- [ ] **Step 3: Implementace** — `jellyai/answerer/graph_answerer.py`: k importům přidat `from jellyai.graph.canon import _stem` a přepsat `_resolve_topic`:

```python
    def _resolve_topic(self, topic_terms):
        """Najde uzel tématu otázky — nejlepší shodu s obsahovými lemmaty.

        **Přesná shoda velikosti má přednost**, case-insensitive je fallback:
        UDPipe někdy velikost zachová (PROPN „Babička" → lemma „Babička" — odliší
        knihu od obecné „babička"), jindy lemmatizuje na malá („Vějíř"→„vějíř" —
        pak by kapitalizovaný uzel byl jinak jménem nedostupný). Třetí patro je
        **kmenová shoda** (`canon._stem` — týž mechanismus jako build-side
        resolver): skloněný termín („Galéna") tak dosáhne na kanonický uzel
        („Galén"), aniž by kdy přebil přesnější shodu. Dál preferuje uzel
        pokrývající **víc témat** (aby „Božena Němcová" přebila „Němcová"), delší
        (víceslovnou) entitu a nakonec vyšší frekvenci.

        Args:
            topic_terms (list[str]): Obsahová lemmata otázky.

        Returns:
            str | None: Id uzlu tématu, nebo None když nic nesedí.
        """
        terms = [t for t in topic_terms if t]
        low_terms = [t.lower() for t in terms]
        stems = [_stem(t) for t in terms]
        best_id, best_score = None, None
        for node in self.graph.nodes.values():
            low_id = node.id.lower()
            low_words = low_id.split()
            ins_hits = sum(1 for t in low_terms if t == low_id or t in low_words)
            node_stems = {_stem(w) for w in low_words}
            stem_hits = sum(1 for s in stems if s in node_stems)
            if ins_hits == 0 and stem_hits == 0:
                continue
            words = node.id.split()
            exact_hits = sum(1 for t in terms if t == node.id or t in words)
            # přesná > case-insensitive > kmenová; pak témata, délka, váha
            score = (exact_hits, ins_hits, stem_hits, len(low_words), node.weight)
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
        return best_id
```

Proč to nemění žádné dnešní pořadí: u uzlů, které dnes projdou, je `stem_hits >= ins_hits` shodně korelované (přesná shoda slova ⇒ shoda kmene), takže prefix `(exact, ins)` rozhoduje stejně jako dřív; kmen jen (a) pouští do hry uzly s `ins_hits == 0` a (b) rozhoduje shody, které dřív spadly na `len`.

- [ ] **Step 4: Ověřit PASS + celá suita**

Run: `.venv/bin/python -m pytest -q`
Expected: vše zelené (zejména oba staré testy v `test_topic_resolve.py` — „Babička" vs „babička" přesnou shodou, „Vějíř" case-insensitive).

- [ ] **Step 5: Etalon**

Run: `.venv/bin/python benchmark/run_etalon.py`
Expected: `JÁDRO: 9/11` (starý graf na disku; nic nekleslo).

- [ ] **Step 6: Commit**

```bash
git add jellyai/answerer/graph_answerer.py tests/test_topic_resolve.py
git commit -m "feat(answer): kmenový fallback v _resolve_topic (týž mechanismus jako resolver)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Rebuild grafu, etalon 10/11, normativní případ rekurze, dokumentace

**Files:**
- Rebuild: `data/graph.pkl` (git-ignored, jen lokálně)
- Modify: `benchmark/etalon.jsonl` (+1 rekurzní případ)
- Modify: `jellyai/graph/canon.py:1-19` (docstring — už není PARKOVÁNO)
- Modify: `docs/superpowers/specs/2026-07-16-entity-identification-blocker.md` (status)

**Interfaces:**
- Consumes: celou pipeline z Tasků 1–4.

- [ ] **Step 1: Rebuild reálného grafu**

Run: `.venv/bin/python cli.py graph`
Expected: `Faktový graf: N uzlů, M faktů → data/graph.pkl` (N znatelně < 255+ osob — clustery se slily).

- [ ] **Step 2: Sonda — ověřit sloučení a kanonické tvary**

```bash
.venv/bin/python - <<'EOF'
from jellyai.graph.graph import FactGraph
g = FactGraph.load("data/graph.pkl")
for name in ["Josefa Čapka", "Karla Čapka", "Boženy Němcové"]:
    assert name not in g.nodes, name
for name in ["Josef Čapek", "Karel Čapek", "Božena Němcová",
             "Karel Antonín Čapek", "Antonína Čapka", "Čapek"]:
    assert name in g.nodes, name
bratr = g.facts_of("Josef Čapek", role="obj", predicate="bratr")
print("bratr:", [p.node for p in bratr[0].participants])
print("narodit Josef:", [p.node for f in g.facts_of("Josef Čapek", role="subj", predicate="narodit") for p in f.participants])
EOF
```

Expected: asserty projdou; bratr-fakt spojuje `Karel Antonín Čapek` ↔ `Josef Čapek`; narodit obsahuje `Hronově`.

- [ ] **Step 3: Etalon — cíl 10/11**

Run: `.venv/bin/python benchmark/run_etalon.py`
Expected: „Kdo byl bratr Josefa Čapka?" → PASS (obsahuje „Karel"); žádný dřívější PASS nespadl → `JÁDRO: 10/11`. („Jaký je robot?" zůstává pre-existing FAIL.)

- [ ] **Step 4: Ruční důkaz rekurze (ne fallback!)**

```bash
.venv/bin/python - <<'EOF'
from config import Config
from jellyai.tasks import make_graph_answerer
a = make_graph_answerer(Config())
ans = a.answer("Kde se narodil bratr Karla Čapka?", [])
print(ans.text, "| zdroj:", ans.sources, "| trace:", a.last_trace)
EOF
```

Expected: text obsahuje `Hronov` (Josefovo rodiště, NE Karlovo z fallbacku), zdroj `['graf']`, trace ukazuje `narodit` nad `Josef Čapek`.

- [ ] **Step 5: Normativní případ do etalonu** — do `benchmark/etalon.jsonl` přidat řádek (kategorie rekurze; obě strany bratr-symetrie už v etalonu jsou):

```json
{"q": "Kde se narodil bratr Karla Čapka?", "expect": ["Hronov"], "cat": "rekurze"}
```

Run: `.venv/bin/python benchmark/run_etalon.py`
Expected: `JÁDRO: 11/12`.

- [ ] **Step 6: Dokumentace** — v `jellyai/graph/canon.py` nahradit „⚠️ PARKOVÁNO…" odstavec docstringu aktuálním stavem (cluster_key/_stem zapojené v `resolve_entities` + query-side `_resolve_topic`; `build_entity_canon` zůstává nezapojený obecný nástroj). Do handoff specu doplnit pod nadpis status řádek: vyřešeno cestou A, kanonický tvar = lex-min (subj_count vyvrácen pro-drop znečištěním), datum, výsledek etalonu.

- [ ] **Step 7: Finální verifikace**

Run: `.venv/bin/python -m pytest -q` → vše zelené; `.venv/bin/python benchmark/run_etalon.py` → `JÁDRO: 11/12`.

- [ ] **Step 8: Commit**

```bash
git add benchmark/etalon.jsonl jellyai/graph/canon.py docs/superpowers/specs/2026-07-16-entity-identification-blocker.md
git commit -m "feat(etalon): normativní rekurzní případ + docs: blocker identifikace entit vyřešen (cesta A)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Rollback / stopky

- Kterýkoli krok, kde etalon klesne pod 9/11 nebo se rozsype existující test: **stop, analyzovat, neopravovat testy podle kódu** (přesně tudy vedla whack-a-mole cesta z handoffu).
- Kdyby lex-min na reálných datech vybral ne-nominativ (Step 2 sondy), tie-break rozšířit, ne hackovat: `min(members, key=lambda m: (len(m), m))` je první kandidát (nominativ je nejkratší i lex-min zároveň ve všech pozorovaných clusterech).
