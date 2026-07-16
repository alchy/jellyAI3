# Faktový graf — jádro (návrh)

**Datum:** 2026-07-16 · **Větev:** `feature/fact-graph`
**Status:** Schváleno (design), čeká na spec-review uživatelem

## 1. Cíl a kontext

Retrieval (V1–B1) hledá pasáž a odpovídá z ní. To selhává, když je odpověď
roztroušená nebo když jedna věta zavádí (B1: „kdy se narodil Karel Čapek? → 1915"
místo 1890). Faktový graf jde jinou cestou: z celého korpusu poskládá **vážený graf
faktů** (uzly = pojmy/entity, hrany = vztahy podmět–sloveso–předmět/atribut) a
odpovídá **průchodem grafu**. Klíč: **váha hrany = kolikrát se fakt v korpusu
opakuje**, takže opakovaně potvrzený fakt (1890) přebije jednorázový šum (1915).

Materiál je hotový — pro každou větu už máme UD rozbor + NameTag entity (větné
anotace z B1). Graf je jejich **agregace přes celý korpus**, ne nový model.

**Dělba práce:** *jellyAI3* postaví vážený graf a odpovídá; **viewBase** (grafový SW
uživatele, Three.js + d3-force-3d) udělá force layout, shluky a později i konverzační
„těžiště". Fyziku tedy nestavíme — exportujeme do viewBase.

## 2. Klíčová rozhodnutí (schválená)

- **Uzly + hrany = trojice** podmět–sloveso–předmět, plus atributy (datum/číslo/místo
  přes `obl`/`nmod`) a spona (`X být Y` = definice). Ne ko-okurence (mlhavá), ne
  všechny závislosti (šum).
- **Váha = opakování.** Shodné trojice se sloučí, `váha = počet výskytů`. To je
  mechanismus „opakovaný fakt vyhrává".
- **Vlastní graf od nuly** (edukační, bez závislostí) — prostá struktura uzlů/hran.
- **Export do viewBase** tenkým adaptérem (`to_networkx`, líný import), ne přímá
  závislost jádra.
- **`GraphAnswerer`** jako nový `answerer.mode="graph"` s fallbackem na
  template/extraktivní.

## 3. Architektura a datový tok

```
větné anotace (B1)  ──►  extrakce trojic (per věta)  ──►  agregace + váhy
   (doc_id, idx věty)         nsubj–verb–obj/atribut         sloučit shodné trojice
                                                                   │
dotaz ──► analyze_question (V4a) ──► GraphAnswerer ──────────────► graf
              (typ, téma, sloveso)       průchod: hrana s max vahou
                                              │
                                    odpověď (uzel) / fallback
                                              
graf ──► to_networkx ──► viewBase (force layout, shluky, těžiště)
```

## 4. Extrakce trojic (`jellyai/graph/extract.py`)

Vstup = jedna anotace věty (`{"entities": [...], "sentences": [[token,...]]}`).
Pro každý sloveso-token `V` ve větě:
- **podmět** `S` = potomek `V` s `deprel ∈ {nsubj, nsubj:pass}`.
- **předmět** `O` = potomek `V` s `deprel ∈ {obj, iobj}`.
- **atributy** `A` = potomci `V` s `deprel ∈ {obl, obl:*, nmod}`, které jsou datum/
  číslo/místo (NameTag typ `t`/`g`, nebo UPOS `NUM`).
- `S+O` → hrana `(node(S)) —lemma(V)→ (node(O))`.
- `S+A` → hrana `(node(S)) —lemma(V)→ (node(A))` (zachytí „Čapek → narodit_se → 1890").

**Spona:** kořen `R` s potomkem `cop` a podmětem `S` → hrana `(node(S)) —být→ (node(R))`.

**`node(token)`** — pokud token leží uvnitř NameTag entity (podle offsetů), uzel =
kanonický text entity + typ entity; jinak uzel = **nominativní lemma** tokenu
(přes `selection._nominative`/`_clean_lemma`) + typ `concept` (číslo → `number`).

Výstup = seznam trojic `(src, relation, dst)` s typy uzlů.

## 5. Struktura grafu (`jellyai/graph/graph.py`)

Vlastní, prostá:
- `nodes: dict[str, Node]` — `Node(id, type, weight)`; `weight` = počet trojic, jichž se uzel účastní.
- `edges: dict[(src_id, relation, dst_id), int]` — hodnota = **váha (opakování)**.
- `build_graph(annotations) -> FactGraph` — projede všechny věty, extrahuje trojice,
  sloučí a nasčítá váhy.
- `neighbors(node_id, relation=None, incoming=False)` — hrany ven/dovnitř (pro průchod).
- `save`/`load` (pickle) → `data/graph.pkl`.

## 6. Odpovídání průchodem (`jellyai/answerer/graph_answerer.py`)

`GraphAnswerer(graph, client, fallback)` — odpovídá z **globálního grafu** (ne z
pasáží; `retrieved` se ignoruje, slouží jen fallbacku).

1. `qa = analyze_question(question, client)` → typ, `verb_lemma`, `topic_terms`, `is_copula`.
2. **Téma → uzel:** `topic_terms` se namapují na `node.id` (normalizovaně); vybere se
   uzel s nejvyšší vahou.
3. Podle typu otázky průchod (hrana s **nejvyšší vahou** vyhrává konflikty):
   - **Kdo/Co** (ne-spona): najdi hrany `(? —verb_lemma→ téma)` → vrať `src` s max vahou.
   - **Kdy**: z tématu hrany `(téma —*→ dst)` s `dst.type=time` → `dst` s max vahou.
   - **Kde**: totéž s `dst.type=geo`. **Kolik**: `dst.type=number`.
   - **Kdo/Co spona, Jaký**: hrany `(téma —být→ Y)` → `Y` s max vahou (definice).
4. Vrátí text uzlu jako odpověď; když nic nesedí → `fallback.answer(question, retrieved)`.

Reuse: `analyze_question` (V4a) a normalizace lemmat (`selection`).

## 7. Perzistence a CLI

- CLI `graph` — postaví graf z `data/annotations.pkl` (větné anotace) a uloží
  `data/graph.pkl`. Vypíše počty uzlů/hran.
- CLI `graph --view` — exportuje do viewBase (viz §8).
- `answerer.mode="graph"` → pipeline načte graf a použije `GraphAnswerer`.

## 8. Export do viewBase (`jellyai/graph/viewbase_export.py`)

- `to_networkx(graph) -> nx.DiGraph` — uzly s atributy `type`, `weight`; hrany s
  `relation`, `weight`. NetworkX **líný import** (jen při exportu), není závislost jádra.
- CLI `graph --view` → `Canvas.from_networkx(G)` (viewBase), váha hrany → tloušťka,
  typ uzlu → barva. viewBase spustí force layout a interakci.
- Alternativně (bez viewBase/networkx): `to_json(graph)` → `{nodes, edges}` k ručnímu
  načtení. (Jeden z obou; `to_networkx` je primární most.)

**viewBase je repo uživatele — smíme ho upravovat** a případné rozšíření přispět do
veřejného repa. Pro jádro stačí dnešní stav (frontend force layout). Rozšíření, které
z viewBase udělá **backendovou fyzikální službu** (spočítá layout a vrátí ho jellyAI3),
je plánováno až pro navazující vrstvu — viz §12.

## 9. Konfigurace (`config.py`)

- `AnswererConfig.mode` získá hodnotu `"graph"`.
- `GraphConfig` (nový): `graph_path="data/graph.pkl"`.

## 10. Testy (hermetické, bez modelů)

- **Extrakce:** z umělé anotace „Božena Němcová napsala Babičku." → trojice
  `(Božena Němcová) —napsat→ (Babička)`. Spona „Rossum je vynálezce." → `—být→`.
  Atribut „Čapek se narodil 1890." → `(Karel Čapek) —narodit→ (1890, number/time)`.
- **Váhy = opakování:** tři věty s 1890 a jedna s 1915 → hrana na 1890 má vyšší váhu.
- **Odpovídání:** „kdy se narodil Karel Čapek?" nad tímto grafem → **1890** (ne 1915).
  „kdo napsal Babičku?" → „Božena Němcová". „kdo je Rossum?" bez spony → fallback.
- **Export:** `to_networkx` vrátí graf se správným počtem uzlů/hran a atributy vah.

## 11. Ošetření chyb a hraniční případy

- Věta bez podmětu/slovesa → žádná trojice (přeskoč).
- Téma otázky není v grafu → `fallback`.
- Více cílů se stejnou vahou → deterministicky první (stabilní řazení podle id).
- Token mimo entitu i bez lemmatu → přeskoč uzel.
- Prázdný graf → `GraphAnswerer` vždy `fallback`.

## 12. Mimo rozsah (YAGNI) — a navazující vrstva

Jádro = graf + statické odpovídání + export k vizualizaci. Mimo něj (navazující fáze):

- **Force layout / dimenzionální rozložení** — dělá viewBase. Navazující krok:
  rozšířit viewBase o **backendovou fyzikální službu**, která layout spočítá a vrátí
  jellyAI3 (pozice, shluky) k programové konzumaci — dnes to viewBase neumí (fyzika je
  ve frontendu). Uděláme to v repu uživatele a přispějeme do veřejného repa.
- **Konverzační „těžiště"** (hmotnost témat, posun v rozhovoru) — postaví na vrácené
  fyzice + na B2: aktivní téma dostane větší hmotnost, těžiště se posune, biasuje další
  odpovědi.
- Koreference (spojení „on/ona" s entitou), víceskoký průchod (A→B→C) — později.
- Vektorové vážení uzlů/hran (hybrid) — až po ověření faktového jádra.
