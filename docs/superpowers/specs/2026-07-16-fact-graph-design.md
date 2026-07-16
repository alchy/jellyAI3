# Faktový graf — jádro (návrh)

**Datum:** 2026-07-16 · **Větev:** `feature/fact-graph`
**Status:** Schváleno (design v2 — reifikované n-ární fakty), čeká na spec-review

## 1. Cíl a kontext

Retrieval (V1–B1) hledá pasáž a odpovídá z ní. To selhává, když je odpověď
roztroušená nebo když jedna věta zavádí (B1: „kdy se narodil Karel Čapek? → 1915"
místo 1890). Faktový graf jde jinou cestou: z celého korpusu poskládá **vážený graf
faktů** a odpovídá **průchodem**. Klíč: **váha faktu = kolikrát se opakuje**, takže
opakovaně potvrzený fakt (1890) přebije jednorázový šum (1915).

**Vztah je uzel (reifikace).** Spoj mezi tématem a hodnotou není jen popisek hrany,
ale samostatný **faktový uzel**. Důvod: k témuž tématu i téže hodnotě může vést víc
různých faktů („Čapek se **narodil** 1890" vs „Čapku 1890 **začaly růst vlasy**"), a
jeden fakt může být **n-ární** (podmět + předmět + **čas + místo** najednou:
„narodil se v Praze 1890"). Reifikace to čistě odděluje, umožní na fakt pověsit
váhu/zdroj a dělá z faktu waypoint pro vizualizaci tras ve viewBase. Volíme
**maximální přesnost** (korpus poroste; výkon neřešíme).

Materiál je hotový — pro každou větu máme UD rozbor + NameTag entity (větné anotace
z B1). Graf je jejich **agregace**, ne nový model.

**Dělba práce:** *jellyAI3* postaví graf a odpovídá; **viewBase** (grafový SW
uživatele, Three.js + d3-force-3d) udělá force layout, shluky, animaci tras a později
konverzační „těžiště".

## 2. Klíčová rozhodnutí (schválená)

- **Reifikovaný n-ární faktový uzel.** Každá slovesná událost = jeden faktový uzel
  (nese `predicate`), k němu **role-hrany** na účastníky (podmět/předmět/čas/místo).
  Spona = fakt s `predicate="být"` (podmět + přísudek).
- **Váha = opakování na faktu.** Shodné fakty (týž predikát + titíž účastníci) se
  sloučí do jednoho uzlu, `váha += 1`. Opakovaný fakt vyhrává konflikty.
- **Vlastní graf od nuly** (edukační, bez závislostí).
- **Export do viewBase** tenkým adaptérem (`to_networkx`/`to_json`, líný import).
- **`GraphAnswerer`** (`answerer.mode="graph"`) — odpovídá **2-skokem** (téma → fakt →
  hodnota); při neúspěchu fallback na extraktivní/template.

## 3. Architektura a datový tok

```
větné anotace (B1) ─► extrakce faktů ─► agregace (sloučit shodné, váha++) ─► FactGraph
   (doc_id, idx)        1 sloveso = 1 fakt        téma/hodnota = entita/lemma
                        role: subj/obj/time/loc     fakt = uzel s predikátem+vahou

dotaz ─► analyze_question (V4a) ─► GraphAnswerer ─► 2-skok: téma → fakt(max váha) → hodnota
FactGraph ─► to_networkx ─► viewBase (force layout, shluky, animace tras, těžiště)
```

Graf je (skoro) bipartitní: **entitní/hodnotové uzly** ↔ **faktové uzly**; hrany mají
**roli** a spojují fakt s jeho účastníky.

## 4. Extrakce faktů (`jellyai/graph/extract.py`)

Vstup = anotace věty. Pro každý sloveso-token `V`, který má podmět:
- **role účastníků**: `nsubj/nsubj:pass → subj`, `obj/iobj → obj`, a `obl/nmod`
  podle typu cíle: čas → `time`, místo → `loc`, číslo → `num`.
- vznikne **jeden `Fact`**: `predicate = lemma(V)`, `participants = {(role, uzel)…}`.
- **spona**: kořen `R` s potomkem `cop` a podmětem `S` → `Fact("být", {(subj,S),(pred,R)})`.

**`node(token)`** (id, typ): uvnitř NameTag entity → kanonická entita + typ; jinak
nominativní lemma (`selection._clean_lemma`), typ `number` pro NUM, jinak `concept`.

**Identita faktu** (pro slučování/váhu) = `(predicate, seřazená n-tice (role, uzel))`.
Tři věty „Čapek se narodil 1890" → jeden faktový uzel s vahou 3. „…v Praze 1890"
(navíc `loc`) → jiný, bohatší fakt (samostatný uzel). Podřazení částečných faktů
(subsumpce) je pozdější zpřesnění.

Výstup = `list[Fact]`, kde `Fact(predicate, participants)`,
`Participant(role, node_id, node_type)`.

## 5. Struktura grafu (`jellyai/graph/graph.py`)

- `nodes: dict[str, Node]` — entitní/hodnotové uzly; `Node(id, type, weight)`,
  `weight` = v kolika faktech-výskytech figuruje.
- `facts: dict[key, FactNode]` — `FactNode(id=key, predicate, weight, participants)`;
  `key = (predicate, participants)`; `weight` = opakování faktu.
- `_by_node: dict[node_id, list[(fact_key, role)]]` — index pro průchod (v jakých
  faktech a roli uzel vystupuje).
- `add_fact(fact)` — sloučí podle klíče (`váha++`) nebo založí; udržuje `_by_node` a
  frekvence uzlů.
- `facts_of(node_id, role=None, predicate=None) -> list[FactNode]` — fakty, v nichž
  uzel figuruje (filtr role/predikát).
- `participants(fact_node, role) -> list[str]` — účastníci faktu dané role.
- `save`/`load` (pickle) → `data/graph.pkl`.

## 6. Odpovídání 2-skokem (`jellyai/answerer/graph_answerer.py`)

`GraphAnswerer(graph, client, fallback)` — z **globálního grafu** (`retrieved` jen pro
fallback).

1. `qa = analyze_question(question, client)` → typ, `verb_lemma`, `topic_terms`, `is_copula`.
2. **Téma → uzel** (`_resolve_topic`): `topic_terms` na `node.id`, uzel s max vahou.
3. **2-skok** (fakt s **nejvyšší vahou** vyhrává):
   - najdi fakty, kde téma vystupuje (role/predikát dle otázky),
   - z faktu s nejvyšší vahou vezmi účastníka cílové role.
   | Otázka | fakt (téma jako) | cílová role |
   |---|---|---|
   | Kdo/Co | obj (příp. subj), predikát = sloveso | subj (příp. obj) |
   | Kdy | subj, predikát = sloveso (jinak libovolný) | time / num |
   | Kde | subj | loc |
   | Kolik | subj | num |
   | Kdo/Co spona, Jaký | subj, predikát „být" | pred |
4. Nic → `fallback.answer(question, retrieved)`.

Příklady: „kde se narodil Karel Čapek?" a „kdy…?" vezmou **týž** narozovací fakt a
z něj `loc` resp. `time` — n-arita v akci. `1890` vyhraje nad `1915`, protože jeho
fakt má vyšší váhu.

## 7. Perzistence a CLI

- CLI `graph` — postaví graf z `data/annotations.pkl`, uloží `data/graph.pkl`, vypíše
  počty entitních/faktových uzlů a hran.
- CLI `graph --view` — export do viewBase (§8).
- `answerer.mode="graph"` (přepínač `--graph`) → pipeline použije `GraphAnswerer`.

## 8. Export do viewBase (`jellyai/graph/viewbase_export.py`)

- `to_json(graph)` → `{nodes, edges}`, kde **nodes** = entitní uzly *i faktové uzly*
  (faktový uzel označen typem `fact`, popiskem = predikát), **edges** = role-hrany
  `(fact, role, participant, weight)`.
- `to_networkx(graph) -> nx.DiGraph` (líný import) — primární most; váha → tloušťka,
  typ uzlu → barva, role → popisek hrany.

**viewBase je repo uživatele — smíme ho upravit** a rozšíření přispět do veřejného
repa. Umí zobrazovat **trasy** (data flows po hranách) → dotaz může vizuálně proletět
sítí přes faktové uzly (viz §12). Backendová fyzikální služba (spočítat layout a vrátit
jej jellyAI3) je také navazující krok.

## 9. Konfigurace (`config.py`)

- `AnswererConfig.mode` získá hodnotu `"graph"`.
- `GraphConfig` (nový): `graph_path="data/graph.pkl"`.

## 10. Testy (hermetické, bez modelů)

- **Extrakce:** „Božena Němcová napsala Babičku." → `Fact("napsat", {(subj,Božena
  Němcová),(obj,Babička)})`. Spona → `Fact("být", …)`. „Čapek se narodil v Praze 1890."
  → jeden `Fact("narodit", {(subj,Čapek),(loc,Praha),(time/num,1890)})`.
- **Váhy = opakování:** tři věty s narozením 1890, jedna se „životem" 1915 → narozovací
  fakt má váhu 3.
- **Odpovídání:** „kdy se narodil Karel Čapek?" → **1890**; „kde…?" z **téhož** faktu →
  Praha; „kdo napsal Babičku?" → Božena Němcová; „kdo je Rossum?" bez faktu → fallback.
- **Export:** `to_json` obsahuje faktové uzly (typ `fact`) i role-hrany s vahami.

## 11. Ošetření chyb a hraniční případy

- Věta bez podmětu/slovesa → žádný fakt.
- Téma otázky není v grafu → `fallback`.
- Více faktů se stejnou vahou → deterministicky první (stabilní řazení podle klíče).
- Token bez lemmatu i entity → účastník se přeskočí (fakt může vzniknout z ostatních).
- Prázdný graf → `GraphAnswerer` vždy `fallback`.

## 12. Mimo rozsah (YAGNI) — a navazující vrstva

Jádro = graf + statické odpovídání + export. Mimo něj:

- **Vizualizace trasy dotazu** *(nápad uživatele)*: `GraphAnswerer` už zná trasu
  (téma → faktový uzel → hodnota). Malé rozšíření: vracet i trasu a předat ji viewBase
  („trasy"), aby dotaz proletěl sítí. Blízký krok po ověření jádra.
- **Force layout / dimenzionální rozložení** — dělá viewBase; navazující krok =
  backendová fyzikální služba vracející layout jellyAI3 (přispět upstream).
- **Konverzační „těžiště"** (hmotnost témat, posun v rozhovoru) — na vrácené fyzice + B2.
- **Konsolidace kódu k grafu** *(nápad uživatele)*: když se graf osvědčí, zredukovat
  stávající vrstvy tímto směrem. **Podmíněné ověřením jádra.**
- Subsumpce částečných faktů, koreference, víceskoký průchod (A→B→C), vektorové vážení
  (hybrid) — pozdější zpřesnění.
