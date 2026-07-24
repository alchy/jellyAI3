# Autorský manuál — PROGRAMOVÉ PŘÍLOHY knihy (dil-N-program.html)

Ke každému kódově významnému dílu vzniká **programová příloha** `dil-N-program.html`. Cíl (zadání
uživatele): *„popisovat principy obecně nad reálnou implementací"* a *„přílohy každé kapitoly musí být
relevantní pro vývojáře"*. Není to referenční výpis funkcí — je to **princip → jak ho konkrétní funkce
realizuje → reálná ukázka z korpusu**.

Píše se do `experiments/hypothesis-one/docs/kniha/`. **NEcommituj** — zápis souboru stačí, review dělá hlavní model.

---

## 0. Zlaté pravidlo: princip NAD implementací

Každý blok přílohy má tři vrstvy, VŽDY v tomto pořadí:

1. **Princip** (obecně, 2–4 věty): jaký problém to řeší a jaká je myšlenka — nezávisle na jazyce/kódu.
2. **Kontrakt** (jak to kód realizuje): signatura funkce `název(vstup) → výstup`, co bere, co vrací,
   vedlejší efekt (mění na místě / zapisuje soubor / jen počítá). Použij `<div class="sig">…</div>` a `<p class="io">`.
3. **Reálná ukázka** (nezbytné): skutečný vstup a skutečný výstup z korpusu — z bloku §5 níže, NIC nevymýšlej.

Pravidlo „ukaž, nevyprávěj": u každého mechanismu konkrétní soubor, konkrétní funkce, konkrétní výstup.
Vývojářská relevance = čtenář po přečtení ví, KTEROU funkci zavolat, CO jí dát, CO dostane a KDE v kódu to je.

---

## 1. HTML kostra + sidebar

Použij PŘESNĚ kostru z `AUTOR-GUIDE.md` odd. 1–2 (stejný `<head>`, `assets/…`, `.layout`, topbar, sidebar).
Titulek stránky: `<title>Díl N — programová příloha — hypothesis-one</title>`.

**Sidebar:** vlož doslova blok z `AUTOR-GUIDE.md` odd. 2, ale za řádek příslušného dílu přidej pod-odkaz na přílohu:
```html
<p class="part">Díl III — Cesta dat</p>
<a href="dil-3.html">III · Fáze pipeline</a>
<a href="dil-3-program.html" class="active">III · programová příloha</a>
```
(`class="active"` jen na té stránce, kterou právě píšeš.)

**Obsah:** `<nav class="crumbs"><a href="index.html">Úvod</a> › <a href="dil-3.html">Díl III</a> › programová příloha</nav>`,
pak `<p class="eyebrow">Díl N — programová příloha</p>`, `<h1>`, `<p class="lead">` (na koho a co příloha cílí),
sekce `<h2 id="…">`, na konci `<div class="pager">` (zpět na narativní díl + na další přílohu).

## 2. Dostupné CSS třídy

- `<div class="sig">match(predicate, hole_role) → [(lemma, Fact)]</div>` — signatura (mono, rámeček).
- `<p class="io"><b>vstup</b> … <b>výstup</b> … <b>efekt</b> …</p>` — kontrakt slovy.
- `<aside class="rationale">` — zdůvodňovací blok (princip/proč), `<aside class="open-question">` — co doprogramovat.
- `<div class="tablewrap"><table>` — kontrakty více funkcí, klíče cache, klíče cs.json.
- `<pre class="mermaid">` — diagram (pravidla v AUTOR-GUIDE odd. 5). `<pre>` (ne mermaid) — reálný výstup/kód.
- Značky důvěry `<span class="mark mark-overeno">Ověřeno</span>` / `mark-prevzato` / `mark-navrh`.
  **Ověřeno jen s důkazem** (spuštěno v této session / hlídá test). Cesty/formáty, které jsi NEOVĚŘIL greppem, označ
  `<span class="todo">OVĚŘIT: …</span>` — nehádej.

## 3. Povinná disciplína

- Před psaním PŘEČTI odpovídající starou stránku v `docs/` (mapa v §7) A zdrojový kód (nadřazený dokumentaci při rozporu).
- **Gap-check:** co je ve staré stránce vývojářsky užitečné a v nové knize (narativ) NENÍ → přenes do přílohy (vlastními slovy).
- Signatury ber z §4 (jsou vytažené z kódu). Cesty souborů (kam se ukládá cache/fakty) OVĚŘ greppem, než napíšeš.
- Reálné výstupy ber z §5. Když potřebuješ jiný, spusť `python` proti kódu — nevymýšlej.
- Čísla ber z AUTOR-GUIDE odd. 6. Kde měření není, napiš „neměřeno".

---

## 4. REÁLNÉ SIGNATURY (vytaženo z kódu, ke commitu d20a5ec)

**answering.py** — živý tah (Díl IV):
- `answer(question, carry_context=False, hole_override=None, return_features=False) → dict|None` — celý tah; dict má `answer, mode, via, …`.
- `_parse(question) → [token]` · `_question(toks) → (q_vzor, lemmas, hole_role)` · `_predicate(toks) → str|None`
- `_answer_role(toks, hole_role) → role` · `_relation_query(toks) → str|None`
- `_candidates(q_vzor, predicate, hole_role, known) → [dict]` · `_assurance(cands, known_lemmas, hole_role) → (mode, best, offer)`
- `_polar(toks) → dict|None` · `_copula_states(subj) → [lemma]` · `_hot_entities(top=6, floor=0.4, idf_min=1.5) → [lemma]`
- `_canon(lemma, doc) → lemma` (alias) · `_gate_feats(best, cands, known_lemmas, hole_role) → [float]` · `_gate_p(feat) → float`

**dataloader.py** — data + brána ② (Díl III):
- `build_indexes() → dict` (postaví tf·idf, subjects, doclinks, aliases; zapisuje pkl) · `load_idf() → {"idf":{…}}`
- `select_files(query_lemmas, top_files=None) → [(doc, score)]` (② brána — které soubory rozsvítit)
- `load_shard(doc_id) → shard` · `mount(doc_ids) → corpus` · `document_ids() → [doc]`

**grammar_vzor.py** — VZOR + role (Díl III):
- `frame_sig(toks, i, modality, r=None) → str` (VZOR = řetězec slotů kolem pivotu) · `slot(tok) → str`
- `role_key(deprel, upos, feats, prep="", nominal_pred=False) → role` · `canon_lemma(tok) → lemma`
- `is_factual(tokens) → bool` · `sentence_modality(toks) → "." | "?"` · `prep_of(sent, tid) → str`

**fact_store.py** — fakty (Díl IV): `Fact(predicate, roles, doc, sent, text=None)`; `match(predicate, hole_role) → [(lemma, Fact)]`;
`mount(doc_ids)` · `append(fact)` · `as_row()/from_row()`.

**template_store.py** — VZOR-šablony (Díl III/IV): `QueryTemplate(fact_vzor, query_vzor, role, answer, surface_q, fact_ref, origin)`;
`match_scored(query_vzor) → [(tpl, shoda)]` · `match(query_vzor)` · `match_fact(fact_vzor)` · `mount/unmount`.

**activation_field.py** — světlo (Díl IV): `build_graph(corpus, grammar)` · `feed(question_lemmas, dataloader)` ·
`spread(hops=None)` · `decay()` · `weight_answer(answer, doc) → float` · `weight(tpl) → float` · `reinforce(answer, doc)`.

**Extrakce (Díl III):** `extract_bio.bio_facts(sent, grammar) → [(pred, roles)]` · `extract_relations.relation_facts(sent, grammar, holder=None) → [(pred, roles)]`
· `chronos.temporal_facts(sent, grammar) → [(pred, roky)]` · `synth_registry` (třída, `.run()`) · `build_facts_all.main()`.

**fill_holes.py** — koreference/díry (Díl IV): `detect(sent) → [díra]` · `resolved_subjects(sentences) → {(klíč,verb):(lemma,konf)}`
· `resolve_document_identity/zone/activation(sentences)` · `build_zone(corpus=None)`.

**viz.py + jellyai/viz/viewbase_view.py** — viewBase (Díl VII): viz `process(answering, q)`, `build_view(answering) → (view, on_ask)`,
`render(st)`, `_claim_singleton()`. Adaptér `ViewBaseView(title, theme="cyber")`: `add_node/update_node/remove_node`,
`add_edge`, `flow(path)`, `focus(node_id)`, `feed_fact/from_graph`, `open_terminal(on_input)/write`, `open_docs_panel/write_docs`,
`open_nodes_panel/write_nodes`, `serve(open_browser, block)`, `stop()`.

---

## 5. REÁLNÉ VÝSTUPY Z KORPUSU (ověřeno v této session — použij doslova)

**Token tabulka** — `wiki_karel_čapek`, věta 0, po UDPipe (`answer._parse` / cache shard):
```
 #  form         lemma      upos  deprel  head  NameType
 1  Karel        Karel      PROPN nsubj   25    Giv
 2  Čapek        Čapek      PROPN flat    1     Giv
 3  ,            ,          PUNCT punct   5
 4  rodným       rodný      ADJ   amod    5
 5  jménem       jméno      NOUN  nmod    1
 6  Karel        Karel      PROPN appos   1     Giv
 …
12  ledna        leden      NOUN  dep     6
```
(slovo → `form`; základní tvar → `lemma`; slovní druh → `upos`; závislost → `deprel`+`head`; jmenná entita → `feats.NameType=Giv`)

**VZOR (frame_sig)** kolem pivotu `spisovatel` (root věty 0): `ADJ:Nom·NOUN:Nom·,·.`
(sloty = `upos:pád` sousedů v okně šířky r kolem pivotu, oddělené `·`)

**Fakty** vytěžené z `wiki_karel_čapek` (fact_store): `zemřít {who:[Karel], when:[1938], where:[Praha]}` ·
`bratr {who:[Josef], whose_of_what:[Karel]}` (vztah = hrana, symetrický).

**Živé odpovědi** (celý tah `answer()`):
```
Kdo je Karel Čapek?          → spisovatel   [answer] via=pred
Kdy se narodil Karel Čapek?  → 1890         [answer] via=pred
Kde se narodil Karel Čapek?  → Svatoňovice  [answer] via=pred
```

**Aktivační pole** (`activation_field.py __main__`, mount karel+pes, feed ["spisovatel","drama"], spread):
horká slova po spreadu = sousedé v okně (idf×idf×proximity); horké soubory = skóre z `select_files`.

**UDPipe cache** — `data/corpus/<doc>.pkl` (53 shardů). Shard = `dict{(doc, idx): {"entities":…, "sentences":[…]}}`;
`wiki_karel_čapek`… `bible_1_jan.pkl` má 129 záznamů. Klíč = `(doc_id, index)`.

---

## 6. Aktivace uzlů/hran/souborů — přesný mechanismus (pro dil-4-program, R9)

Ze `activation_field.py` (doslova z kódu):
- **Uzly** = `words` (`lemma → teplo`) a `files` (`doc → teplo`). **Hrany** = `adj` (`lemma → {soused: vodivost}`).
- **Stavba grafu** `build_graph(corpus, grammar)`: pro každou větu vezme content-lemmata (bez PUNCT), v OKNĚ šířky
  `window` (config) přidá hranu s **vodivostí = idf(a) × idf(b) × exp(−(d)/tau)** (funkční slova nevedou; bližší silněji). Symetrická.
- **Rozsvícení** `feed(question_lemmas, dataloader)`: každé lemma otázky dostane teplo `+= idf(lemma)`; soubory z
  `select_files` dostanou teplo `+= score` (② brána). Kumuluje se přes tahy.
- **Šíření** `spread(hops)`: teplo uzlu teče do sousedů úměrně **normalizované** váze hrany × `spread_lambda`, `hops` skoků.
- **Útlum** `decay()`: mezi tahy `teplo *= decay` (slova i soubory) — světlo bez opory pohasne.
- **Váha kandidáta** `weight_answer(answer, doc) → teplo souboru + teplo odpovědní entity` (sdílené oběma cestami matcheru).
- **Zpětný tok** `reinforce(answer, doc)`: odpověď přihřeje svou entitu (+1) a soubor (+1) → řetěz drží téma napříč tahy.

Parametry jsou DATA v `config.json` bloku `activation` (`decay, spread_lambda, hops, window, tau`) — ne v kódu.

## 6b. cs.json — co, proč, kde se čte (pro dil-6-program, R7)

Jazyk je DATA (zásada „jazyk jako JSON"). `lang/cs.json`, načítá `grammar_vzor.__init__` do `self.LANG`
(`grammar_vzor.py:40`), cesta z `config.json["paths"]["lang"]`. Klíče (velikost | ukázka):
```
copula_lemma (1: být)        stopwords (9: kdo,co,být,…)      deprel_to_role (6: nsubj→who/what_subject)
role_ask (21: who→Kdo)       role_catalog (21: who→Podmět…)  role_prepositions (4: where→[v,na,u,…])
relation_nouns (15: bratr…)  symmetric_relations (7: bratr…) interrogative_slots (14: who→PRON:Nom)
relative_place_adverbs (4: kde,kam,…)  relative_time_adverbs (3: kdy,…)  clause_markers (4: aby→for_what_purpose)
participle_active/passive_suffixes  possessive_adj_suffixes (3: ův,ový,in)  uncertainty_markers (12)  deprel_structural (12)
```
Čtenáři (grep): `grammar_vzor.py` (role_key přes `deprel_to_role`, `role_prepositions`), `role_catalog.py`/`roles.py`
(role_catalog/role_ask), `extract_relations` (relation_nouns/symmetric), `phase1.py`. Ukázka „proč": přidáš `švagr`
do `relation_nouns` + `symmetric_relations` → extrakce vztahů ho začne brát jako hranu, BEZ zásahu do kódu.

## 6c. viewBase integrace (pro dil-7-program, R5)

`viz.py` je JEDINÉ místo, kde se viewBase importuje (`from jellyai.viz.viewbase_view import ViewBaseView`, líný adaptér).
- `process(answering, q)`: zavolá `answering.answer(q, carry_context=True)` (mount+světlo přetrvá mezi tahy — to je přesně,
  co vizualizace ukazuje) a vrátí stav pole + odpověď.
- `build_view(answering) → (view, on_ask)`: postaví `ViewBaseView`, otevře terminál (`open_terminal`) a panely dokumentů/uzlů.
- `render(st)`: **dynamický loader** — odebere uzly, které vychladly (`remove_node`), přidá nové horké (`add_node`,
  jas = aktivace), nakreslí tok otázka→odpověď (`flow`), barva dle režimu (answer/clarify/unsure).
- `_claim_singleton()`: self-restart přes `.viz.pid` (zabije starou instanci, zapíše PID) — jeden živý server.
- Adaptér mluví s prohlížečem přes WebSocket (živé updaty). hypothesis-one má viewBase jako VOLITELNÝ výstup — běh
  bez něj (ask.py) funguje stejně; viewBase je jen okno do stejného pole.

## 6d. Chronos / Mnemos / Topos (pro dil-4-program, R12)

Tři pojmenované subsystémy z původního konceptu (parent). Popiš u každého: **co umí dnes · jak vázán · jak doprogramovat.**
- **Chronos (čas)** — ŽIJE: `chronos.py`, `temporal_facts(sent, grammar) → [(pred, roky)]`; váže se ve `build_facts_all`
  (temporal fakty vedle bio/relations), rozpozná roky (`_is_year`). Dnes: narození/úmrtí/vydání. Doprogramovat:
  relativní čas, intervaly, řazení událostí. (Ověř greppem, co přesně dnes extrahuje.)
- **Topos (místo/prostor)** — ČÁSTEČNĚ: prostorové role (`where`) přes `extract_bio._place` + gazetteer; váže se do faktů
  jako role `where`. Dnes: místo narození/děje. Doprogramovat: gazetteer seed (chybějící obce), směr (k+dativ), vztahy míst.
- **Mnemos (paměť/zápis)** — NENÍ (návrh): write-path, aby si systém pamatoval z dialogu (parent má memory.jsonl).
  Dnes v hypothesis-one nezabudováno. Doprogramovat: zápis vyřešených faktů zpět do mapy, projekce event-logu.
  Označ `<span class="mark mark-navrh">Návrh</span>` + `<aside class="open-question">`.

Ověř aktuální stav greppem (`chronos.py`, `extract_bio.py`, `gazetteer`, `memory`) — nepiš z hlavy, kde jde o „dnes".

---

## 7. Mapa STARÉ docs/ → díl (gap-check, R6) — přečti odpovídající stránku

| příloha | stará docs/*.html k porovnání |
|---|---|
| **dil-3-program** | faze-1-preprocessing, faze-2-dataloader, faze-3-grammar, faze-4-synthesis, faze-5-queryparser, **anotacni-cache**, dynamicke-nahravani, **vzor-matcher**, terminy |
| **dil-4-program** | **aktivace-dialog**, dialogovy-automat, koreference, koreference-do-faktu, diry, aliasy-identita, naucene-smerovani |
| **dil-6-program** | terminy, externi-zdroje |
| **dil-7-program** | anotacni-cache (deploy část), externi-zdroje, pouziti |

Co je tam vývojářsky užitečné a v nové knize chybí → přenes do přílohy (vlastními slovy, ne kopií). Co je zastaralé/nahrazené → ignoruj.

## 8. Rozdělení příloh (kdo co pokrývá)

- **dil-3-program.html** (Cesta dat): R2 slovo→lemma→druh→token · R3+R8 VZOR formát/uložení/načtení/stavba ·
  R10+R11 UDPipe cache proč/jak/nakládání · kontrakty annotate/dataloader/grammar/extract/synth · reálné ukázky §5.
- **dil-4-program.html** (Jak funguje): R9 aktivace souborů/uzlů/hran (§6) · R12 Chronos/Mnemos/Topos (§6d) ·
  kontrakty answering/activation_field/fact_store/template_store/fill_holes · reálné ukázky §5.
- **dil-6-program.html** (Rozšiřování): R7 cs.json co/proč/kde (§6b) · kontrakty reindex/build_indexes · jak přidat vztah/pojem.
- **dil-7-program.html** (Provoz): R5 viewBase integrace (§6c) · kontrakty viz.py · vazba na běh bez viewBase.

Každá příloha: ≥1 zdůvodňovací blok, značky důvěry u každého kontraktu, ≥1 tabulka kontraktů, reálné ukázky. Bez commitů.
