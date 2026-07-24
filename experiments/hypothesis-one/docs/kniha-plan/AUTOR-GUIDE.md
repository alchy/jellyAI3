# Autorský manuál — psaní kapitol knihy (fáze 4)

Tímto se řídí KAŽDÝ, kdo píše díl knihy. Cíl: díly od různých autorů vypadají jako jedna kniha.
Struktura: **jeden HTML soubor na DÍL** (`dil-1.html` … `dil-8.html`, `prilohy.html`); kapitoly
uvnitř jsou sekce `<h2 id="…">`. Vzor stylu = `docs/kniha/dil-4-1-dialogovy-automat.html`.

Píše se do `experiments/hypothesis-one/docs/kniha/`. NEcommituj — zápis souboru stačí.

---

## 1. Přesná HTML kostra stránky (vyplň jen `<div class="content">`)

```html
<!doctype html>
<html lang="cs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Díl N — NÁZEV — hypothesis-one</title>
<link rel="stylesheet" href="assets/style.css">
<script defer src="assets/mermaid.min.js"></script>
<script defer src="assets/script.js"></script>
</head>
<body>
<div class="layout">
  <!-- SIDEBAR: vlož přesně blok z odd. 2 -->
  <main class="main">
    <div class="topbar">
      <button class="iconbtn" id="theme-toggle" type="button" aria-label="Přepnout světlý/tmavý režim">◐ režim</button>
    </div>
    <div class="content">
      <!-- SEM obsah dílu -->
    </div>
  </main>
</div>
</body>
</html>
```

## 2. Sidebar (vlož DOSLOVA do každé stránky)

```html
  <aside class="sidebar">
    <p class="brand"><a href="index.html">hypothesis-one</a></p>
    <p class="tag">knižní dokumentace</p>
    <nav aria-label="Obsah knihy">
      <a href="index.html">Úvod</a>
      <p class="part">Díl I — Než začneme</p><a href="dil-1.html">I · Co řeší, slovník, mapa</a>
      <p class="part">Díl II — První kroky</p><a href="dil-2.html">II · Instalace, běh, data</a>
      <p class="part">Díl III — Cesta dat</p><a href="dil-3.html">III · Fáze pipeline</a>
      <p class="part">Díl IV — Jak to funguje</p><a href="dil-4.html">IV · Automat, aktivace, fakty</a>
      <p class="part">Díl V — Testování</p><a href="dil-5.html">V · Testy, tvary, měření</a>
      <p class="part">Díl VI — Rozšiřování</p><a href="dil-6.html">VI · Data, vztahy, testy</a>
      <p class="part">Díl VII — Provoz</p><a href="dil-7.html">VII · UDPipe, viz, pasti</a>
      <p class="part">Díl VIII — Otevřené body</p><a href="dil-8.html">VIII · Nedodělky a cesty</a>
      <p class="part">Přílohy</p><a href="prilohy.html">Rejstřík, konfigurace, registry</a>
    </nav>
  </aside>
```

Na aktivní díl přidej `class="active"` k jeho `<a>` (např. `<a href="dil-3.html" class="active">`).

## 3. Struktura obsahu dílu

- `<nav class="crumbs"><a href="index.html">Úvod</a> › Díl N</nav>`
- `<p class="eyebrow">Díl N — NÁZEV</p>` a `<h1>` s názvem dílu.
- `<p class="lead">` — otevírací „kde jsme a kam jdeme" (naváž na předchozí díl).
- Každá kapitola = `<h2 id="n-1">N.1 Název</h2>` … výklad.
- Konec dílu: `<h2>Co už umíš</h2>` (shrnutí) + `<div class="pager">` s odkazy předchozí/další díl.

## 4. Stylová pravidla (knižní tón)

- Souvislé ODSTAVCE, ne odrážky (odrážky jen pro skutečné výčty).
- Nový pojem v pořadí: **problém → intuice/analogie → přesná definice → kód/příklad → co se stalo**.
- Oslovuj čtenáře („teď spustíme", „všimni si"). Čtenář = Python-začátečník, NEZNÁ NLP ani projekt.
- Ukaž, nevyprávěj: konkrétní soubor, konkrétní výstup, co čtenář uvidí. Přesná slovesa (vrací/vypisuje/ukládá/mění na místě).
- ZAKÁZÁNO: „samozřejmě", „stačí jen", „jak je známo". Nedefinované zkratky. Kód bez vysvětlení výstupu.
- NEKOPÍRUJ formulace ze staré dokumentace — přepiš vlastními slovy.

## 5. Povinné prvky

- **≥1 Mermaid diagram na díl** (u workflow a datových modelů vždy). Pravidla:
  - Popisky uzlů VŽDY v uvozovkách: `A["Načtení korpusu"]`. Bez toho parser padne na diakritice/závorkách.
  - ID uzlů/stavů ASCII bez diakritiky; český text jen v uvozovkovém popisku (`state "…" as S1`).
  - Nejvýš ~15 uzlů. Přidej `accTitle:` a `accDescr:`. Nad diagramem odstavec „co hledat", pod ním slovní popis.
  - Zdroj v `<pre class="mermaid">…</pre>`. Typy: architektura `flowchart LR` se subgraph; workflow `sequenceDiagram`;
    cesta dat `flowchart TD`; datový model `classDiagram`/`erDiagram`; stavy `stateDiagram-v2`; ladění `flowchart TD` s větvením.
- **Zdůvodňovací blok** `<aside class="rationale">` — v Dílech III–VI ≥1 na díl. Struktura (viz vzor IV.1):
  Rozhodnutí · Problém · Mechanismus (příčina-následek) · Zvažované alternativy (≥2, konkrétní důvod odmítnutí) · Kdy přestane platit · Cena.
- **„Co nefunguje a proč"** — ke každému workflow/rozšiřujícímu bodu: první nápad nováčka, PROČ selže (mechanismus), jak se selhání projeví (tichá chyba/pomalý běh).
- **Značka důvěry** u postupů: `<span class="mark mark-overeno">Ověřeno</span>` (jen s důkazem — test/spuštění),
  `mark-prevzato` (odpovídá kódu, nespuštěno), `mark-navrh` (nápad). U appendixů „tabulka ověření".
- **Tabulky** obal do `<div class="tablewrap">`. Datové modely, konfigurace, příkazy, chyby → tabulka.
- Neznámé/neověřené → `<span class="todo">OVĚŘIT: …</span>`, ne domněnka.

## 6. SKUTEČNÁ ČÍSLA — používej JEN tato, NIC nevymýšlej

Zdroj: měření v projektu (eval_large.py/eval_answers.py, ke commitu b8b2d5c…d20a5ec). Kde měření chybí, napiš „neměřeno".

- **Velký etalon (gold_large.json, 145 otázek, eval_large.py):** PASS **63/145 = 43 %**; confident-wrong **30**.
  Po kategoriích: honest-negative 7/7 (100 %), copula 16/21, spatial 12/17, polar 9/14, temporal 9/25,
  authorship-who 6/20, relation 4/18, attribute 0/11, count 0/7, authorship-what 0/3, taxonomy 0/2.
  Formulační robustnost: 8/14 skupin parafrází konzistentních.
- **Malý etalon (gold_answers.json, 25 otázek):** 18/25 = 72 % (HIT 18, WRONG 2, CLARIFY 4, NO_QVZOR 1).
- **Naučená brána (gate_pilot, LOO-CV, 77 answer-mode případů):** přesnost 68.8 %; práh 0.30 → −13 confident-wrong / −3 správné (LOO).
  NASAZENÍ přeučuje (76 vzorků × 13 rysů; reálný běh −19 potlačených) → **gate_enabled=false (vypnutá)**.
- **NN router pilot (nn_pilot.py, LOO-CV, 20 strukturních rysů):** 75.9 % vs 17.2 % (majorita).
- **Oracle routing-strop:** +4 PASS / −6 confident-wrong (routing je minor páka).
- **Křivka velkého etalonu:** 33 → 40 (polar+extrakce) → 43 % (vztahy); confident-wrong 35 → 30.
- **Fakty:** 16028 registry + 24 bio-závorky + 114 vztahových hran, nad 52 soubory korpusu.

Ke každému číslu v knize: data | velikost sady | příkaz pro zopakování | zdroj. Rozpad po kategoriích = nejcennější.
Ukázka vyhodnocení: tabulka 5–10 skutečných případů (vstup | očekáváno | model vrátil | shoda), ≥2 SELHÁNÍ + proč.
Reálné příklady odpovědí (ověřené v session): „Kdo je Karel Čapek? → spisovatel", „Kdy se narodil? → 1890",
„Kde se narodil? → Svatoňovice", „Kdo napsal Švejka? → Hašek", „Kdo byl bratr Karla Čapka? → Josef",
„Je pes šelma? → ano", „Je kůň šelma? → ne", „Kdo napsal Babičku? → Němcová".
SELHÁNÍ (reálná): „Kdo objevil mloky? → kapitán" (má být Van Toch), „Kdo byl otec Kundery? → Jan" (má být Ludvík),
„Co je R.U.R.? → doptání/bůh" (doména bible, R/U/R tokenizace), „Kdo postavil archu? → bůh".

## 7. Zdroje pro každý díl (přečti před psaním)

- Fakta a nálezy: `docs/kniha-plan/FAZE-1-pruzkum.md` (registr N1–N17, čísla, otevřené body, slovník).
- Osnova (co má díl obsahovat): `docs/kniha-plan/FAZE-2-osnova.md`.
- Stará dokumentace (VĚCNÝ základ, přepsat vlastními slovy): `docs/*.html` dle tabulky pokrytí ve FÁZI-1 odd. B.
- Kód (platí nad dokumentací při rozporu):
  - Díl I: README/adresáře, `index.html`. · Díl II: `ask.py`, `config.json`, `annotate_corpus.py`, reindex skripty.
  - Díl III: `annotate_corpus.py`, `dataloader.py`, `grammar_vzor.py`, `synth_registry.py`, `build_facts_all.py`, `extract_bio.py`, `extract_relations.py`, `chronos.py`, `answering.py`.
  - Díl IV: `answering.py`, `activation_field.py`, `fact_store.py`, `template_store.py`, `fill_holes.py`, `role_catalog.py`, `gate.json`.
  - Díl V: `eval_large.py`, `eval_answers.py`, `gold_large.json`, `gold_answers.json`, `test_*.py`, `nn_pilot.py`, `train_gate.py`.
  - Díl VI: reindex pipeline, `lang/cs.json`, `config.json`. · Díl VII: `viz.py`, `config.json`, deploy poznámky. · Díl VIII: FÁZE-1 odd. F.
  - Přílohy: `config.json`, `lang/cs.json`, všechny `*.py` (příkazy), FÁZE-1 (registry).

## 8. Kvalita

Píšeš pro nováčka, který o projektu neví nic. Každý pojem vysvětli při prvním výskytu nebo odkaž do
Dílu I slovníku (`dil-1.html#slovnik`). Raději delší a jasnější než stručný a předpokládající znalost.
