# Fáze 2 — Osnova knižní dokumentace (NÁVRH KE SCHVÁLENÍ)

Zadání předepisuje: po fázi 2 se zastavit a nechat osnovu schválit. Dvě věci, které si nesmím
domyslet, navrhuji zde: **cílový čtenář** a **umístění výstupu**.

---

## 1. Cílový čtenář (návrh)

**Kdo:** vývojář, který ovládá základy Pythonu (funkce, moduly, dict/list, spuštění skriptu),
ale NEZNÁ tento projekt, jeho doménu ani zpracování přirozeného jazyka.

**Co konkrétně NEumí / nezná (vysvětlit při prvním výskytu):**
- co je závislostní rozbor věty, UDPipe 2, CoNLL-U, lemma / upos / deprel / NameType;
- co je „VZOR", „aktivace / světlo / glow", „díra / koreference", „fakt / role", „mount";
- koncept **dialogového automatu** (proč NE „SELECT FROM databáze");
- proč **deterministický symbolický** přístup místo velkého jazykového modelu;
- proč se jazykový model netestuje na přesnou rovnost.

**Co UMÍ:** spustit `python skript.py`, číst tabulku, orientovat se v repozitáři.

---

## 2. Umístění výstupu (návrh)

- **Nová kniha:** `experiments/hypothesis-one/docs/kniha/` — `index.html` + soubor na kapitolu +
  `assets/style.css` + `assets/script.js` + `assets/mermaid.min.js` (lokálně, ne CDN — funkčnost
  z `file://` bez sítě). Statické HTML5, bez build kroku.
- **Stávající `docs/*.html`** zůstává, dokud kniha nepokryje jeho obsah (tabulka pokrytí, FÁZE-1
  odd. B); pak se stará sada archivuje/smaže po odsouhlasení.
- **Plánovací artefakty** (tento adresář `docs/kniha-plan/`): registr nálezů, osnova, tabulky
  pokrytí — vedeny v Markdownu, verzované, přežijí víc sezení.

---

## 3. Ukázkový scénář (jeden průběžný napříč knihou)

**Vlákno „Karel Čapek".** Táž entita provede čtenáře všemi fázemi:
- surový text: „Karel Čapek (9. ledna 1890 Malé Svatoňovice – …) byl český spisovatel. Byl
  mladším bratrem malíře … Josefa Čapka."
- anotace → indexy → fakty (narodit/who/when/where, kopula/state=spisovatel, bratr=hrana) →
  dotazy: „Kdo je Karel Čapek? → spisovatel", „Kdy se narodil? → 1890" (řetězení kontextu),
  „Kdo byl bratr Karla Čapka? → Josef" (vztah=hrana).
Pokrývá copula, temporal, spatial, relation, carry_context — a máme k tomu naměřená čísla.

---

## 4. Osnova — díly a kapitoly

### Díl I — Než začneme (L0)
- **I.1 Co hypothesis-one řeší a pro koho.** Deterministický český Q&A ze surového textu; odlehčení
  systému KARET parentu (jellyai/). Proč deterministický a symbolický (zdůvodňovací blok).
- **I.2 Slovníček pojmů.** (seed FÁZE-1 odd. D; každý pojem: definice | kde narazíš | související)
- **I.3 Mapa repozitáře.** Moduly, data/, docs/, testy, etalony. Diagram `flowchart TD` adresářů.

### Díl II — První kroky (L1)
- **II.1 Předpoklady a instalace.** `.venv`, requirements, UDPipe 2 na portu 8092 (config), viewBase (volitelně).
- **II.2 První úspěšný běh.** `python ask.py` → „Kdo je Karel Čapek?" → spisovatel. Co uvidíš.
- **II.3 Přestavba dat z nuly.** Proč jsou data gitignored; reindex pipeline (annotate → build_*).

### Díl III — Cesta dat projektem (L2) — kapitola na fázi (šablona zadání 3.1)
- **III.1 Vstup / raw korpus.**
- **III.2 Anotace UDPipe 2 (+ merge zkratek).** Nejpodrobnější (zadání: pozornost fázím 2-3). CoNLL-U, tokenizace, R.U.R. merge.
- **III.3 Indexace (tf·idf, subjects, doclinks, aliases).**
- **III.4 Gramatika a VZOR (frame_sig, role_catalog).**
- **III.5 Syntéza mapy faktů (registry → fakty → šablony; bio/relations/chronos/gazetteer).**
- **III.6 Inference: živý dotaz (mount → aktivace → kandidáti → assurance → polar/relation/gate).**
- **III.7 Odpověď a režim (answer / clarify / unsure).**
- **III.8 Rozhraní mezi fázemi.** `flowchart LR` + tabulka přechodů + vazba na původní text (offsety, doc/sent).

### Díl IV — Jak to uvnitř funguje (L3)
- **IV.1 Dialogový stavový automat (jádro).** Světlo→VZOR→assurance. Zdůvodňovací blok „proč ne SELECT FROM".
- **IV.2 Aktivace / světlo nad grafem.** Váhy uzlů (idf), vodivost hran, spread, decay, glow-orders-ties.
- **IV.3 VZOR matcher.** Šablony FAKT×QUERY×ODPOVĚĎ × aktivace.
- **IV.4 Fakt-store a role.** Datový model `Fact` (`erDiagram`/`classDiagram`), predikát+role.
- **IV.5 Assurance a naučená brána.** Jasno/nejasno/nehádej; NN nad rysy+aktivací (dnes vypnutá).
- **IV.6 Koreference, aliasy, řetězení kontextu.** Díry, identita, carry_context.
- Zdůvodňovací bloky (`<aside class="rationale">`): deterministický runtime, glow-orders-ties, VZOR místo karet, alias merge, gate off.

### Díl V — Testování jazykového modelu (L4) — zadání 4.8
- **V.1 Proč ne rovnost** (nedeterminismus, rozdělení, degradace).
- **V.2 Patra testů** (dnes: jednotkové tenké + metrické/regresní přes etalony; chybějící patra = otevřený bod).
- **V.3 Matice tvarů dat** (tvarosloví, pravopis, znaky, členění, slovní zásoba, rozsah) se stavem pokrytí.
- **V.4 Zlaté sady** (gold_large 145, gold_answers 25, gold_coref) + jak přidat případ.
- **V.5 Metriky a prahy** (etalon %, confident-wrong; reprodukce).
- **V.6 Naměřené výsledky** (křivka, rozpad po kategoriích, ukázky ≥2 selhání). Seed FÁZE-1 odd. E.

### Díl VI — Rozšiřování (L4)
- **VI.1 Přidat dokument / doménu** (raw → reindex). **VI.2 Přidat vztahové slovo / tázací slovo** (cs.json).
- **VI.3 Přidat testovací případ pro nový jazykový jev.** **VI.4 Styl kódu** (třída=soubor, jazyk=data, měř-first).

### Díl VII — Provoz a řešení potíží (L5)
- **VII.1 Závislost na UDPipe** (port 8092, výpadek). **VII.2 Živá vizualizace viz.py** (viewBase, self-restart).
- **VII.3 Pasti** (mrtvá data při deployi, gate overfit, R.U.R.). **VII.4 Nasazení** (reindex na serveru).

### Díl VIII — Otevřené body a možné cesty — zadání 4.7
- Node-based · gate overfit · relation zbytek · R.U.R. doména · temporal · VZOR-síť r · Mnemos · bible mount.
- Každý: kategorie · stav · proč nevyřešeno · ≥2 cesty · signál · kde se dotkneš. `<aside class="open-question">`.

### Přílohy
- Rejstřík pojmů · přehled konfigurace (config.json + cs.json) · seznam příkazů (CLI) · registr
  návrhových rozhodnutí · registr otevřených bodů · tabulka ověření · matice tvarů dat ·
  **tabulka pokrytí staré dokumentace** (FÁZE-1 odd. B).

---

## 5. Diagramy (nejméně 1/díl; typy dle zadání 4.1)
- Díl I: adresář `flowchart TD`. Díl III: `flowchart LR` celé cesty + `sequenceDiagram` u inference.
- Díl IV: architektura `flowchart LR` se subgraph; `stateDiagram-v2` dialogového automatu; `classDiagram` Fact/QueryTemplate.
- Díl V: patra testů `flowchart TD`. Díl VII: ladění `flowchart TD` s větvením.

---

## 6. Co potřebuji odsouhlasit
1. **Cílový čtenář** (odd. 1) — sedí?
2. **Umístění výstupu** (odd. 2) — `docs/kniha/`, stará sada zatím zůstává?
3. **Osnova** (odd. 4) — díly/kapitoly, ukázkové vlákno Karel Čapek?

Po schválení: fáze 3 (kostra + 1 vzorová kapitola s ověřeným Mermaid z `file://`) → fáze 4 (psaní) → fáze 5 (ověření).

---

## 7. Stav realizace

- **Fáze 1 (průzkum):** hotovo → FAZE-1-pruzkum.md.
- **Fáze 2 (osnova):** SCHVÁLENO — čtenář = Python-začátečník bez NLP; umístění = nahradit docs/
  (vývoj v docs/kniha/, přesun na konci); osnova Díl I–VIII + vlákno Karel Čapek.
- **Fáze 3 (kostra):** hotovo + **OVĚŘENO** — Mermaid stateDiagram v dil-4-1 se vykresluje z
  `file://` bez sítě, přepínač režimu funguje (potvrzeno uživatelem v prohlížeči). Offline setup
  (lokální mermaid.min.js v10) tím doložen jako funkční pro všechny další kapitoly.
- **Fáze 4 (psaní):** HOTOVO (commit 531b750) — všech 8 dílů + přílohy (dil-1..8.html, prilohy.html)
  napsáno 9 paralelními agenty dle AUTOR-GUIDE, review hlavním modelem. Vzorová IV.1 vložena do dil-4,
  samostatný soubor smazán. Ověřeno strojově: tagy vyvážené, žádné mrtvé odkazy, 15 Mermaid diagramů,
  jen skutečná čísla, dopředné soon→odkazy. Sidebar sjednocen (díl-level).
- **Fáze 4b (programové přílohy — VÝVOJÁŘSKÁ VRSTVA):** HOTOVO — ke kódově významným dílům vznikly
  přílohy `dil-3/4/6/7-program.html` (manuál PROGRAM-GUIDE.md, 4 agenti, review hl. modelem). Princip
  „princip → kontrakt funkce (vstup→výstup) → reálná ukázka z korpusu"; reálné signatury + výstupy
  spuštěné proti kódu. Pokryto (požadavky uživatele): slovo→lemma→druh→token, VZOR formát/uložení/
  stavba/načtení, UDPipe cache (LINDAT cloud vs port 8092 — ověřeno), indexy+brána ②, cs.json co/proč/
  kde-čteno, aktivace souborů/uzlů/hran (spread/decay/reinforce), fakt-store+role, koreference/díry,
  subsystémy Chronos(žije)/Topos(částečně)/Mnemos(návrh), viewBase integrace, **logování+diagnostika
  (runtime neloguje → doporučení zapnutelné stopy)**. Sidebar sjednocen (14 stránek, pod-odkazy
  `sub-prog` + `prog-callout` z narativu). Gap-check proti staré docs/ proveden per příloha.
- **Konvence „Correct way":** ke KAŽDÉ nedokonalosti (open-question / „Co nefunguje a proč" = skutečný
  dluh / Návrh) přidán blok `<aside class="correct-way">` s CÍLOVÝM stavem (CO, ne JAK) — 38 bloků
  napříč knihou (5 agentů, konvence CONVENTION-correct-way.md). By-design a uživatelské chyby
  přeskočeny (dnešek = ideál). Ověřeno: tag-balance OK, 0 mrtvých odkazů, 38 cw=lbl.
- **Fáze 5 (ověření + přesun):** ZBÝVÁ — (a) vizuální kontrola všech Mermaid diagramů v prohlížeči
  (struktura ověřena strojově, konstrukce shodná s ověřenou IV.1); (b) DESTRUKTIVNÍ přesun
  docs/kniha/ → docs/ (nahradí starou sadu 24 souborů) — až po vizuální kontrole a potvrzení uživatele.
