# Tvary, anafora, tematické role — Implementation Plan (arc 3)

> ✅ **DOKONČENO (2026-07-16).** Výsledky (baseline → po arc 3):
> coverage **41 % → 31 %** vět bez faktu (943 → 1102 vět s faktem); etalon
> **15/15 drží** po celou dobu; graf 3510 → **4068 faktů** (+45 kvalitních
> anafor, +433 theme, +150 koordinace, −šum falešných pro-drop faktů
> demonstrativ); testů 240 → **252**. Navíc Task V: morfologie v popisu uzlů
> vizualizace (rod, kmen, sloučené pádové tvary — `graph.aliases`).

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Navazuje na arc 2 (etalon 15/15). Pokyn: průběžně reportovat metriky.

**Goal:** Snížit podíl vět bez faktu (41 %) univerzálními mechanismy: morfologické
rysy (feats) jako signál, zájmenná anafora přes aktivační pole, tematické role
pro koncepty, distribuce koordinace.

**Architecture:** UDPipe služba vrátí FEATS (CoNLL-U sl. 6) → tokeny nesou
`feats` dict. Anafora = zobecněný pro-drop: osobní zájmeno váže nejteplejší
rodově shodnou osobu (demonstrativum osobu neváže nikdy — nahradí plošný
blok z arc 2). Koncepty v obl/nmod dostanou roli `theme`. Koordinace (`conj`)
distribuuje fakt přes souřadné účastníky.

**Metriky (po KAŽDÉM tasku, reportovat):**
- `benchmark/run_etalon.py` — JÁDRO nikdy pod 15/15, gapy sledovat.
- `benchmark/run_coverage.py` (Task 0) — % vět bez faktu + kbelíky příčin.
- Velikost grafu (uzly/fakty), počet testů, pylint.

**Baseline:** etalon 15/15 (100 %), GAP 2/3; coverage 656/1599 (41 %):
409 bezslovesných / 171 sloveso-bez-účastníků / 65 anafora / 11 spona;
graf 2243 uzlů, 3510 faktů; 240 testů.

## Global Constraints

- Univerzálnost (žádné téma-vzory), determinismus, jazyková specifika jen
  v `jellyai/lang/`. Refactoring bez zpětné kompatibility povolen.
- Práce na větvi `tvary-anafora-role`; commit po každém tasku.

---

### Task 0: Coverage audit jako stálý benchmark

`benchmark/run_coverage.py` — spočítá věty bez faktu a kbelíky příčin
(bezslovesné / sloveso bez účastníků / zájmenný podmět-předmět / spona).
Vedle etalonu (správnost) měří, KDE text mlčí (výtěžnost).

### Task A: FEATS ze služby UDPipe

`services/udpipe_service.py::_parse_conllu` — sloupec 5 (`Case=Gen|Gender=…`)
→ token `"feats": dict`. Re-anotace korpusu. Metriky beze změny (jen data navíc).

### Task B: Zájmenná anafora (zobecněný pro-drop)

`extract.py`: osobní zájmeno (feats `PronType=Prs`) v roli subj/obj se rozváže
na nejteplejší **rodově shodnou** osobu (`Gender` zájmena × rod jména);
demonstrativum (`PronType=Dem`, „to") osobu neváže nikdy — nahradí plošný
copular-blok z arc 2. Vyžaduje protáhnout aktivační pole (kandidáty) do
extrakce — dnes jde dovnitř jen jediný `default_subject`.

### Task C: Tematická role pro koncepty

`extract.py::_ATTR_ROLE` — konceptové obl/nmod účastníky nezahazovat: role
`theme` („uvažovat o souvislosti", „vydobyla pozici patronky"). Cíl: ukrojit
kbelík 171. Riziko šumu → hlídá etalon.

### Task D: Distribuce koordinace

`extract.py`: `conj` řetěz podmětů/předmětů → fakt pro každý prvek
(„Karel a Josef napsali X" → 2 fakty; „psal romány a dramata" → 2 fakty).

### Task E: Uzávěr — metriky, docs, merge

Finální tabulka metrik (baseline → po arc 3), aktualizace paměti, merge do main.

## Rollback / stopky

Etalon pod 15/15 nebo rozbitý test → stop a analýza. Šum z theme role /
anafory se měří etalonem a coverage — nezvyšovat výtěžnost za cenu správnosti.
