# hypothesis-one — stávající stav (baseline pro hypothesis-two)

*Stav ke commitu `3bfdece` na větvi `hypothesis-one`, 2026-07-21.*

## Kde jsme

Postaven a spustitelný **první cut celé smyčky** z úvodní hypotézy:
otázka → syntetická otázka → match na uloženou syntetickou otázku → (plain
vazba) → syntetický fakt s answer-slotem → **aktivace vybere konkrétní výskyt**.
Vše symbolicky, deterministicky, měřitelně.

## Co je postavené a funguje

`experiments/hypothesis-one/run.py` (server na `:8080`):
- **Index** Occurrence + Frame (r=1, modalita v klíči, deterministický `frame_id`).
- **Kanonizace** do 1. pádu: lemma, epentetické -e- příjmení (`Čapků`→`Čapek`),
  přivlastňovací ADJ → jméno (`Karlova`→`Karel`, uniformně), filtr jednopísmenných
  iniciál (`T`/`G`).
- **Salience tf·idf** (df-based) → emise = `mass` (fyzika+aktivace) = `size` = jas.
  Ověřeno: `Čapek` nejjasnější, `v/a/být` = 0.
- **Graf**: uzly = základní tvary; hrany sousednost (bigram, váha=četnost) + mesh λ^d.
- **Vizualizace viewBase 2D/3D** s per-hranovým jasem, fyzikou řízenou vahou hran
  a hmotou uzlů (link/gravitace/odpuzování), `show_sentence` (věta červeně, swap
  R↔B), okna ⚡ info + 🏆 TOP 5, všechny atributy v detail okně.

viewBase (repo `/Users/j/Projects/viewBase`, main `61c56b5`): per-hranový jas
(vertex-colors) + fyzika vahami/hmotami — commitnuto, zpětně kompatibilní.

Demonstrováno skripty (zatím ne v `run.py`):
- **`generate_questions`** — holé otázky ze slotů predikátu (díra + tázací + mod `?`).
- **Syntetická vazba Q→A** — nutná (slovosled mění rámy, nelze swapem modality).
- **Registr + match** — naše otázka nejdřív frame-matchne uloženou synt. otázku
  (různá hladina), teprve tím se dohledá vazba na synt. fakt.
- **Multiplicita** — jedna synt. otázka matchne VÍC textových faktů (predikát
  `patřit` je po korpusu všude) a víc answer-slotů; hladina = shoda rámu × obsahu.
- **Light-beam** — aktivace seedovaná reálnými slovy otázky (základní tvar) +
  kontextem rozhovoru; distribuce ∝ váha hran × síla uzlů; vybere vítěze.
  *Čapek → bratři Čapkové, prezident → Masaryk.*

## Model (pět fází trasy jedné otázky)

```
① otázka (syrová)
② syntetická otázka  (Occurrence + Frame, mod ?)
③ MATCH na uloženou syntetickou otázku (registr, frame-match, hladina)
④ vazba (plain vzor): synt. otázka → synt. fakt (answer-slot = pozice, BEZ aktivace)
⑤ aktivace: seed = reálná slova otázky (základní tvar) + kontext → vybere výskyt
```

Fáze ①–④ = struktura bez aktivace; ⑤ = výběr aktivací.

## Co drhne (klíčový nález pro hypothesis-two)

Light-beam zatím jede **jen po bigramovém řetězu**, syntetické Q→A vazby **nejsou
zhmotněné jako hrany**. Proto slova otázky dosvítí na odpověď jen slabě (~0.01,
přes dlouhý vyhaslý čtecí řetěz) a výběr pak přebíjí kontext. Answer-sloty téhož
faktu by měly být **přímo a silně** spojené s otázkou/predikátem — což je přesně
úloha syntetické vazby.

## Hypothesis-two — co se bude stavět

1. **Zhmotnit syntetické Q→A vazby jako hranový typ** v grafu (predikát/kontext
   faktu → answer-sloty, silná váha) a light-beam pustit PŘES ně.
2. **Materializovat registr** synt. otázek + jejich vazeb (offline nad korpusem).
3. **Kalibrovat distribuci aktivace** (otevřená otázka: vodivost z váhy hran,
   zesílení ze síly uzlů, útlum, počet skoků, práh) — měřicí harness.
4. **Krátkodobá paměť kontextu** + přidávání/expirace aktivačních slov.
5. **Light-beam do živého grafu** — rozsvícená cesta k vítězi ve viewBase.
6. Dořešit gramatiku generátoru (shoda, nominativizace, složení roztrhaných jmen).

## Hypothesis-two — validace (stav, samostatná session)

Vše commitnuto na větvi `hypothesis-two` (simulace v `experiments/hypothesis-one/`):

| část | simulace | výsledek |
|---|---|---|
| dokumentová brána (#60 / swap) | `sim_gate.py` | **8/8** napříč korpusem; cross-dok swap-in kandidáti smysluplní |
| výběr odpovědi + polish syntet. hran | `sim_answer.py` | **33×** zesílení (zhmotnění hran); **2/2** dle kontextu |
| auto-materializace registru | `gen_registry.py` | **9377** vazeb / 2223 predikátů za ~0.7 s (bez LLM) |
| verifikace lematizace + atributů | `sim_verify.py` | chytila korupci jmen (fold), lematizace jinak čistá |
| MorphoDiTa cross-check | `sim_morpho.py` | **negativní** — služby lematizaci nepomohou, UDPipe cache zůstává |
| **full pipeline** | `sim_pipeline.py` | **2/2** — brána→registr→light-beam→kontext vybere odpověď |

Nálezy: (a) bezpodmínečný PROPN fold korumpoval 398 jmen (Egypt→Egypet) → zpět
na podmíněný (2 dvojice). (b) MorphoDiTa `/analyze` produkuje nesmysl (tvar místo
lemmatu, halucinace) → UDPipe cache je lepší. (c) celá smyčka drží v simulaci.

Polish k dořešení: skupina (`pátečník`) by neměla být answer-kandidát; těsné
aktivační marže → kalibrace distribuce (otevřená otázka). Ollama+qwen (lokální,
zdarma) jen na plošnou generalizaci parafrází — glue zbývá.

## Zdroje pravdy

- Spec: `docs/superpowers/specs/2026-07-21-hypothesis-one-synteticky-otazkovy-graf-design.md`
- Experiment: `experiments/hypothesis-one/run.py`
- Korpus: `data/annotations.pkl` (nacachované UDPipe anotace, 13 dokumentů).
