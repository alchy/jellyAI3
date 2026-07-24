# hypothesis-one

Český symbolický systém: ze surového textu se staví katalog syntetických vazeb
**FAKT ↔ OTÁZKA ↔ ODPOVĚĎ**. Neuronový nástroj (UDPipe 2) se používá **jen na anotaci
předem, offline**; runtime je deterministický a symbolický — v rozhodovací smyčce žádný
neuron není.

Adresář je **self-contained**: kopie `experiments/hypothesis-one/` běží i buildí samostatně
(všechna data uvnitř, viz níže).

## Architektura ve vrstvách

```
raw text ─▶ ① preprocessing ─▶ anotovaný korpus (shardy) ─▶ ② dataloader ─▶ graf + tf·idf index
                                                                    │
              ③ grammar (VZOR / rolový katalog) ───────────────────┤
                                                                    ▼
                                        ④ synthesis: grafikon FAKT↔OTÁZKA↔ODPOVĚĎ
                                                                    │
  živá otázka ─▶ ⑤ queryparser ─▶ brána dokumentu → fakt → odpověď ◀┘
```

## Moduly (jméno modulu nese vrstvu, jméno třídy = jméno souboru)

| vrstva | modul | třída | dělá |
|---|---|---|---|
| ① preprocessing | `annotate_corpus.py` | `AnnotateCorpus` | raw text → UDPipe 2 → anotovaný korpus (shardy per soubor) |
| ② dataloader | `dataloader.py` | `Dataloader` | tf·idf indexy + nahrávání souborů do grafu podle matche |
| ③ grammar | `grammar_vzor.py`, `role_catalog.py` | `GrammarVzor`, `RoleCatalog` | VZOR (frame_sig) + rozklad věty na role (deprel → náš klíč) |
| ④ synthesis | `synth_registry.py`, `synth_determination.py` | `SynthRegistry`, `SynthDetermination` | grafikon FAKT↔OTÁZKA + slovník VZOR→role |
| ⑤ queryparser | `answering.py` | `Answering` | živá otázka → odpověď / klarifikace / upřímný terminál |
| sdílené | `logger.py` | — | `logger(severity, message)` → `[i] DDMMYYHHMM : zpráva` |

Konfigurace: `config/config.json` (bloky po vrstvách — nic natvrdo v kódu).

## Struktura adresáře

```
*.py            kód (jen .py, žádný JSON)
config/         config.json, lang/cs.json, curated.json
data/           self-contained data (reprodukovatelné → gitignored):
  raw/ corpus/ facts/ index/ templates/  gazetteer.json annotations.pkl
  registry.jsonl determination.json query_templates.jsonl …
  gold/         kurátorovaný ground truth (etalony) — TRACKOVANÝ
  results/      výstupy měření (baseline_*)
docs/           knižní dokumentace (docs/index.html) + last-test.html (scoreboard)
__old__/        nepoužité / legacy (sim_*, phase1-klastr, staré HTML docs)
```

## Jak spustit

```bash
# ① anotace korpusu (raw → shardy data/corpus/<doc>.pkl)
python annotate_corpus.py

# ② přepočet indexů pro dynamické nahrávání (data/index/)
python dataloader.py           # build_indexes() + sanity výběr souborů

# ④ syntéza: grafikon FAKT↔OTÁZKA + slovník VZOR→role
python synth_registry.py       # → data/registry.jsonl
python synth_determination.py  # → data/determination.json

# dotaz
python ask.py "Kdo napsal R.U.R.?"

# etalony (vyžadují běžící UDPipe 2)
python eval_answers.py         # malý faktický etalon (25)
python eval_large.py           # velký etalon (145)
python eval_domain.py          # doménový etalon (71) → docs/last-test.html
python eval_domain.py --sweep  # + porovnání módů mountu (scoreboard routingu)

# offline testy (bez UDPipe)
python test_entity_routing.py
```

## Data (perzistentní)

| cesta | co |
|---|---|
| `data/raw/*.txt` | surové texty (vstup) |
| `data/corpus/<doc>.pkl` | anotační cache — shard per soubor |
| `data/index/<doc>.pkl`, `_idf.pkl` | tf·idf indexy pro dynamické nahrávání |
| `data/facts/<doc>.jsonl`, `data/templates/` | fakty a šablony (mount per otázka) |
| `config/config.json`, `config/lang/cs.json` | konfigurace + jazyková data |
| `config/curated.json`, `data/determination.json`, `data/registry.jsonl` | rolový slovník + grafikon |
| `data/gold/*.json` | kurátorovaný ground truth (etalony) — jediná trackovaná data |

Data `data/` jsou gitignorovaná (reprodukovatelná přes `annotate_corpus` / `build_*` / `synth_*`);
self-contained je přes **kopii adresáře**, ne přes git.

## Přidání nového obsahu

1. vlož `.txt` do `data/raw/`;
2. `python annotate_corpus.py` (nový shard);
3. `python dataloader.py` (přepočte **všechny** staty — idf závisí na počtu souborů).

## Dokumentace

Otevři **`docs/index.html`** — knižní dokumentace (Díl I–VIII + přílohy). Aktuální výsledky
testů: **`docs/last-test.html`** (scoreboard po doménách, přepisuje `eval_domain.py`).
