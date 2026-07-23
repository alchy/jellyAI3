# hypothesis-one

Český symbolický systém: ze surového textu se staví katalog syntetických vazeb
**FAKT ↔ OTÁZKA ↔ ODPOVĚĎ**. Neuronový nástroj (UDPipe 2) se používá **jen na anotaci
předem, offline**; runtime je deterministický a symbolický — v rozhodovací smyčce žádný
neuron není.

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
| ⑤ queryparser | `answering.py`, `phase1_api.py` | — | živá otázka → odpověď _(staví se)_ |
| sdílené | `logger.py` | — | `logger(severity, message)` → `[i] DDMMYYHHMM : zpráva` |

Konfigurace: `config.json` (bloky po vrstvách — nic natvrdo v kódu).

## Jak spustit

```bash
# ① anotace korpusu (raw → shardy data/corpus/<doc>.pkl)
python annotate_corpus.py

# ② přepočet indexů pro dynamické nahrávání (data/index/)
python dataloader.py         # build_indexes() + sanity výběr souborů

# ④ syntéza: grafikon FAKT↔OTÁZKA + slovník VZOR→role
python synth_registry.py     # → registry.jsonl
python synth_determination.py  # → determination.json

# testy
python test_phase1.py
python test_determiner.py
```

## Data (perzistentní)

| cesta | co |
|---|---|
| `../../data/raw/*.txt` | surové texty (vstup, verzované) |
| `../../data/corpus/<doc>.pkl` | anotační cache — shard per soubor |
| `../../data/index/<doc>.pkl`, `_idf.pkl` | tf·idf indexy pro dynamické nahrávání |
| `config.json`, `lang/cs.json` | konfigurace + jazyková data |
| `curated.json`, `determination.json`, `registry.jsonl` | rolový slovník + grafikon |

## Přidání nového obsahu

1. vlož `.txt` do `../../data/raw/`;
2. `python annotate_corpus.py` (nový shard);
3. `python dataloader.py` (přepočte **všechny** staty — idf závisí na počtu souborů).

## Dokumentace

Otevři `docs/index.html`. Každá fáze má stránku stavěnou na příkladu (vstup → akce →
výstup) s popisem každé třídy a metody (≥ 2 věty, ukázka volání, vstup/výstup). Samostatné
stránky: [Dynamické nahrávání](docs/dynamicke-nahravani.html),
[Anotační cache](docs/anotacni-cache.html), [Termíny a klíče](docs/terminy.html),
[Externí zdroje](docs/externi-zdroje.html).
