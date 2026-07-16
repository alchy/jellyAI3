# V3 — TemplateAnswerer + ÚFAL služby — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans nebo subagent-driven-development. Kroky mají checkbox (`- [ ]`).
> **Realizace:** většinu kódu píše Claude sám (preference uživatele).

**Goal:** Pravidlový answerer, který z retrievalu + ÚFAL služeb (NameTag/UDPipe/MorphoDiTa) složí plynulou, správně skloňovanou českou odpověď. Každý ÚFAL nástroj běží jako vlastní localhost REST proces (řeší SWIG konflikt).

**Tech Stack:** Python 3.11, stdlib `http.server`/`urllib`/`json`, `ufal.nametag`, `ufal.morphodita`, `ufal.udpipe`. Navazuje na V1/V2a (z `main`, nezávisle na V2b).

## Global Constraints

- Python 3.11 venv; commity končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Bez nových třetích stran** — služby stdlib `http.server`, klient `urllib`.
- Služby poslouchají **jen na `127.0.0.1`**.
- Hermetické testy přes `FakeUfalClient` (žádné modely/síť); reálné služby = `skipif`.
- `data/annotations.pkl`, `model/udpipe*` gitignorovat.
- Větev `feature/v3-template-answerer`.

---

### T1 — Konfigurace + gitignore
**Files:** `config.py` (Modify), `.gitignore` (Modify), `tests/test_config.py` (Modify)
- `AnswererConfig.mode: str = "extractive"`.
- Nový `ServicesConfig`: `host="127.0.0.1"`, `nametag_port=8081`, `udpipe_port=8082`,
  `morpho_port=8083`, `nametag_model`, `morphodita_model`, `udpipe_model="model/udpipe-czech.model"`,
  `startup_timeout=30.0`, `annotations_path="data/annotations.pkl"`. Přidat `Config.services`.
- `.gitignore`: `data/annotations.pkl`, `model/udpipe*`.
- Test: defaulty (mode, services porty, annotations_path).

### T2 — HTTP klient + FakeUfalClient (`jellyai/ufal_client.py`)
**Files:** `jellyai/ufal_client.py` (Create), `tests/test_ufal_client.py` (Create)

**Interfaces (Produces):**
- `UfalClient(services_config)` s `entities(text)->list[dict]`, `parse(text)->list[dict]`,
  `analyze(text)->list[dict]`, `generate(lemma, tag_wildcard)->list[str]`. Služby startuje
  líně (subprocess), čeká na `/health`, `atexit` je složí. `_post(port, path, payload)->dict`
  přes `urllib`.
- `FakeUfalClient(entities=None, parse=None, generate=None)` — vrací nakonzervovaná data
  podle textu/lemmatu (jako `FakeTagger`); pro testy.

- [ ] Test: `FakeUfalClient` vrací nakonzervované entity/parse/generate; neznámé → `[]`.
- [ ] `_post` a lazy-spawn logika napsat tak, aby šla otestovat proti fake HTTP serveru
  spuštěnému v testu (stdlib `http.server` na ephemeral portu) — health + jeden echo endpoint.
- [ ] Commit.

**Skeleton klienta (klíč):**
```python
import atexit, json, subprocess, sys, time, urllib.request

class _Handle:
    def __init__(self, script, port, timeout):
        self.proc = subprocess.Popen([sys.executable, script, "--port", str(port)])
        atexit.register(self.close)
        self._wait_health(port, timeout)
    def _wait_health(self, port, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(f"Služba na portu {port} nenaběhla")
    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
```
(`time.sleep` je zde OK — je to čekání na health v samostatné pomocné smyčce, ne foreground blok v hlavní logice.)

### T3 — Tři ÚFAL služby (`services/*.py`)
**Files:** `services/nametag_service.py`, `services/udpipe_service.py`, `services/morpho_service.py` (Create), `tests/test_services.py` (Create, skipif)

- Každá: `argparse --port`, načte svůj model, `ThreadingHTTPServer` na `127.0.0.1:port`,
  `BaseHTTPRequestHandler` s `/health` (GET 200) a POST endpointy vracejícími JSON.
- `nametag_service`: `/entities` → `[{text,type,start,end}]`.
- `udpipe_service`: `/parse` → věty s tokeny `[{form,lemma,upos,head,deprel,start,end}]`.
- `morpho_service`: `/analyze` → tokeny s lemma+tag; `/generate` body `{lemma, tag}` →
  `[forms]` (skloňování).
- Importy ÚFAL **jen uvnitř dané služby** (proces izolovaný).
- Test (skipif, když modely nejsou): spustit službu jako subprocess, ověřit `/health`
  a jeden reálný dotaz. Hermetická sada tím netrpí.

### T4 — Stažení UDPipe modelu (`qagen/download_models.py`)
**Files:** `qagen/download_models.py` (Modify), `requirements-v2.txt` (Modify)
- Přidat `ufal.udpipe` do requirements-v2.
- Přidat český UDPipe model (LINDAT handle, DSpace 7 resolve jako u ostatních) do `MODELS`.
- Test: `_resolve`/`MODELS` logika (bez sítě) — struktura, ne stažení.

### T5 — Offline anotace (`jellyai/annotate.py`, CLI `annotate`)
**Files:** `jellyai/annotate.py` (Create), `cli.py` + `jelly` (Modify), `tests/test_annotate.py` (Create)

**Interfaces:** `annotate_passages(passages, client) -> dict[(doc_id,index)->Annotation]`
(entity + tokeny/role per věta); `save_annotations(ann, path)`, `load_annotations(path)`.
- Volá `client.entities` a `client.parse` na text pasáže; uloží sidecar.
- CLI `cmd_annotate(config, client=None)` → načte pasáže z indexu/korpusu, anotuje, uloží.
- `./jelly annotate`.
- Test: s `FakeUfalClient` anotace vzniknou a uloží/načtou se (roundtrip).

### T6 — Šablony (`jellyai/templates.py`)
**Files:** `jellyai/templates.py` (Create), `tests/test_templates.py` (Create)

**Interfaces:** `TEMPLATES: dict[qtype -> (frame, target_case)]`;
`fill(qtype, inflected_answer) -> str`; `target_case(qtype) -> str|None`.
- Test: `Kdo` + „Božena Němcová" → „Božena Němcová"; `Kde` + „Praze" → „v Praze".

### T7 — Výběr odpovědní entity (`jellyai/answerer/selection.py`)
**Files:** `jellyai/answerer/selection.py` (Create), `tests/test_selection.py` (Create)

**Interfaces:** `select_answer(question_analysis, annotation) -> Candidate|None`.
Kandidát = (lemma, form, cnec_type, deprel). Pravidla:
- Kdo → entita-osoba, přednostně `deprel` podmět (`nsubj`) slovesa z otázky,
- Co → token/entita s `deprel` předmět (`obj`),
- Kde → entita místo (CNEC `g*`), Kdy → čas (`t*`), Kolik → `upos == NUM`.
- Test: věta se dvěma osobami (anotace přes FakeUfalClient) — „Kdo" vybere podmět,
  „Co" vybere předmět.

### T8 — TemplateAnswerer (`jellyai/answerer/template.py`)
**Files:** `jellyai/answerer/template.py` (Create), `tests/test_template_answerer.py` (Create)

**Interfaces:** `TemplateAnswerer(config, client, annotations, extractive_fallback)` s
`answer(question, retrieved) -> Answer`; `explain()`.
- Klasifikace otázky (typ + sloveso) přes `client.parse`; výběr (`selection`); skloňování
  (`client.generate` na lemma → target_case); `templates.fill`. Bez kandidáta → fallback.
- Test (FakeUfalClient + fake annotations): „kdo napsal Babičku?" → „Božena Němcová" se zdrojem;
  bez kandidáta → fallback extraktivní věta.

### T9 — Integrace (pipeline mode, CLI, jelly)
**Files:** `jellyai/pipeline.py`, `cli.py`, `jelly`, `jellyai/explain.py` (Modify), `tests/test_v3_integration.py` (Create)
- `pipeline._make_answerer(config)`: mode `"template"` → `TemplateAnswerer` (klient + načtené
  anotace), jinak extraktivní. Líný import ÚFAL klienta.
- `--template` přepínač (jako `--gen` u V2b) → `config.answerer.mode="template"`.
- `explain` registr o `template`.
- Test: pipeline mode="template" s FakeUfalClient → Answer.

### T10 — Reálné ověření + README
**Files:** `README.md` (Modify)
- `./jelly qa-models` (stáhne i UDPipe) → `./jelly annotate` → `./jelly ask --template "…"`.
- Porovnat s extraktivním na několika otázkách; zapsat honest výsledky.
- README sekce V3.

## Self-Review
Spec pokryt: služby (T3) + klient (T2) + SWIG izolace, anotace (T5), šablony (T6),
výběr role (T7), TemplateAnswerer + fallback (T8), config/mode (T1), UDPipe model (T4),
integrace (T9), ověření (T10). Hermetické testy přes FakeUfalClient; reálné služby skipif. ✔
Placeholder scan: konkrétní rozhraní a klíčové skeletony; přesné LINDAT URL UDPipe modelu
se ověří při T4 (jako u ostatních modelů). ✔
