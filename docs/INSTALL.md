# INSTALL — instalace a spuštění jellyAI3

Návod na rozběhání projektu z čistého klonu. Pro produkční nasazení (systemd +
nginx + SSL + basic auth) viz `DEPLOY-ithosaudio.md` (příklad reálného serveru).

## Požadavky

- **Python 3.11 nebo 3.12** (`requires-python >=3.11`).
- Linux/macOS; k dispozici `pip`.
- Přístup na internet (stažení ÚFAL modelů z LINDATu a Python balíčků).
- Pro **webovou vizualizaci** i vývoj: nic navíc — frontend je předpřipravený.
  (Pro vlastní build frontendu viewBase je potřeba Node.js ≥ 20, viz jeho repo.)

## 1. Prostředí a závislosti

```bash
./jelly setup      # vytvoří .venv a nainstaluje requirements.txt (numpy, pytest)
```

Pro **V3 (faktový graf, web, dialog Iris)** doinstaluj ÚFAL nástroje a viewBase:

```bash
.venv/bin/pip install 'ufal.morphodita>=1.11' 'ufal.nametag>=1.2' 'ufal.udpipe>=1.3'
.venv/bin/pip install viewbase          # webová vizualizace (FastAPI/uvicorn)
.venv/bin/pip install websockets        # jen pokud chceš skriptovat WS klienta
```

> ÚFAL balíčky se běžně instalují z hotových wheelů (není nutný kompilátor).
> Kdyby se stavěly ze zdroje, doinstaluj `gcc-c++` a `swig`.

## 2. ÚFAL modely

```bash
./jelly qa-models   # NameTag (entity), MorphoDiTa (morfologie), UDPipe (syntax)
```

Stáhne ~96 MB do `data/models/` z LINDATu (CC BY-NC-SA — nekomerční).

## 3. Data → faktový graf

```bash
./jelly prepare     # bootstrap: seed R.U.R. + vyčistí + zaindexuje
./jelly annotate    # anotace vět (spustí ÚFAL služby na localhostu)
./jelly graph       # postaví reifikovaný faktový graf → data/graph.pkl
```

Vlastní korpus: vhoď `.txt` (UTF-8) do `data/raw/` a spusť `./jelly index` →
`./jelly annotate` → `./jelly graph`.

> **Paměť:** retriever staví HUSTOU matici term×dokument — velký korpus (desítky
> tisíc pasáží) může chtít mnoho GB RAM. Web/graf běh ale `index.pkl` nepotřebuje
> (jede z `graph.pkl`); index je jen pro `./jelly ask` (extraktivní/šablonové QA).

## 4. Spuštění

```bash
./jelly web                 # webová vizualizace grafu + dialog Iris (http://localhost:8080)
./jelly ask "Kdo napsal R.U.R.?"   # jednorázový dotaz (CLI)
./jelly ask                 # interaktivní prompt
./jelly graph-ask "..."     # dotaz přes faktový graf
```

`./jelly web` spustí viewBase na `127.0.0.1:8080` a líně si nastartuje Iris REST
(`:8084`) + ÚFAL služby (nametag/udpipe/morpho). Porty jsou v `config.py`
(`ServicesConfig`) — přemapuj při kolizi s jinými službami.

## 5. Chytáky

- **Porty ÚFAL služeb** (`config.py`): výchozí 8081/8082/8083 mohou kolidovat s
  jinými aplikacemi na stroji → přemapuj (např. 8091–8093).
- **Titulky uzlů ve vizualizaci** vyžadují font, který si troika-three-text tahá
  z `https://cdn.jsdelivr.net` (přes `fetch()`). Za striktní **CSP** (`connect-src`)
  je to blokované → titulky zmizí. Povolit `cdn.jsdelivr.net`, nebo font zabalit lokálně.
- **Python 3.11 vs 3.12:** wrapper `./jelly` volá `python3`; když máš jen 3.12, je to OK.
- **Testy:** `.venv/bin/python -m pytest tests -q` (jádro). viewBase testy chtějí
  `httpx` — spouštěj jen `tests/`.

## 6. Ověření

```bash
.venv/bin/python -m pytest tests -q          # jádro projektu
.venv/bin/python benchmark/run_etalon.py     # etalon (guardrail)
curl -s -X POST localhost:8084/query -H 'Content-Type: application/json' \
     -d '{"question":"Kdo napsal R.U.R.?"}'  # REST dotaz (Iris běží)
```
