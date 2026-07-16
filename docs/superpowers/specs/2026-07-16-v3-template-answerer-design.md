# Design V3: pravidlový TemplateAnswerer + ÚFAL služby

**Datum:** 2026-07-16
**Autor:** Jindřich Němec + Claude
**Status:** Schváleno

## 1. Cíl a kontext

V2b (neuronový generátor) dával krátké, nesklonované, často chybné odpovědi —
kvalitu limitovala data i to, že se model snažil „naučit" morfologii. V3 mění
přístup: **fakta z retrievalu, gramatika ze šablon, tvary z MorphoDiTy** (která
češtinu už umí). Výsledek: plynulé, správně skloňované, pravdivé odpovědi — bez
trénování báze. Buduje se nezávisle na V2b (z `main`); `answerer.mode` získá
volbu `"template"` vedle `"extractive"`.

## 2. Klíčová rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Přístup | Pravidlový: výběr entity → šablona → morfologické skloňování |
| Fakta | Z retrievalu (BM25) — model si nic nepamatuje, nehalucinuje |
| Gramatika | Šablony per typ otázky + MorphoDiTa generování tvarů |
| Výběr entity | NameTag (typ) + UDPipe (syntaktická role: podmět/předmět) |
| SWIG konflikt | **Každý ÚFAL nástroj = vlastní proces s localhost REST API** |
| Závislosti | Bez nových — stdlib `http.server` (služby) + `urllib` (klient) |
| Fallback | Když nic nesedí → extraktivní answerer (nikdy nelže/nemlčí) |
| Integrace | `answerer.mode = "extractive" \| "template"`, pluggable |

## 3. Architektura

### 3.1 Vrstva ÚFAL služeb (řeší SWIG konflikt)
`ufal.nametag`, `ufal.morphodita` a `ufal.udpipe` se v jednom procesu perou
(sdílený SWIG typ `vector<string>`). Řešení: **každý běží jako samostatný proces**
s malým HTTP API na `127.0.0.1`.

| Služba | Endpointy | Nástroj |
|---|---|---|
| `services/nametag_service.py` | `POST /entities` | NameTag |
| `services/udpipe_service.py` | `POST /parse` | UDPipe |
| `services/morpho_service.py` | `POST /analyze`, `POST /generate` | MorphoDiTa |

- Každá služba je tenký `ThreadingHTTPServer` (stdlib), načte svůj model, poslouchá
  jen na localhost, `/health` vrací 200 po načtení. JSON dovnitř i ven.
- Bonus: tím se uklidí i původní konflikt v `qagenu` — může volat stejné služby.

### 3.2 Klient a životní cyklus (`jellyai/ufal_client.py`)
- `ServiceHandle` — spustí službu jako subprocess (`.venv/bin/python services/X.py --port N`),
  počká na `/health` (timeout), na `atexit` ji složí.
- `UfalClient` — HTTP klient s metodami `entities(text)`, `parse(text)`,
  `analyze(text)`, `generate(lemma, tag_wildcard)`. Služby startuje líně při první
  potřebě, drží je po dobu procesu (načtení modelu se platí jen jednou).
- Pro testy je klient **injektovatelný** — `FakeUfalClient` vrací nakonzervovaná
  data (hermetické testy bez modelů/sítě).

### 3.3 Offline anotace (`jellyai/annotate.py`, `./jelly annotate`)
Pasáže se jednou obohatí o entity (NameTag) a syntaktický rozbor (UDPipe) a uloží
se do sidecaru `data/annotations.pkl` klíčovaného `(doc_id, passage_index)`.
Query-time se pak jen čte — žádné parsování za běhu.

Anotace pasáže: seznam vět, každá s:
- `entities`: [(text, cnec_type, start, end)],
- `tokens`: [(form, lemma, upos, head, deprel, start, end)].

### 3.4 Šablony (`jellyai/templates.py`)
Mapa typ otázky → šablona odpovědi se sloty a cílovým pádem. Např.:
- `Kdo` → `"{answer:nom}"` (podmět v 1. pádě): „Božena Němcová".
- `Kde` → `"v {answer:loc}"`: „v Praze".
- `Kdy` → `"{answer}"` (datum beze změny).
- `Kolik` → `"{answer}"` (číslo beze změny).
- `Co` → `"{answer:nom}"` (předmět převedený do 1. pádu): „Babička".

### 3.5 TemplateAnswerer (`jellyai/answerer/template.py`)
`answer(question, retrieved)`:
1. Klasifikace otázky (Kdo/Co/Kde/Kdy/Kolik) + hlavní sloveso (parse otázky přes
   UDPipe službu).
2. V top pasáži (dle anotací) vybrat **kandidátní odpověď** podle typu a role:
   - Kdo → entita-osoba, přednostně **podmět** slovesa z otázky,
   - Co → **předmět** slovesa,
   - Kde/Kdy → entita místo/čas, Kolik → číslovka (`NUM`).
3. Lemma kandidáta → `morpho /generate` do pádu ze šablony → tvar.
4. Poskládat větu ze šablony. Když žádný kandidát nesedí → **fallback extraktivní**.
Vrací `Answer(text, sources=["doc#idx"], score)`.

## 4. Konfigurace (`config.py`)
- `AnswererConfig.mode: str = "extractive"` (volba `"template"`).
- Nový `ServicesConfig`: cesty modelů (nametag/morphodita/udpipe), porty, host
  `127.0.0.1`, `startup_timeout`, `annotations_path = "data/annotations.pkl"`.

## 5. Tok dat

```
offline:  pasáže → nametag /entities + udpipe /parse → data/annotations.pkl
online:   otázka → udpipe /parse (typ+sloveso)
          retrieval → top pasáž → anotace → výběr entity (typ+role)
          → šablona → morpho /generate (skloň) → věta   (fallback: extraktivní)
```

## 6. Ošetření chyb a hraniční případy
- Služba nenaběhne (chybí model/UDPipe) → jasná hláška; answerer padne na extraktivní.
- Žádná vhodná entita v pasáži → fallback extraktivní.
- MorphoDiTa neumí lemma skloňovat → použije se původní tvar entity.
- Chybějící `annotations.pkl` → hláška „spusť `annotate`"; fallback extraktivní.
- Porty obsazené → konfigurovatelné; služby jen na localhost.

## 7. Testování (pytest)
- **Klient/služby:** `FakeUfalClient` vrací nakonzervované entity/parse/generate →
  hermetické testy answereru, výběru, šablon, skloňování bez modelů.
- **Templates:** typ + entita + pád → očekávaný string.
- **Selection:** z anotované věty s víc osobami vybere podmět pro „Kdo", předmět pro „Co".
- **TemplateAnswerer:** end-to-end s FakeUfalClient → správně sklonovaná věta + zdroj;
  bez kandidáta → fallback extraktivní.
- **Integrační (skipif):** reálné služby jen když jsou modely stažené (NameTag+UDPipe+
  MorphoDiTa); jinak přeskočit.

## 8. Nové ÚFAL modely
- MorphoDiTa (`ufal.morphodita`) + model MorfFlex/PDT-C — už stažený z V2a.
- UDPipe (`ufal.udpipe`) + český UD model — **nový**; stáhne `download_models`.
- NameTag — už z V2a.

## 9. Mimo rozsah (YAGNI)
- #3 hierarchický retrieval (věta→odstavec→soubor) — samostatné vylepšení retrieveru.
- Generativní neuronové odpovědi (to je V2b, jiná větev).
- Složitá souvětí / víc faktů v jedné odpovědi.
