# jellyAI3 — český QA nad texty s faktovým grafem a dialogem Iris

Knihovna skládatelných bloků pro odpovídání na otázky o obsahu českých textů.
Fakta se berou **přímo z textu** (reifikovaný faktový graf + retrieval), takže
odpovědi jsou dohledatelné a nic se nevymýšlí. Vše běží **lokálně, bez
externích služeb** a s minimem závislostí.

Aktuální jádro: **faktový graf** (fakty jako uzly s role-hranami) + **Iris** —
stavový automat dialogového zaostření řízený JSON kartami. Když si systém není
jistý, ptá se; když neví, řekne to („dialog > figly").

## Rychlý start

Vyžaduje **Python 3.11 nebo 3.12**. Wrapper `./jelly` volá vždy správný Python
z projektového `.venv` — nic se ručně neaktivuje.

```bash
./jelly setup        # jednorázově: vytvoří .venv a nainstaluje závislosti
./jelly qa-models    # jednorázově: stáhne ÚFAL modely (NameTag+MorphoDiTa+UDPipe)
./jelly web          # spustí web na http://localhost:8080
```

Web otevře graf a **tři okna**: 💬 dialogovou konzoli, ⚡ aktivační okno uzlů
(seřazený jas) a 📄 aktivní dokumenty. Ptáš se v konzoli; graf se živě
rozsvěcuje. Prohlížeč mluví s automatem Iris výhradně přes REST (`:8084`).

> **Po každé změně kódu nebo grafu je nutné služby restartovat** — běžící
> instance drží starý kód i graf:
>
> ```bash
> pkill -f iris_service.py; lsof -ti :8080 | xargs kill; pkill -f "cli.py web"
> ./jelly web        # znovu nastartuje web i REST službu
> ```

## Přidání vlastních textů (celá pipeline)

```bash
# 1) vhoď libovolné .txt do data/raw/  (nebo ./jelly wiki pro kurátorované články)
cp ~/moje_kniha.txt data/raw/

# 2) vyčisti a zaindexuj (data/raw → data/processed → data/index.pkl)
./jelly index

# 3) anotuj věty (entity + role; vstup grafu) → data/annotations.pkl
./jelly annotate

# 4) postav faktový graf (extrakce + hygiena + kanonizace + instance)
./jelly graph

# 5) restartuj web (viz výše) a ptej se
```

Build grafu vypisuje, co hygiena vyřadila (slepence jmen, mis-tagy, sémantické
guardy) a co kanonizace sloučila (nominativizace id, jmenné střepy). Smažeš-li
text z `data/raw/` a zopakuješ kroky 2–4, zmizí i z korpusu.

> Generuješ-li vstupní texty (lokálním) modelem, dej mu jako system prompt
> `docs/JAK-PSAT-FAKTA.md` — pravidla formulace, aby extrakce vytěžila maximum.

**Paměť uživatele (Mnemos):** konstatování v dialogu („Dnes jsem měl
knedlíky.", „Venku prší.") se ukládají do deníku `data/memory.jsonl`
a přežívají restart — smazáním souboru paměť vynuluješ.

## Mapa `data/` — statická × uživatelská znalost

Tři druhy obsahu s různou cenou: **odvozené** soubory se kdykoli
přegenerují pipeline (`./jelly index → annotate → graph`), **statické
zdroje** jsou vstupem buildu, a **uživatelské sklady** jsou
NENAHRADITELNÉ — vznikají jen dialogem a zaslouží zálohu.

| cesta | druh | vzniká | verzováno | smazání znamená |
|---|---|---|---|---|
| `raw/` | statický zdroj (korpus) | uživatel / `./jelly wiki` | ne | ztrátu vlastních textů |
| `processed/`, `index.pkl` | odvozené | `./jelly index` | ne | nic — přegeneruje se |
| `annotations.pkl` | odvozené (anotace ÚFAL) | `./jelly annotate` | ne | nic — přegeneruje se |
| `graph.pkl` | odvozené (faktový graf) | `./jelly graph` | ne | nic — přegeneruje se |
| `models/` | statické (ÚFAL modely) | stažení při setupu | ne | nutnost stáhnout znovu |
| `qa/` | odvozené (extraktivní QA) | `./jelly ask` | ne | nic |
| `memory.jsonl` | **uživatelská paměť** (deník Mnemos, append-only, po restartu se přehrává) | dialog | ano | ztrátu paměti uživatele |
| `reminders.jsonl` | **uživatelská** (připomínky Chronos) | dialog | ne | ztrátu plánu připomínek |
| `sub_topos_gazetteer.jsonl` | smíšená (kurátorský seed + učení za pochodu) | seed + dialog | ano | ztrátu naučených míst |
| `telemetry.jsonl` | provozní stopa (triage #38) | provoz | ne | nic — jen diagnostika |
| `web_inbox.txt` | dev most do GUI | vývojář | ne | nic — efemérní |

Pozn.: subsystémové sklady nesou prefix `sub_<subsystém>_…` (#28);
budoucí provenience faktů (#39) rozliší i zdroje uvnitř grafu.

## Příkazy `./jelly`

| Příkaz | Co dělá |
|---|---|
| `./jelly setup` | jednorázově vytvoří `.venv` a nainstaluje závislosti |
| `./jelly qa-models` | stáhne ÚFAL modely (NameTag + MorphoDiTa + UDPipe) |
| `./jelly wiki` | stáhne kurátorované české wiki články do `data/raw` |
| `./jelly index` | po změně textů v `data/raw/` vyčistí a přegeneruje index |
| `./jelly annotate` | offline anotace vět (entity + role) — vstup faktového grafu |
| `./jelly graph [--view]` | postaví faktový graf z anotací (`--view` = export do viewBase) |
| `./jelly web` | prohlížeč: graf + tři okna Iris (dialog / uzly / dokumenty) |
| `./jelly triage` | shluky neúspěšných tahů ze stopy provozu (`data/telemetry.jsonl`, #38) |
| `./jelly qgraph` | vypíše kompilát otázkového grafu (uzly, hrany, schéma predikátů) |
| `./jelly graph-ask [otázka]` | dotaz nad grafem z příkazové řádky (bez webu) |
| `./jelly ask "otázka?"` | jednorázový dotaz retrieval cestou (V1) |
| `./jelly explain <blok>` | vysvětlí blok (bez argumentu vypíše seznam bloků) |
| `./jelly prepare` | bootstrap dat: seed R.U.R. + vyčistí + zaindexuje |

## Iris — dialogový automat (REST `:8084`)

Chování neřídí kód, ale **JSON pattern-karty**
(`jellyai/iris/patterns/cs/` — 1 karta = 1 vzor: trigger → dialog → akce →
teach; i prahy rozhodování nesou karty). Pod prahem jistoty (**QueryAssurance**)
se automat ptá místo hádání; bez odpovědi přizná „nenašel" a nabídne nejbližší
kandidáty. Otázky překládá **šablonový parser pseudo-QL** (`jellyai/answerer/
query.py`) — vzorové karty + poziční šablony, dotazová cesta je UDPipe-free
(řez #14). Vstup směruje **otázkový graf** (#57/#51): kompilát karet s pěti
rodinami uzlů (otázky, výroky, příkazy, workeři, clarify) — pořadí bran nesou
data, ne kód. Tápání není terminál: systém dá **částečnou odpověď** z rolí
faktů, **chytrou clarifikaci** (které role děj zná) nebo **nabídku kandidátů**
s volbou (kaskáda jistot, princip „data ověř — pak se ptej").

Subsystémy (`jellyai/iris/subsystems/`): **Chronos** (intervaly „dnes/v 19.
století/minulý týden", tvrdý časový filtr odpovědí, hodinové otázky,
PŘIPOMÍNKY — „připomeň mi za deset minut / zítra / v září…" s chytrými
defaulty, vlastní vlákno hodin, okno ⏰ Reminder, správa plánu „zruš/posuň/
přeplánuj" i výpis „Mám něco v plánu?"), **Mnemos** (paměť uživatele
v témže grafu: konstatování, explicitní „zapamatuj si/nezapomeň…",
poznámky-přísloví doslovně, vzpomínání „Co jsem ti řekl včera?"),
**Topos** (kontejnment míst — gazetteer `data/sub_topos_gazetteer.jsonl`,
místní filtr „Pršelo v Čechách?", učení za pochodu z vnořených míst
výroku), **focus-shift** a **jmenné rodiny** (instanční vrstva).

```bash
.venv/bin/python services/iris_service.py --port 8084 --model data/graph.pkl
curl -s localhost:8084/schema     # na co se lze ptát (predikáty, role, karty)
```

- `POST /query {"question", "temperature"?}` → odpověď/dialog + metadata
  (assurance, použité karty, aktivační okno uzlů i dokumentů),
- `POST /graphql {pattern JSON}` → přímé vykonání pseudo-QL patternu,
- `POST /reset` → nový rozhovor, `GET /schema` → popis dotazovatelného,
- `GET /version` → SHA buildu (verzovací handshake #40 — klient křičí,
  když se lokální kód a běžící služba rozejdou).

## Benchmarky — objektivní řízení změn

Každá změna se měří; normativy neklesají (guardrail). Stav 2026-07-20
(ověřeno spuštěním): **582 testů, etalon 33/33 (15 gap řádků: 11
opraveno / 4 otevřeno), focus 12/12, dialog 45/45 tahů ve 20
scénářích, zápis 34/34, otázkový graf 5 rovin 100 % v obou
variantách; výstupy deterministické napříč hash seedy (#58).**

```bash
.venv/bin/python -m pytest -q               # jednotkové testy
.venv/bin/python benchmark/run_etalon.py    # správnost odpovědí (JÁDRO + gap tracking)
.venv/bin/python benchmark/run_focus.py     # zaostření aktivace: expect uzly v top-K jasu
.venv/bin/python benchmark/run_dialog.py    # dialogové scénáře Iris (fixní hodiny)
.venv/bin/python benchmark/run_mnemos.py    # ZÁPIS: výrok → parse (zápisový etalon)
.venv/bin/python benchmark/run_qgraph.py    # otázkový graf: dispatch/výroky/stav/dekorace
.venv/bin/python benchmark/run_coverage.py  # výtěžnost extrakce — diagnostika
```

Architektonická dokumentace pro nové vývojáře: **`docs/architektura-web/index.html`**
(17 kapitol, offline, Mermaid) + výkladový `docs/ARCHITEKTURA.md`.

Otevřené body a priority: `docs/BACKLOG.md`. Předání práce mezi sezeními:
`docs/HANDOVER.md`.

## Knihovna (`import jellyai`)

Skládáš granulární **bloky (porty)** — každý jde vyměnit či parametrizovat,
včetně pozdějšího zapojení NN. Fasáda `Jelly` je tenké dráty.

```python
import jellyai
jellyai.demo()                              # bez modelů: mini-graf + odpovědi

jelly = jellyai.Jelly(answerer=muj_answerer)  # injektuj vlastní/NN port
ans = jelly.ask("kdo napsal Babičku?")
print(ans.explain())                        # vysvětlitelná odpověď (trasa grafu)
```

Porty: `Tokenizer`, `QuestionAnalyzer`, `FactExtractor`, `Composer`,
`CorpusPort`, `GraphView`, `Retriever`, `Answerer`. Runnable ukázky
v `examples/` (01–06). Jazyk je **datový modul**: pravidla a tabulky češtiny
žijí v `jellyai/lang/cs.json` — nový jazyk = nový JSON, bez zásahu do kódu.

## Jak to funguje

```
texty → annotate (UDPipe+NameTag) → extrakce faktů → hygiena (korpusová
evidence) → kanonizace + nominativizace + instanční vrstva → graf.pkl
                                                                │
otázka → pseudo-QL šablony → match nad grafem → odpověď/dialog (Iris karty)
```

- **Faktový graf**: každá slovesná událost je faktový uzel (predikát + váha =
  opakování) s role-hranami `subj/obj/time/loc/num/pred/attr/theme`. Zanořená
  data (datum → pod-fakty rok/měsíc/den), aktivační koreference (pro-drop),
  konverzační těžiště pro navazující otázky.
- **Hygiena** (`jellyai/graph/hygiene.py`): korpus ví líp než jedna věta —
  hlasování o upos/pádu/životnosti/lemmatu vyřazuje slepence jmen a mis-tagy
  (lokální tag je nejslabší evidence).
- **Instanční vrstva** (`jellyai/graph/instance.py`): jméno není entita —
  jmenné střepy srůstají jen na základě textového tvrzení („X řečený Y").

## Historie vývojových arců (zkráceně)

V1 retrieval + extraktivní QA → V2a syntetická QA data (NameTag) → V3
pravidlové odpovědi (ÚFAL služby) → V4a bohatá analýza otázky → B1 větný
retrieval (zakonzervováno) → **faktový graf** → **pseudo-QL parita** (šablony
místo UDPipe) → **Iris** (karty, Chronos, Mnemos, REST, web). Detaily
a poctivé výsledky každého arcu: `docs/superpowers/specs/` a
`docs/superpowers/*-results.md`; zakonzervované komponenty:
`docs/superpowers/conserved-components.md`.

## Testy

```bash
.venv/bin/python -m pytest -q     # bez aktivace venv (konvence projektu)
```
