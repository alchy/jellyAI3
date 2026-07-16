# jellyAI3 — český QA nad texty (V1)

Výuková knihovna skládatelných bloků pro odpovídání na dotazy o obsahu českých
textů. Fakta se berou **přímo z textu přes retrieval** (ne z „paměti" modelu),
takže odpovědi jsou dohledatelné a model si nevymýšlí. Vše běží **lokálně, bez
externích služeb** (žádná Ollama, žádné API) a s minimem závislostí (numpy).

> Dřívější verze byla slovní LSTM model pro predikci dalšího slova — zůstává
> dohledatelná v git historii. Proč a jak vznikla tahle podoba, viz
> `docs/superpowers/specs/2026-07-15-cesky-gpt-design.md`.

## Rychlý start (přes `./jelly`)

Wrapper `./jelly` volá vždy správný Python z projektového `.venv`, takže nemusíš
nic ručně aktivovat. Vyžaduje **Python 3.11 nebo 3.12** (novější verze mají
problém s torch, který přijde až ve V2).

```bash
./jelly setup                    # jednorázově: vytvoří .venv a nainstaluje závislosti
./jelly prepare                  # připraví data (seed R.U.R. + vyčistí + zaindexuje)
./jelly ask "kdo vynalezl roboty?"
```

Příklad výstupu:

```
$ ./jelly ask "kdo vynalezl roboty?"
Podle textu: Stojí tam například, že Roboty vynalezl starý pán.
(zdroj: rur#85)
```

## Přidání vlastního textu

Přesně tři kroky — vhoď text, přegeneruj vektory, ptej se:

```bash
# 1) vlož libovolný .txt do data/raw/
cp ~/moje_kniha.txt data/raw/

# 2) přegeneruj index (vyčistí data/raw → data/processed a postaví vektory)
./jelly index

# 3) ptej se — buď jednorázově, nebo v interaktivním promptu
./jelly ask "na co se chci zeptat?"
./jelly ask                      # bez otázky spustí interaktivní prompt
```

Index se ukládá na disk (`data/index.pkl`), takže se při dotazu nestaví znovu a
prompt naskočí okamžitě. Smažeš-li text z `data/raw/` a dáš `./jelly index`,
zmizí i z korpusu (processed zrcadlí raw).

Interaktivní prompt vypadá takhle (ukončíš prázdným řádkem, `konec` nebo Ctrl-D):

```
$ ./jelly ask
Ptej se česky. Prázdný řádek nebo 'konec' ukončí.

❓ kdo je Helena?
💬 Podle textu: Helena: Robotka Helena.
   (zdroj: rur#1404)

❓ konec
Měj se! 👋
```

## Příkazy `./jelly`

| Příkaz | Co dělá |
|---|---|
| `./jelly setup` | jednorázově vytvoří `.venv` a nainstaluje závislosti |
| `./jelly prepare` | bootstrap dat: seed R.U.R. + vyčistí + zaindexuje |
| `./jelly index` | po změně textů v `data/raw/` přegeneruje a uloží index |
| `./jelly ask "otázka?"` | jednorázový dotaz |
| `./jelly ask` | interaktivní prompt na dotazování |
| `./jelly explain <blok>` | vysvětlí blok (bez argumentu vypíše seznam bloků) |

## Bez wrapperu (přímo přes Python)

Kdo chce vidět, co se děje pod pokličkou, může volat CLI napřímo (po aktivaci venv):

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python cli.py prepare-data
python cli.py reindex
python cli.py ask "co znamená R.U.R.?"
python cli.py repl
python cli.py explain retriever
```

## Jak to funguje — bloky

Pipeline vede dotaz řadou malých, nezávislých bloků:

```
Loader → Chunker → Retriever → ExtractiveAnswerer → Pipeline
```

| Blok | Co dělá |
|---|---|
| **Loader** | načte vyčištěné texty do objektů Document |
| **Chunker** | rozseká dokumenty na překrývající se pasáže (po větách) |
| **Retriever** | najde k dotazu nejrelevantnější pasáže — TF-IDF / BM25 psané od nuly v numpy |
| **ExtractiveAnswerer** | vybere z pasáží nejrelevantnější větu + uvede zdroj |
| **Pipeline** | pospojuje bloky do `ask(otázka) → odpověď` |

Veškerá konfigurace (velikost pasáží, metoda vyhledávání, top-k, …) je v
`config.py`. Každý blok umí `explain()` popsat, co dělá — zkus `./jelly explain retriever`.

## Testy

```bash
python -m pytest -v
```

## V2a — syntetická QA data (příprava pro generátor)

Generuje dvojice otázka→odpověď z korpusu pomocí českého NER modelu ÚFAL
**NameTag** (entity → typy otázek Kdo/Co/Kde/Kdy/Kolik). Slouží jako tréninková
data pro budoucí generátor (V2b). Vyžaduje extra závislosti a model (CC BY-NC-SA,
nekomerční = pro osobní/výukové OK):

```bash
pip install -r requirements-v2.txt
./jelly qa-models      # stáhne český NameTag model do data/models
./jelly qa             # vygeneruje data/qa/qapairs.jsonl z data/processed
```

Kvalita otázek hodně závisí na zdroji: **čistá próza** je mnohem lepší než
divadlo. Nejlíp funguje česká Wikipedie:

```bash
./jelly wiki           # stáhne kurátorované české wiki články do data/raw
./jelly index          # vyčistí (reference, datové závorky) → data/processed
./jelly qa             # QA dataset s lepším dělením vět + filtrem kvality
```

Každý řádek datasetu je `{question, context, answer, type, doc_id, passage_index}`.
Otázky vznikají šablonou z entit; filtr kvality zahodí zjevné paskvily. Část
zbývajících otázek je pořád gramaticky kostrbatá — to je očekávaný „realistický
strop" jednoduché šablony.

## V3 — pravidlové odpovědi (experimentální)

Odpověď se **skládá pravidly**, ne generuje: fakta z retrievalu, výběr entity podle
typu (NameTag) a role (UDPipe podmět/předmět), tvar z lemmat/MorphoDiTy, věta ze
šablony. ÚFAL nástroje běží každý jako **vlastní localhost služba** (řeší SWIG
konflikt). Když nic nesedí, spadne na extraktivní. Vyžaduje `requirements-v2.txt`
a modely.

```bash
./jelly qa-models        # stáhne NameTag + MorphoDiTa + UDPipe modely
./jelly wiki && ./jelly index
./jelly annotate         # offline: entity + role k pasážím (data/annotations.pkl)
./jelly template "kdy se narodil Karel Čapek?"   # → 1890
./jelly template         # interaktivní pravidlový prompt
```

Přepínač `--template` funguje i u `ask`/`repl`. Experimentální: u jasných otázek
dává čisté správné odpovědi, u víceslovných jmen zatím ustřeluje shoda (další krok =
zapojit MorphoDiTa skloňování celé fráze). Detaily v
`docs/superpowers/2026-07-16-v3-results.md`.

## V4a — bohatá analýza otázky + sponové odpovědi (experimentální)

Rozšíření pravidlového jádra V3. Otázka se rozebere **podle lemmatu** (`Jaká/Jaké`
→ `Jaký`, přibyly `Který`/`Čí`), takže tvary tázacích slov přestaly padat na
extraktivní. U otázek „X je/byl Y" a u `Jaký/Který` vezme **přísudek sponové věty**
(„Němcová byla významná spisovatelka" → „významná spisovatelka"), ale jen když
podmět té věty **sedí na téma otázky**. U definice bez vhodného přísudku už
answerer nevrací podmět (žádné „Kdo je Rossum? → Rossum"), ale spadne na
extraktivní větu. Bez změny dat (anotace z V3 stačí).

```bash
./jelly template "kdo je Rossum?"          # → věta o Rossumovi (ne tautologie)
./jelly template "Jaká byla Božena Němcová?"
```

Poctivé omezení: čistou jednoslovnou definici složí, jen když retrieval vytáhne
definiční větu nahoru — což měla zlepšit fáze B. Detaily v
`docs/superpowers/2026-07-16-v4a-results.md`.

## B1 — vzdálenostní jádro (⚠️ zakonzervováno)

> Neukázalo se jako jasná výhra → **zakonzervováno** (kód i testy zůstávají, default
> `passage`). Hlavní směr je faktový graf. Viz `docs/superpowers/conserved-components.md`.


Větný retrieval: skóruje po **větách**, nalezená věta vyzařuje skóre do okolí
s útlumem podle vzdálenosti (`exp(−d/τ)`, soubor = tvrdá hranice), vrchol → ostřicí
okno. Anotace jsou nově **větné** (`(doc_id, index věty)`), takže answerer skládá
anotaci libovolné pasáže z rozsahu jejích vět.

```bash
# v config.py: RetrieverConfig.granularity = "sentence"
./jelly annotate      # větné anotace (nutné po změně formátu)
./jelly index         # postaví větný index
./jelly template "Jaká byla Božena Němcová?"
```

**Poctivý výsledek — smíšený, ne jasná výhra.** Na tomhle korpusu větný režim jednu
otázku zlepší (Němcová → přísudek), ale definiční („Josef Čapek → český malíř") a
datové („1890") zhorší, protože kanonická definice bývá v úvodní pasáži, kterou
hrubší passage režim zachytí celou. **Default proto zůstává `"passage"`.** Detaily
a další směr (ladění τ/poloměru, hybrid) v `docs/superpowers/2026-07-16-vb1-results.md`.

## Faktový graf (experimentální)

Jiná cesta než retrieval: z celého korpusu se poskládá **reifikovaný n-ární faktový
graf** — každá slovesná událost je **faktový uzel** (predikát + váha = opakování),
k němu role-hrany na účastníky (`subj/obj/time/loc/num/pred`). Odpovídá **průchodem**
(téma → fakt s nejvyšší vahou → hodnota), takže opakovaně potvrzený fakt vyhrává a
n-ární fakt odpoví na „kdy" i „kde" z jedné události.

```bash
./jelly annotate            # větné anotace (vstup grafu)
./jelly graph               # postaví data/graph.pkl (fakty jako uzly)
./jelly graph-ask "kdy se narodila Božena Němcová?"   # → 2. května 1818
./jelly graph --view        # export do viewBase (force-graph; potřebuje networkx)
```

**Výsledek:** funguje mechanismus i n-arita (Němcová: „kdy"→1818 a „kde"→Slezsku
z téhož faktu; „kdo napsal Babičku"→Božena Němcová; „kde se narodil Karel Čapek"→
Malých Svatoňovicích). Extrakce má **aktivační koreferenci** (pro-drop podmět +
přesun tématu), **kanonizaci osobních jmen** a **precizní answerer** (žádná špatná
odpověď). Zbývá zachytávání zanořených dat (rok narození). Default `mode` zůstává
`extractive` → nulová regrese. Detaily v
`docs/superpowers/2026-07-16-fact-graph-results.md`.

## Roadmapa

- **Hotovo:** V1 (retrieval + extraktivní QA), V2a (syntetická QA data),
  V3 (pravidlové odpovědi — služby + role + šablony), V4a fáze A (bohatá analýza
  otázky + sponové odpovědi), B1 (větný retrieval — experimentální), faktový graf
  (reifikované n-ární fakty — experimentální). V2b (generátor od nuly) je na
  samostatné větvi.
- **Další:** zlepšit extrakci faktů (filtr podmětů, reflexiva, normalizace tvarů),
  vizualizace tras dotazu ve viewBase, konverzační „těžiště" (B2), a *podmíněně*
  konsolidace kódu k grafu.

Detaily v `docs/superpowers/specs/2026-07-15-cesky-gpt-design.md` a
`docs/superpowers/plans/2026-07-15-cesky-qa-v1.md`.
