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

## Roadmapa

- **Hotovo:** V1 (retrieval + extraktivní QA), V2a (syntetická QA data),
  V3 (pravidlové odpovědi — služby + role + šablony). V2b (generátor od nuly) je
  na samostatné větvi.
- **Další:** zapojit MorphoDiTa skloňování celé fráze (shoda), predikátové otázky,
  hierarchický retrieval; obecné znalosti mimo korpus.

Detaily v `docs/superpowers/specs/2026-07-15-cesky-gpt-design.md` a
`docs/superpowers/plans/2026-07-15-cesky-qa-v1.md`.
