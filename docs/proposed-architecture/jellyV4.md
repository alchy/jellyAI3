# jellyV4 — návrh nové architektury

> Kritická reflexe prototypu V3 a návrh elegantnějšího, DRY a přívětivějšího
> modelu. **Jediné, co přežívá beze změny záměru: viewBase** (jen se z něj stane
> přímý odběratel grafu, ne ručně mostěný klient). Cíl: stejná funkce, jiné a
> lepší základy. Psáno po reálném provozu a ladění V3 na `jelly.ithosaudio.eu`.

---

## 0. Diagnóza: proč je V3 na vratkých nohou

Prototyp V3 **funguje**, ale jeho křehkost není náhoda — pramení z několika
architektonických rozhodnutí, která se navzájem znásobují. Konkrétně (z trápení,
ne z teorie):

1. **Dva grafy, ručně sešívané.** Web proces (viewBase) a `iris_service` mají
   KAŽDÝ svou instanci `FactGraph`. Runtime zápis (Mnemos) jde do iris grafu,
   ale vizualizace čte z webového. Musel jsem postavit most (`memorized`/
   `forgotten` v REST odpovědi → `feed_fact`/`remove_facts`, replay deníku do
   webového grafu). Každá operace (přidání, zapomenutí, kanonizace) potřebuje
   most zvlášť → resurrection bug (aktivace křísila zapomenuté uzly), nekonzistence.

2. **Dvě jazykové cesty.** Korpus se anotuje UDPipem (kontext), Mnemos má vlastní
   LEHKÝ parser (`_l_form`/`_finite_verb`). Predikát na zápisu („bydli"→„bydl")
   ≠ predikát z dotazu → nesejdou se. „bydli" (bez diakritiky) se zmrší na „bydl".

3. **Čtyři paralelní stemmery/nominativizace.** `_l_form`, `name_stem`,
   `cluster_key` (canon), `_to_nominative` (template), `hygiene.nominativize`,
   `predicate_catalog` — každý dělá kus kanonizace jinak a nekonzistentně
   (cluster_key slévá Pavel≡Pavla; morpho izolovaně mrší „Brně"; UDPipe zas
   slévá Pavla→Pavel). **Chybí JEDEN spolehlivý určovač tvaru.**

4. **Morfologie na izolovaném tvaru.** Mnemos pracuje po tokenech bez věty →
   morpho hádá pád špatně, jména na -l (Emil/Karel) `_l_form` bere jako
   l-příčestí, vymyšlená jména selhávají, typ (osoba/pojem) se přiřazuje natvrdo
   (žádný NER za běhu).

5. **Služby jako líné subprocess + health-first.** `UfalClient` spouští
   nametag/udpipe/morpho jako HTTP procesy na pevných portech, líně, a sdílí je
   přes health-check. Kolize portů (musel jsem přemapovat), zaseklé staré
   instance (web se přes health-first napojil na starý kód). Nedeterministický
   životní cyklus.

6. **Aktivace jako stav bez disciplíny.** `theme=uživatel` je na KAŽDÉM faktu
   1. osoby → uzel „uživatel" chronicky přehřátý → „kdo bydlí s Karlem?" driftuje
   na „uživatel".

7. **Karty vybírané specificitou** (`deck.best`) jsou elegantní, ale subtilní —
   rys „email" jsem musel vynutit potlačením ostatních rysů, aby karta vyhrála.

**Společný jmenovatel:** stav je roztroušený (dva grafy, aktivace, sklady), cesty
jsou duplikované (zápis vs dotaz, korpus vs runtime), a chybí jedna důvěryhodná
lingvistická vrstva. To se neopravuje záplatami — to se přestaví.

---

## 1. Principy jellyV4

1. **Jeden zdroj pravdy = append-only log událostí.** Vše ostatní (graf,
   aktivace, vizualizace) je DETERMINISTICKÁ projekce logu. Mnemos = tento log.
2. **Jedna jazyková cesta.** Text (výrok i otázka) prochází JEDNÍM pipeline
   jednou → typované tokeny. Zápis a dotaz sdílí normalizaci → nikdy nesejde
   predikát write vs query.
3. **Pozorovatelný graf.** Mutace grafu emituje delty; viewBase je odběratel.
   Žádný ruční most, žádné dva grafy.
4. **DRY jako zákon.** Právě jedna cesta pro: normalizaci tvaru, přidání faktu,
   entity-resolution, kanonizaci. (Dnes je každá 2–4×.)
5. **Rozhodnutí v datech, mechanismus v kódu** (zákon V3 zůstává) — ale nad
   ČISTÝMI typovanými tokeny, ne nad ad-hoc rysy.
6. **Deklarativní projekce > imperativní mutace.** Zapomenutí = `retract`
   událost, ne sáhnutí do grafu. Kanonizace = pravidlo projekce, ne krok navíc.

---

## 2. Architektura — vrstvy

```
   Text (výrok / otázka)
        │
  ┌─────▼──────────────────────────────────────────────────────────┐
  │ [1] LINGVISTICKÁ BRÁNA  (jediná, sdílená zápisem i dotazem)      │
  │     tokenizace → UDPipe(kontext) → morpho lemma → NER            │
  │     → diakritika-restore → KATALOG výjimek → gender/animacy guard │
  │     výstup: TypedSpan[] {form, lemma, upos, case, gender, ner}   │
  └─────┬───────────────────────────────────────────────────────────┘
        │
  ┌─────▼───────────────┐        ┌──────────────────────────────────┐
  │ [2] INTERPRET       │        │ [7] ANSWERER (dotaz)             │
  │ TypedSpan+stav →    │        │ TypedSpan → pseudo-QL → match    │
  │ Intent (assert |    │        │ nad grafem → odpověď + aktivace  │
  │ retract | remind |  │        └────────────┬─────────────────────┘
  │ recall | ask)       │                     │ (jen ČTE projekci)
  │ rozhoduje: KARTY    │                     │
  └─────┬───────────────┘                     │
        │ append                              │
  ┌─────▼───────────────────────────────┐     │
  │ [3] EVENT LOG (append-only JSONL)    │◀────┼─── ZDROJ PRAVDY (Mnemos)
  │ {ts, actor, kind, spans, raw}        │     │
  └─────┬───────────────────────────────┘     │
        │ fold (čistá funkce)                  │
  ┌─────▼───────────────────────────────┐     │
  │ [4] PROJEKCE: log → FactGraph        │     │
  │ reifikované fakty + entity-resolution │     │
  │ (kanonizace = lemma + typ + guard)   │     │
  └─────┬───────────────────────────────┘     │
        │                                      │
  ┌─────▼───────────────────────────────┐     │
  │ [5] OBSERVABLE FactGraph  ───delty───┼─────┘
  └─────┬───────────────────────────────┘
        │ subscribe (v procesu, žádný REST most)
  ┌─────▼───────────────────────────────┐
  │ [6] viewBase projekce (odběratel)    │  ← PŘEŽÍVÁ z V3, jen se odpojí od mostu
  └──────────────────────────────────────┘
```

### [1] Lingvistická brána — „vlastní lematika" (viz doporučení z BACKLOGu)

**Jediný důvěryhodný modul určující tvar.** Sjednocuje dnešní čtyři stemmery.
Vstup = surový text (celá věta → kontext!), výstup = `TypedSpan[]`:

```
TypedSpan {
  form:   "Karlem"          # povrch
  lemma:  "Karel"           # nominativ (jméno/místo) / infinitiv (sloveso)
  upos:   PROPN             # slovní druh
  case:   ins               # pád
  gender: masc              # rod (guard Pavel≠Pavla)
  ner:    person            # typ entity (osoba/místo/dílo/…)
}
```

Vnitřní řetěz (fallback shora dolů, KAŽDÝ krok s pojistkou):
1. **diakritika-restore** — čeština bez háčků je realita; „bydli"→„bydlí",
   „Petrovicich"→„Petrovicích" ještě PŘED morfologií (malá tabulka + morpho
   guesser jen na kandidáty).
2. **UDPipe kontextově** — pád, POS, NER; nejlepší pro MÍSTA (Brně→Brno).
3. **morpho lemma** — pro JMÉNA (Karlem→Karel), kde UDPipe slévá rody.
4. **gender/animacy guard** — nikdy neslít Pavel(masc)≠Pavla(fem); Le/„vakuová"
   past (poučení V3).
5. **KATALOG výjimek** (data) — pro známé zmršené tvary; rozšiřuje se, ne
   univerzalizuje (univerzální algoritmus rozbil klasifikaci — poučení V3 #32).

Fungovalo by: nominativ jmen i míst, správný typ, čistý predikát (bydlet) —
a hlavně **stejně pro zápis i dotaz**, protože je to jedna brána.

### [3]+[4] Event log + projekce — jádro elegance

- **Mnemos NENÍ mutace grafu, ale append do logu.** Fakt = `assert` událost se
  spany. Zapomenutí = `retract`. Uživatelova připomínka = `remind`.
- **Graf = `fold(log)`** — čistá deterministická funkce. Kdokoli (server, test,
  budoucí druhý uzel) přehraje log a dostane BITOVĚ týž graf. → zmizí „dva grafy",
  replay-nekonzistence, resurrection.
- **Entity-resolution je KROK projekce**, ne runtime záplata: shluk podle lemmatu
  (brána [1]) + gender/animacy guard. Korpus i runtime = TÁŽ funkce (dnes
  `resolve_entities` vs `_canonicalize_names` — dvě).
- **Forget je čistý** — `retract` událost; graf se přeprojektuje. Žádné ruční
  `remove_facts`/`remove_node`/hlídání osiřelých uzlů.

### [5]+[6] Pozorovatelný graf → viewBase odběratel

- FactGraph emituje `NodeAdded/NodeUpdated/NodeRemoved/EdgeAdded/…`.
- viewBase je **odběratel v témže procesu** — dostane deltu, přeloží na viewBase
  akci. Titulky/atributy jdou automaticky (uzel nese lemma+typ z brány [1]).
- Zmizí: `memorized`/`forgotten` v REST, `feed_fact`/`remove_facts` most, replay
  do webového grafu, resurrection guard v `update_node`. **Jeden graf, jeden
  vlastník, view jen zrcadlí.**

### [2] Interpret + karty

- Karty (JSON) zůstávají jako mechanismus rozhodování, ale triggery jsou nad
  ČISTÝMI `TypedSpan` rysy (upos/case/ner), ne nad ad-hoc `_l_form`/features.
  → email atribut je prostě „span s ner=email", ne vynucené potlačení rysů.

---

## 3. Jak V4 řeší každou bolest V3

| Bolest V3 | Řešení V4 |
|---|---|
| Dva grafy, ruční most | Jeden graf = projekce logu; viewBase odběratel v procesu |
| Resurrection (aktivace křísí zapomenuté) | Forget = `retract` → přeprojekce; aktivace je odvozená vrstva, ne zdroj uzlů |
| write ≠ query predikát | Jedna lingvistická brána sdílená zápisem i dotazem |
| bydli→bydl, 4 stemmery | Jedna brána [1] + katalog; lemma = nominativ/infinitiv |
| Pavel≡Pavla / Brně | UDPipe pro místa, morpho pro jména, gender/animacy guard, katalog |
| jména = „pojem" | NER v bráně [1] → typ osoba/místo/pojem hned |
| zaseklé služby, porty | Jeden supervised model-worker s čistým API, ne líné subprocess+health |
| theme=uživatel drift | Aktivace jako odvozená projekce s tlumením; user není účastník každého faktu (autor = metadata události, ne uzel grafu) |
| karty subtilní specificitou | Triggery nad typovanými spany (ner/upos), ne nad rysy z heuristik |

---

## 4. Procesní model

- **Jeden serverový proces** vlastní log + graf + aktivaci. Iris (dialog) i web
  (viewBase) běží v něm (async), ne jako oddělené procesy s REST mostem.
- **Model-worker**: UDPipe/morpho/nametag načtené JEDNOU v jednom supervisovaném
  workeru (proces s frontou nebo přímo `ufal.*` ve vlákně s GIL disciplínou),
  za JEDNÍM rozhraním `analyze(text) → TypedSpan[]`. Konec pevných portů,
  health-first, kolizí.
- **Perzistence** = jen event log (JSONL) + volitelný snapshot grafu (cache
  projekce). Restart = přehraj log. Deterministické.

---

## 5. Co přežívá, co se přepisuje

**Přežívá (a integruje se čistěji):**
- **viewBase** — jen se stane odběratelem observable grafu (zmizí most).
- **Karty** (JSON) jako mechanismus rozhodování — nad typovanými spany.
- **ÚFAL modely** — za jedním rozhraním.
- **Reifikovaný faktový graf** jako datový model (osvědčil se).

**Přepisuje se:**
- Mnemos lehký parser → jedna lingvistická brána.
- `UfalClient` (subprocess+porty) → jeden model-worker.
- Ruční graf-mutace + most do viz → event log + projekce + observable graf.
- Roztroušené stemmery/nominalizace → brána [1].
- Retriever (pokud zůstane) → ŘÍDKÁ matice (V3 hustá = OOM na velkém korpusu).

---

## 6. Migrace (návrh fází)

1. **Brána [1]** samostatně (testovatelná: text→TypedSpan[], s katalogem a guardy).
   Nahradí 4 stemmery. Měřeno na dnešních pastech (Pavel/Pavla, Karlem, Brně, bydli).
2. **Event log + projekce [3][4]** — Mnemos jako log; graf = fold. Forget/kanonizace
   jako projekce. Paralelně s V3, ověřit bitovou shodu grafu.
3. **Observable graf + viewBase odběratel [5][6]** — zahodit REST most.
4. **Model-worker** — nahradit subprocess+health.
5. **Interpret+karty [2] nad spany**, Answerer [7] sdílí bránu.

Každá fáze měřená (etalon/focus/dialog zůstávají guardrail).

---

## 7. Střízlivě: co V4 NEvyřeší kouzlem

Morfologie češtiny (natož bez diakritiky) je vnitřně tvrdá. V4 ji nezmagičí —
jen ji **soustředí do jedné důvěryhodné vrstvy s pojistkami a katalogem**, místo
aby ji rozpustil do heuristik po celém kódu. Vymyšlená jména (Ronik) morpho
nezná vždy; guard je nechá být, katalog je doplní. To je přijatelná, ČITELNÁ
hranice — na rozdíl od dnešního stavu, kde selhání jsou nečitelná a rozprostřená.

**Podstata:** V3 drží stav na více místech a jazyk řeší dvakrát. V4 má jeden log,
jeden graf jako jeho projekci, jednu jazykovou bránu a jednu vizualizaci, která
graf jen zrcadlí. Míň kódu, víc DRY, deterministické, a hlavně — pochopitelné.
