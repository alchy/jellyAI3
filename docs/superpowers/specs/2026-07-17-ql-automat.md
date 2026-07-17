# Spec: Iris — modulární stavový automat QL pro zaostření aktivace

> Přestrukturovaný brainstorming (uživatel, 2026-07-17) + interní iterace.
> Navazuje na dokončený arc pseudo-QL (parita 28/28, šablony primární/hybrid,
> commit `1c37ca8`). Vstřebává nedokončené Tasky 11–15 + 13b původního plánu
> (`2026-07-17-query-sablony.md`) — REST API, web, clarifikace, odchod UDPipe.

## 0. Terminologie (návrh — k vetu uživatelem)

- **Iris** — jméno stavového automatu (clona oka/objektivu: řídí, kolik světla
  — aktivace — a KAM dopadne; sedí k metafoře „rozsvěcení uzlů"). V dialogu
  s uživatelem vystupuje dál pod značkou **„QL:"**.
- **Pattern-karta** — jeden JSON = jeden vzor chování (trigger + dialog +
  akce + výuková poznámka). Slovník karet = konfigurace automatu.
- **Zaostření (focus)** — cílový stav: správné uzly svítí, špatné ne.
- **QueryAssurance** — číselná jistota zaostření (0–1), řídí přechody automatu.
- **Sharpener** — aktivační vrstva (kombinace aktivačních funkcí).
- **Pohrobek** — funkcionalita mimo směr automat+graf → knihovna s prefixem
  `conserved_` (nepoužitý, ale zachovaný kód).

## 1. Vize a role (z brainstormingu)

1. **Iris je primárně STAVOVÝ AUTOMAT pro správné ZAOSTŘENÍ na obsah** —
   aktivaci uzlů grafu. Jde o kvalitu dat, ne o formu odpovědi (tu později
   vyřeší jiná, zatím neexistující vrstva).
2. **Dotazování a interakce není nadstavba** — je rovnocenným partnerem
   jazykové analýzy. Když specifické matchování (šablona, pattern) není
   dost jisté nebo je chybové, NEpoužije se a nahradí ho **aktivační
   sharpener + dialog** (viz [[ql-dialog-first]]: dialog > figly).
3. Dialog může být vícekolový a **rekurzivní** — automat ostří s uživatelem,
   dokud QueryAssurance nedosáhne prahu; pak poctivě odpoví, NEBO poctivě
   řekne „nepodařilo se mi zaostřit; nejbližší možné odpovědi jsou: A, B…".

### 1.1 Ukázky dialogu (bez pevné struktury — strukturu dává konfigurace)

```
User: Co řekl Ježíš?
QL:   Mám zde více dat. Chceš upřesnit oblast datové aktivace?
      (aktivační okno: Janovi, Lukášovi, Magdaléně, učedníkům…)
User: ano, učedníkům
QL:   Po zaostření mám k učedníkům více výroků. Chceš vše,
      nebo konkrétního učedníka?
User: co řekl Janovi
```

```
User: Kde žil Ježíš?
QL:   V dokumentu jsem nenašel. Nejblíž je: (aktivační okno: Ježíš)
User: ano
QL:   Specifikuj období zaostření — mám tato období: (aktivační okno)
```

## 2. Stavový automat + pattern-karty (konfigurace jako data)

4. **Automat v kódu neřeší specifika jazyka** — chování řídí JSON
   pattern-karty. Přidání karty = rozšíření komunikačních schopností,
   bez zásahu do kódu.
5. **Jedna karta = jeden vzor**: obsahuje (a) **trigger** — při jaké události
   ji automat vytáhne (např. flag `focus-attention-query`, událost
   `resolve.ambiguous`, práh assurance), (b) **dialog** — šablona textu
   k uživateli, (c) **akce** — co s aktivací (warm kandidátů, čekej na volbu),
   (d) **teach** — výukové vysvětlení vzoru.
6. Vzorů bude „přehršel" a **doplňují se během testování** — karty pro
   upřesnění osoby, souvislosti, vztahu (ke komu/čemu), oblasti, roku,
   století, celého jména…; i pro stavy „nemám dost dat", „mám dat příliš".
6b. **Karty se modelují průběžně podle složitosti — komplexita roste**
    (doplněk uživatele): začíná se jednoduchými (first-match dle priority),
    postupně **matching dle otázky** — automat smí vyhodnotit VÍCE
    kandidátních karet a vzít tu s největším benefitem pro daný dotaz.
    Benefit = specificita triggeru (kolik podmínek sedí a jak těsně) ×
    priorita × očekávaný zisk zaostření; výhledově i spekulativní vyzkoušení
    karty (simuluj akci, změř nárůst aktivace správným směrem).
6c. **Benefit se MĚŘÍ jako nárůst aktivace** (potvrzení uživatele): po akci
    karty se měří Δ zaostření (koncentrace jasu na relevantních kandidátech);
    automat vede telemetrii karet (použití + zisk, součást API metadat).
    Díky stavovému automatu lze karty **dynamicky průběžně přidávat** a
    pokrývat vzory dotazů — vývoj karet řídí data (run_focus nad dialogovými
    scénáři), ne dojem.
7. **Výměna jazyka = přepnutí adresáře** s kartami + jazykovým JSON
   (stávající `jellyai/lang/cs.json` princip se rozšiřuje na celý automat).
   Automat využívá poznatky z českého korpusu, ale není na něj fixován.

### 2.1 Schéma pattern-karty (iterace — konkrétní tvar)

```json
{
  "name": "focus-offer-homonym",
  "trigger": {"event": "resolve.ambiguous", "min_candidates": 2,
              "assurance_below": 0.5},
  "dialog": "Mám možnost se věnovat tématu: {candidates} — kam mám zaostřit?",
  "action": {"warm_candidates": 0.5, "await": "user-pick"},
  "teach": "Více rovnocenných kandidátů jména (Kdo je Čapek?) → nabídka témat."
}
```

Taxonomie událostí (výchozí, roste testováním): `resolve.ambiguous`
(víc kandidátů), `resolve.miss` (nic — nabídni nejbližší), `focus.low`
(assurance pod prahem), `data.overflow` (příliš mnoho rovnocenných hodnot —
„Co řekl Ježíš?"), `data.empty`, `focus.ok` (odpověz).

## 3. Aktivace (sharpener)

8. Stávající aktivace = vztahy v grafu (trasy dotazu + spread). Nově
   **kombinace více aktivačních funkcí**: (a) trasová (dnešní), (b)
   **cross-distribuce** — svítí uzly kontextu i bez tras, (c) **vyzařování
   focusu (attention) po hranách** z aktivovaných uzlů. Původní koncept byl
   správný, jen jednoduchý.
9. **Aktivace se MĚŘÍ** — nový benchmark kvality zaostření (cílový uzel
   v top-K aktivačního pole po dotazu), vedle etalonu správnosti.
10. Úvaha (odloženo, nízká priorita): **hybridní model uzel×hrana** — hrana
    jako vztahové pojítko by mohla s menším ziskem aktivovat i navázané
    hrany. Vyžaduje experimenty a větší refaktor.

## 3b. Časové primitivy a intervaly — Chronos (doplněk uživatele, 2026-07-17)

10b. Iris umí pracovat s **časovými primitivy**: „dnes", „včera", „zítra",
     „za hodinu", „před dvěma hodinami", „před týdnem"… — rozkládá je na
     absolutní datum a čas relativně k okamžiku dotazu.
10c. „Tento týden", „tento měsíc" → Iris generuje **struktury časových
     intervalů** (půlotevřené ⟨start, end)) a umí s nimi počítat: příslušnost,
     průnik, granularita (hodina/den/týden/měsíc/rok).
10d. **Smysl = rozsvěcení přes čas**: má-li znalostní báze data provázaná na
     čas, časový uzel spadající do intervalu se rozsvítí a přes něj (hranami
     faktů) i propojení účastníci — časové zaostření je další aktivační
     funkce sharpeneru; reverse lookup („Co se stalo…?") umí interval místo
     přesného data.
10e. Návrhová rozhodnutí (iterace): slovník primitiv, směrovek („před"/„za"/
     „tento") a číslovek je **jazykové datum** (`cs.json`, klíč `temporal`);
     intervalová aritmetika je univerzální kód (`jellyai/iris/chronos.py`);
     **„teď" je VŽDY parametr** (`now`) — testy a benchmarky ho fixují
     (determinismus!), živé API bere systémové hodiny. Napojení na graf přes
     stávající `parse_date` (rok/měsíc/den časových uzlů); jemnější
     granularita (hodiny) je v intervalech připravená pro budoucí báze.
10f. **Princip: Iris je orientován v čase.** Automat zná „teď" svého běhu
     (vstřikované, ne natvrdo) — každý relativní výraz umí ukotvit na časovou
     osu a promítnout do aktivace. Čas je tak plnohodnotná osa zaostření,
     rovnocenná jmenné (kdo/co) a prostorové (kde).

## 4. Dialogové UX — tři okna (web)

11. **Okno 1 — dialog**: jen konverzace uživatel ↔ QL (dnešní konzole),
    včetně neaktivačních reakcí automatu („můžeš specifikovat osobu?…").
12. **Okno 2 — aktivační okno uzlů** (nové): seznam uzlů seřazený podle
    aktivace (největší nahoře) + váhy. Žádný dialog.
13. **Okno 3 — Aktivní dokumenty** (existuje): aktivace nad dokumenty.

## 5. API

14. **REST**: dotaz běžným jazykem dovnitř; odpověď nese **metadata** —
    které komponenty a které pattern-karty (dialog, vytěžení faktu, tematika…)
    byly použity pro odpověď/dialog; + assurance skóre, aktivační okno
    (seřazené uzly), aktivní dokumenty. (Absorbuje /query, /graphql, /schema
    z původního plánu — přímý pseudo-QL endpoint zůstává pro testovatelnost.)

## 6. Architektura

15. **Knihovna, ne monolit**: Iris je samostatná knihovna s **pluginy**
    (parser pseudo-QL, existence, reverse-date, drill… = zásuvné moduly).
    Core automat je malý; vyšší logika se staví NAD knihovnou.
16. **OOP**: třídy a jejich metody jsou dominantní architektura automatu.
17. **Výukovost**: vše má výukový kontext — minimálně polopatické
    vysvětlení funkcí/metod (docstring konvence projektu).
18. **Pohrobci**: funkcionality mimo směr automat+graf → `conserved_`
    knihovny (inventura, ne mazání).
19. **Underlying graf je prototyp** — smí se přetvářet bez zpětné
    kompatibility (i API), pokud to zlepší výsledky.
20. Postupný **refaktor celého QL kódu** do této architektury.

### 6.1 Struktura adresářů (návrh)

```
jellyai/iris/
  __init__.py      # fasáda knihovny (veřejné API)
  automaton.py     # jádro: stavy, události, výběr pattern-karet
  state.py         # FocusState — aktivační pole, historie, práh
  assurance.py     # QueryAssurance skóre (jistota zaostření)
  sharpener.py     # aktivační funkce (trasy, cross-distribuce, vyzařování)
  presenter.py     # zaostřená data → seřazené uzly + metadata (bez formy!)
  patterns.py      # loader/registr pattern-karet
  plugins/         # zásuvné dotazovací moduly (pseudo-QL parser, …)
  patterns/cs/     # pattern-karty češtiny (1 JSON = 1 vzor)
```

## 7. Vazba na existující kód (iterace)

- `answerer/query.py` (šablonový parser, parita 28/28) → **plugin Iris**
  beze změny sémantiky; `Pattern`/`SubQuery` zůstávají jazykem dotazu.
- `_resolve_topic` evidence patra (coverage/exact/ins/stem/da/loose
  + cluster-afinita) → **vstup QueryAssurance** (už spočtené, jen vytéct).
- `ActivationField` + `_spread` → základ sharpeneru (funkce (a)); (b) a (c)
  přibývají vedle, měřeny benchmarkem.
- Etalon (`run_etalon.py`, 28 jader + dialogy) zůstává **guardrail** — nikdy
  pod 100 %; dialogové scénáře přibývají jako `dialog` řádky.
- Nález „souvislost": (1) popisek hran `kontext` ve viz je ZÁMĚR
  (`viewbase_view.py:72`) — asociace nemá reálné slovo; (2) šumové uzly
  `souvislost`, „nutné pojímat v souvislosti…" = artefakt extrakce theme
  účastníků → stop-list funkčních substantiv (Fáze 0).

## 8. Fáze (rozdělení konceptu)

- **Fáze 0 — hygiena dat**: šum „souvislost" a větné slepence (stop-list
  v cs.json + délkový guard), inventura pohrobků (`conserved_`), baseline
  metriky aktivace.
- **Fáze 1 — skelet Iris**: knihovna (automaton/state/patterns/assurance),
  pattern-karty infrastruktura, pseudo-QL parser jako první plugin,
  **REST API** s metadaty (absorbuje /query, /graphql, /schema), etalon drží.
- **Fáze 2 — dialogové zaostřování + Chronos**: karty clarifikace/focus-offer/
  poctivý terminál/overflow; dialogové etalonové scénáře (ukázky z §1.1);
  rekurzivní ostření (SubQuery); **časové primitivy a intervaly** (§3b —
  `chronos.py`, `temporal` tabulky v cs.json, rozsvěcení časových uzlů
  intervalem, reverse lookup přes interval).
- **Fáze 3 — web tři okna**: dialogové okno + aktivační okno uzlů + Aktivní
  dokumenty; vše přes REST API.
- **Fáze 4 — sharpener**: cross-distribuce + vyzařování focusu; benchmark
  aktivace (`run_focus.py`), ladění vah funkcí.
- **Fáze 5 — čistý řez**: UDPipe pryč z query strany (gate: etalon 100 %
  šablonami), monolitický answerer → pluginy, pohrobci do `conserved_`.
- **Fáze 6 — experimenty** (odloženo): hybridní aktivace uzel×hrana.

## 9. Ne-cíle / mantinely

- Neřeší se **forma odpovědi** (kompozice textu) — jen kvalita zaostřených dat.
- Jazyk nikdy v kódu — jen v JSON (karty, tabulky).
- Determinismus, lokálnost (localhost API), žádná zpětná kompatibilita
  není vyžadována.
- Etalon + coverage + (nově) focus benchmark = objektivní řízení každé změny.

---
Souvisí s [[ql-dialog-first]], [[jellyai3-fact-graph]],
[[user-preferences-universality]]. Plán: `docs/superpowers/plans/2026-07-17-ql-automat.md`.
