# FÁZE 2 — od rolí ke grafikonu (návrh)

Fáze 2 bere **výstup fáze 1** (role slov + provenience) a staví z něj syntetický graf
`FAKT ↔ OTÁZKA ↔ ODPOVĚĎ`. **Stav: NÁVRH** — kód zatím nestavíme; tento dokument fixuje
mechanismus. Architektura stejná jako fáze 1: třídy, **konfig v JSON**, testy po třídách,
dokumentace po třídách, atomy `WORD*/SLOT*`, atributy `UPOS/CASE/TENSE`.

> Souvisí: `README.md` (fáze 1), `SOULAD.md` (pojmy/principy), `PRUCHOD.md` (průchod větou).

---

## 0. Přehled fáze 2

**Vstup:** výstup fáze 1 — per slovo `(WORD_W_ATTR, role, provenance)` + per-klauzulové
rolové rámce, **s vyznačenými vadami** (pro-drop díra, holé zájmeno, `v+akuz`).

```
FÁZE 1 → role slov (+ vady)
   ▼
[2a] REPAIR      narovnání vad     → ÚPLNÝ rolový rámec faktu (bez děr)
[2b] FACT        rámec → FAKT       → strukturovaný fakt (predikát + role)
[2c] QUESTION    fakt → OTÁZKY      → jedna na díru + DOTAZY(Ollama) + VZOR (SLOT_ARRAY)
[2d] ANSWER      fakt → ODPOVĚĎ     → fragment + věta (odpověď = fakt → rekurze)
[2e] GRAPHIKON   spojení            → katalog FAKT ↔ OTÁZKA ↔ ODPOVĚĎ nad VZORy
```

**Runtime (dotaz):** živá OTÁZKA → VZOR → match v grafikonu → **projekce díra→výplň** → ODPOVĚĎ.

---

## 1. Krok 2a — Repairer: jak vznikne ÚPLNÝ rolový rámec

### 1.1 Odkud díry jsou a jak je poznáš

- **pro-drop** — klauzule má přísudek, ale **žádný token v roli `who`** („Odešel" = kdo?).
  Díru poznáš takto: *přísudek je, podmětová role chybí*. **Rysy přísudku** (`VERB:Past`,
  rod/číslo/životnost) říkají, **jaký** ten podmět má být → cíl doplnění.
- **holé zájmeno** — role je obsazená, ale hodnotou `on`/`je`/`jim` (zástupný token bez referentu).

### 1.2 Mechanismus — TĚŽIŠTĚ (`ActivationField`), NE graf

Doplnění díry je **sekvenční průchod + slábnoucí SKALÁRNÍ pole nad ENTITAMI**. Není to
šíření po grafu; `ActivationField` je jen mapa `entita → jas`:

```python
field = ActivationField(decay=0.55)      # slábnoucí mapa entita → jas
for sentence in document:                # POŘADOVĚ (antecedent = předchozí kontext)
    # 1) díru doplň ze STÁVAJÍCÍHO stavu pole
    if clause má elidovaný podmět:
        who = field.hottest( filtr = shoda(rod/číslo/životnost s VERB) )
    # 2) pak zahřej entity TÉTO věty
    for e in named_entities(sentence):
        field.warm(e, 2.0 if e is subject else 1.0)   # podmět teplejší
    field.step()                         # vše ×0.55 → pohasne
```

- **klíče = ENTITY** (Ježíš, Mojžíš), **ne každé slovo** — jde o salience *osob v diskurzu*,
  ne o difúzi slov;
- `warm` = zmínka přihřeje, `step` = každá věta pohasne → pole drží „kdo je teď v centru"
  (recentní + častý = horký);
- `hottest` **se shodovým filtrem** = nejteplejší entita, která **gramaticky sedí** k rysům
  přísudku/zájmena — to je celý „shodový filtr";
- zájmeno se rozřeší stejně (nebo koreferenčním řetězcem k antecedentu).

Je to tentýž mechanismus, co má originál (`jellyai/graph/extract.py` doplňuje elidovaný
podmět kopule z nejteplejší osoby přes `jellyai/graph/activation.py`) — naroubovaný.

### 1.3 Tři aktivace — NEPLÉST

V systému jsou tři různá „aktivační" pole pro tři různé úlohy:

| mechanismus | model | pro co |
|---|---|---|
| **`ActivationField`** | slábnoucí **skalár** nad entitami (sekvenční `warm`/`step`/`hottest`) | **pro-drop / těžiště** ← krok 2a |
| **`_spread`** | šíření po **hranách grafu** s váhou (glow) | asociativní retrieval (Ježíš→Bůh), runtime |
| **`spread_field`** | difúze tepla po **tokenech věty** (pseudo-attention) | doplnění titulů (jiná úloha) |

Krok 2a používá **jen ten první**. Graf-spread přijde až v runtime nad hotovým grafikonem.

### 1.4 Proč skalár, a ne graf

Pro-drop je otázka **SALIENCE** („kdo je teď hlavní"), ne **asociativní struktury**
(„co s čím souvisí"). Slábnoucí skalár přesně modeluje recency+frekvenci, je deterministický
a laciný. Graf-spread odpovídá na jinou otázku (najdi *související* uzly) → jiný krok.

### 1.5 „Bez děr" ≠ jistota → PROVENIENCE hodnoty

Doplněná hodnota je **hypotéza**, ne pozorování. Role proto nese příznak zdroje HODNOTY:

| provenance (2a) | význam |
|---|---|
| `OBSERVED` | role z reálného tokenu (fáze 1) |
| `FILLED` | doplněno z těžiště (pro-drop / koreference) — nižší jistota |

> Pozor, jiná osa než provenience fáze 1: `CURATED/CONFIRMED/CANDIDATE` říká, **jak byla role
> URČENA** (lookup); `OBSERVED/FILLED` říká, **odkud je HODNOTA** (token vs doplněno). Ortogonální.

Když **žádná shodná entita není dost teplá** (ambiguita, nový kontext), díra **zůstane**
(*tápání ≠ terminál* → ptáme se / přiznaná mezera). „Bez děr" = doplněno tam, kde těžiště dá
jistého shodného kandidáta, jinak označeno.

### 1.6 Na vzorové větě (bible_lukas)

„Odešel do galilejského města Kafarnaum a učil je v sobotu."
```
«odejít»  who = Ø → Ježíš   [FILLED, Masc Sg]   where = Kafarnaum [OBSERVED]
«učit»    who = Ø → Ježíš   [FILLED]   whom_what = je → zástup [FILLED/koref]   when = sobota [OBSERVED→opraveno]
```

### 1.7 Je `ActivationField` jediná volba? — VYNUCENÉ vs VOLBA

**Vynucené (ne volba): stavový sekvenční průchod.** Referent pro-dropu je *dokument-lokální*
(v Lukášovi Ježíš, v Exodu Mojžíš) → **nejde statický slovník `VZOR→entita`** (na rozdíl od
určení role, které statické JE). Anafora potřebuje běžící stav diskurzu — to je dané povahou úlohy.

**Volba: který model salience/koreference.** Alternativy (všechny deterministické; ML koreference
je mimo tezi a vyloučena):

| model | jak | pozn. |
|---|---|---|
| **`ActivationField`** (decay + role-váha) | `warm`/`step`/`hottest`, podmět těžší | nejjednodušší; už míchá recency + gramat. salienci |
| **Centering Theory** | Cb/Cf řazené gramatickou rolí + přechody | principiálnější, přesnější na zájmena; `warm(subj,2.0)` je jeho light verze |
| **Hobbs** | průchod závislostním stromem za antecedentem | čistě syntaktické |
| **nejbližší shodná zmínka** | recency baseline | degenerovaný `ActivationField` (decay→0) |

`ActivationField` a `Centering` **nejsou protiklady** — první je odlehčená verze druhého a dá se
plynule upgradovat. **Rozhodnutí:** `ActivationField` jako **vyměnitelný default** (model = konfig),
**změřit** na gold sadě pro-drop případů, a při nedostatečné přesnosti **povýšit na Centering**.
Nezabetonovat jeden model.

### 1.8 Inventář placeholderů + typová šablona (co nahrazovat)

Placeholdery **neseznamujeme ručně** — ÚFAL je značí rysem; každý má **typ = konečnou množinu**
přípustných výplní (agreement jako *eval type*):

| placeholder | ÚFAL rys | řeší se NA | typová šablona (fit) |
|---|---|---|---|
| **elidovaný podmět (Ø)** | *(chybí nsubj)* | nejteplejší shodná entita | Person/Gender/Number přísudku |
| **osobní 3. os.** | `PronType=Prs, Person=3`, lemma `on` | koreferenční antecedent | Gender/Number/Animacy |
| **osobní 1./2. os.** | `PronType=Prs, Person=1/2` (já/ty) | mluvčí / adresát | z rámu přímé řeči |
| **přivlastňovací** | `Poss=Yes` (jeho/její/jejich) | vlastník = antecedent | Gender/Number vlastníka |
| **přivlastň. zvratné** | `Poss=Yes` + zvratné (svůj) | podmět klauzule | — |
| **ukazovací samostatné** | `PronType=Dem` (to/ten) | anaforický antecedent | shoda |

Detekce: `PronType=Prs OR Poss=Yes OR (Dem samostatné) + Ø podmět`.
**NEnahrazovat:** `Reflex` (se/si — gramatické), `Int,Rel` (kdo/co/který — díra/vazba, #59),
`Tot/Neg/Ind` (všechen/nikdo/něco — kvantifikace/neurčitost).

### 1.9 Resoluce = zpětné okno se skórováním (persistentní slovník, obdoba fáze 1)

Obdoba `PersistentDeterminer`, ale pro koreferenci — **statický kurátorský slovník**. Mechanismus:

1. u placeholderu znáš **typ** → **konečnou množinu** přípustných entit (agreement = type-check);
2. **jdeš doleva větu po větě** — okno **n vět** (definovatelné jako `r`; discourse-jednotka > holý počet slov);
3. **plníš kandidáty dle váhy akumulace** — entita, co v okně padne častěji / blíž, má vyšší **score**;
4. vybereš **nejvýše skórujícího, který SEDÍ** (agreement). Nesedí → další; nikdo v okně → rozšiř okno / global salience / díra zůstane.

`to tam pasne?` = ten agreement/typový check je právě „sedí"; **score bez shody se nepočítá**.

**Backward cache = předchozích n vět (`WORD_W_ATTR_ARRAY`-ů), které už máme.** Nepotřebujeme zvláštní
strukturu — kroky 1–4 jsou **zpětný dotaz do předchozích vět**, jdeme **větu po větě** (rysy → agreement,
**vzdálenost ve větách → recency**). „Jdeme doleva větu po větě" = **lookup do backward cache dle
`WORD_W_ATTR_ARRAY`**. Pro efektivitu lze držet rolující index `entita → poslední věta`, ale zdrojem jsou
ta pole. Tím se `scan-left` a `ActivationField` **sjednotí**: decay = recency (vzdálenost ve větách) nad
backward cache, typová množina = agreement filtr nad ním.

**Persistentní slovník (VZOR placeholderu → resoluce):** klíč = `SLOT_ARRAY` okolo placeholderu;
hodnota = **vzorec resoluce** (typ + potvrzená relativní pozice kandidáta) s proveniencí
`CURATED/CONFIRMED/CANDIDATE`. **Statické je PRAVIDLO** (typ, okno, který relativní kandidát
vyhrává), **dynamická je konkrétní entita** — dokument-lokální, plní se za běhu. Slovník tedy
necuruje „Ježíš", curuje „vyhraje nejbližší shodný podmět 1–2 klauzule zpět".

**Provenience HODNOTY:** `OBSERVED` (token je tam) · `FILLED` (zpětné okno) · `ASSOCIATED` (global/graf — Bůh).

### 1.10 Otevřené (k měření)

- **okno vs global** — krátké okno = tuhá recency, ale mine vzdáleného protagonistu (zmíněný 40 slov
  zpět); dlouhé = chytí, ale šum. Možná **dva signály**: lokální okno (recency) + global salience
  (protagonista). `n` je `r`-obdoba, měřitelné.
- **co je statické** — vzorec/typ ano; konkrétní entita ne. Slovník curuje resoluční PRAVIDLO, ne entitu.

### 1.11 Naměřeno (build & measure — 5 knih, 3251 osobních zájmen 3. os.)

| shoda | kandidát ≤2 věty | nenalezeno | unikát @n=2 |
|---|---|---|---|
| rod **exact** + číslo | 5 % | **89 %** 💀 | 67 % |
| rod **overlap** + číslo | 38 % | 40 % | 54 % |
| **jen číslo** | 44 % | 31 % | 53 % |

- **Rod: ZAPNOUT (overlap), je to SPRÁVNÝ silný filtr — ne tie-breaker.** ÚFAL značí 72 % zájmen
  kombinovaně `Masc,Neut` (ho/mu/jej — syncretismus) → **exact-match je rozbité** (89 % none), ale
  **overlap to řeší** (permisivní, mužská zájmena nic chybně neodmítnou). Ověřeno: rod odmítne
  nejbližšího číselného kandidáta v **14 %**, a **295/298 SPRÁVNĚ** (Fem zájmeno → Fem entita:
  „k **ní**"→Maria ne Gabriel, „za **ni**"→tchyně ne Ježíš). **ÚFAL rod tu NENÍ nepřesný.**
- **Dřívější „pokles pokrytí rodem" = artefakt KANDIDÁTNÍ MNOŽINY, ne rodu.** Klíč: „za ni" =
  **tchyně** (obecné jméno, NOUN) — rod správně odmítl Ježíše (Masc), ale `tchyně` nebyla v
  PROPN-only množině → „none". Rod konal správně; selhala množina kandidátů.
- **n ≈ 2–3 věty** — přírůstek pokrytí je do ≤2–3 vět, dál plochý.
- **Score: stačí recency (nejbližší)** — ~47 % má víc kandidátů, ale nejbližší rozhodne; složité
  vážení na start netřeba.
- **PRAVÝ gap = kandidátní množina.** Cache MUSÍ obsahovat: **ženská obecná jména** (tchyně, matka),
  **všechny zmínky**, **řetěz vyřešených zájmen** (vyřešenou hodnotu **vracíme do cache**). Ne jen
  PROPN osoby. Jinak rod správně odmítne špatného, ale správný Fem referent chybí → díra / `ASSOCIATED`.

---

## 2. Kroky 2b–2e (stručně; detail až u stavby)

- **2b · FactFramer** — z narovnaných rolí sestaví FAKT jako rámec (per klauzule):
  `predikát + who/where/whom_what/when/…`. Fakt jako strukturovaný objekt, ne holá věta.
- **2c · QuestionSynthesizer** — fakt je **n-ární** → jedna OTÁZKA na každou rolovou díru;
  `role_ask` dá tázací slovo, Ollama offline **DOTAZY** (povrchy), `frame_sig` → **VZOR** = match-klíč.
- **2d · AnswerTemplater** — ODPOVĚĎ: fragment (výplň díry) + věta (odpověď jako tvrzení = fakt → rekurze).
- **2e · GraphikonBuilder** — spojí do grafikonu `FAKT ↔ {OTÁZKA/VZOR} ↔ {ODPOVĚĎ}`; hustotu hran řídí `r`.

---

## 3. Napojení na řez (dva světy)

Fáze 1 = **frame front-end** (slova → role → VZOR). Fáze 2 = **graf back-end** (grafikon +
asociativní retrieval). Tři pojistky asociace z řezu žijí tady:
- **těžišťová atribuce** → 2a (rodí `who=Ježíš`, později i `Bůh`),
- **hub-brána** a **identitní hlas** → runtime retrieval nad grafikonem.

---

## 4. Stav

Návrh. Až na pokyn: třídy `Repairer / FactFramer / QuestionSynthesizer / AnswerTemplater /
GraphikonBuilder` (+ `Phase2`), konfig v JSON (decay, warm váhy, práh shody, cesty),
testy po třídách (pro-drop doplní shodně, nedoplní při chybějícím kandidátovi, provenience),
dokumentace po třídách + HTML. Doporučené pořadí stavby: **2a Repairer první** (rozjede
těžiště, které nese i asociace v runtime).
