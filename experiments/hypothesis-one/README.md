# hypothesis-two — FÁZE 1: věta → standardizované role

Precizní dokumentace **první fáze**: jedna věta z blobu (korpusu) se přesně, deterministicky
převede na **standardizované role sektorů** (naše čitelné klíče) + jejich **VZORy**, s
**kurátorskou opravou** chyb anotátoru. ÚFAL (UDPipe) slouží **jen k anotaci**; rozhodování je
naše a deterministické.

> Souvisí: `SOULAD.md` (pojmy/principy), `PRUCHOD.md` (názorný průchod jednou větou),
> `STATE.md` (stav). Tento README je **technický referenční popis fáze 1 a jejích datových typů**.

---

## 0. Přehled fáze 1

```
BLOB (dokument)
  │  věta (text)
  ▼
[1a] ANOTACE (UDPipe, offline)        →  SEKTORY (tokeny s UPOS/lemma/pád/deprel/head)
  ▼
[1b] STANDARDIZACE (standard_role)    →  ROLE (náš klíč: who/where/action/preposition…)  ← deprel PRYČ
  ▼
[1c] VZOR (frame_sig, r)              →  signatura okna (přesná: UPOS+pád+čas, pivot nese pád)
  ▼
[1d] KURÁTORSKÁ OPRAVA (curated.json) →  VZOR v DB ⇒ opravená role; jinak ponech
  ▼
VÝSTUP: [(sektor, role, kurátorováno?)]
```

Vše je **deterministické a čisté** (bez sítě, kromě jednorázové anotace `1a`, která je v cache
`data/annotations.pkl`).

---

## 1. Datové typy (slovníček)

| typ | Python | popis / pole |
|---|---|---|
| **Blob** | — | dokument korpusu; identifikátor `doc` (např. `"bible_lukas"`). V `annotations.pkl` klíč `(doc, sentence_index)`. |
| **WORD_PLAIN** | `str` | holé slovo (bez atributů) = `tok["form"]`. |
| **WORD_W_ATTR** | `dict` | slovo + atributy (pole níže) — anotovaná jednotka. |
| **WORD_W_ATTR_ARRAY** | `list[dict]` | pole slov s atributy = **věta / okno** (workhorse; v kódu `sent`). |
| **UPOS** | `str` | slovní druh z pevné sady (`VERB`, `NOUN`, …). Slovníček §6.1. |
| **Pád (Case)** | `str` | `feats["Case"]` ∈ {`Nom`,`Gen`,`Dat`,`Acc`,`Voc`,`Loc`,`Ins`}. Slovníček §6.2. |
| **deprel** | `str` | závislostní vztah z ÚFAL (`nsubj`,`obl`,…). **VNITŘNÍ vstup**, do výstupu se nedostane. |
| **Role (náš klíč)** | `str` | standardizovaný klíč katalogu (`who`,`where`,`action`,`preposition`,…) nebo `None`. Slovníček §6.3. |
| **SLOT** | `str` | match-obálka jednoho WORD_W_ATTR: `UPOS[:pád][:čas]` (`"PRON:Nom"`). Fce `run.slot()`. |
| **Modalita** | `str` | koncové znaménko věty ∈ {`.`,`?`,`!`,`:`}; `run.sentence_modality()`. |
| **SLOT_ARRAY** (≡ VZOR) | `str` | pole slotů: `slot·…·PIVOT·…·slot·modalita`. Hranice `^`/`$`. Fce `run.frame_sig()`. |
| **Kurátorský zápis** | `dict` | v `curated.json`: `{ SLOT_ARRAY: {"role": str, "note": str} }`. |
| **Výstup fáze 1** | `list[tuple[WORD_W_ATTR, str, bool]]` | `(word_w_attr, role, kurátorováno?)` z `Phase1.run()`. |

### WORD_W_ATTR — pole

| pole | typ | příklad | zdroj |
|---|---|---|---|
| `form` | `str` | `"Odešel"` | povrchový tvar |
| `lemma` | `str` | `"odejít"` | základní tvar (ÚFAL) |
| `upos` | `str` | `"VERB"` | slovní druh (ÚFAL) |
| `feats` | `dict[str,str]` | `{"Case":"Nom","Tense":"Past"}` | morfologie (ÚFAL) |
| `head` | `int` | `4` | 1-based index řídícího tokenu; `0` = kořen (ÚFAL) |
| `deprel` | `str` | `"nsubj"` | závislostní vztah (ÚFAL) — **vnitřní** |

---

## 2. Krok 1a — BLOB → VĚTA → WORD_W_ATTR_ARRAY (anotace ÚFAL)

- **Vstup:** text věty z blobu.
- **Nástroj:** **UDPipe** (`:8092`, offline), výsledek cachován v `data/annotations.pkl`.
  MorphoDiTa (`:8093`) je **měřeno rozbitá** — nepoužívá se; NameTag (`:8091`) doplňuje jména.
- **Výstup:** `Věta = list[Token]` (viz §1).
- **Zásada:** *anotace je VSTUP, ne PRAVDA.* Nese chyby (`nsubj = Kafarnaum`) i **pro-drop**
  (elidovaný podmět). Fáze 1 to musí vynést na světlo, ne propustit dál.

---

## 3. Krok 1b — SEKTOR → standardizovaná ROLE (náš klíč)

Funkce **`roles.standard_role(t, sent, byid, cop_heads) -> str|None`** — **nahrazuje nečitelné
deprel našimi klíči**. Pořadí rozhodování:

| # | podmínka | výsledek |
|---|---|---|
| 1 | `upos == "VERB"` | `action` (přísudek slovesný) |
| 2 | token je řídící sponové vazby (`cop`) | `state` (přísudek jmenný se sponou) |
| 3 | `deprel ∈ deprel_structural` | strukturní klíč: `case→preposition`, `cc→conjunction`, `mark→subordinator`, `aux→auxiliary`, `cop→copula`, `punct→punctuation`, `flat/appos→name_part` |
| 4 | jinak | `roles.role_of()` → obsahová role |

**`roles.role_of(t, sent, byid)`** (obsahové role):
- `conj` → zdědí roli řídícího (souřadné spojení),
- argumentové deprely (`nsubj`,`obj`,`iobj`,`obl`,`obl:arg`) → **`run.role_key(deprel, upos, feats, prep)`**,
- `advmod` → `where`/`when` (vztažné příslovce) nebo `how`,
- `amod`/`nmod`/`det` → `which_attribute`, `xcomp` → `as_what_state`.

**`run.role_key(deprel, upos, feats, prep="", nominal_pred=False)`** — JEDINÝ mapovač, sdílený i
registrem/bundle. Data v `cs.json` (zákon 3):
1. `nominal_pred` → `state`;
2. `prep` v `role_prepositions[k]` → `k` (`v/na→where`, `o→about_whom_what`, `s→with_whom_what`, `pro→for_what_purpose`);
3. `deprel_to_role[deprel]`: má-li `"anim"`, rozliší podle životnosti (`who`/`what_subject`), jinak podle **pádu** (`Acc→whom_what`, `Dat→to_whom`, `Loc→where`, `Gen→whose_of_what`, …).

**Výstup:** `roles.standardize(sent) -> list[(Token, role)]` — každý sektor v našem klíči, **deprel ven**.

---

## 4. Krok 1c — SEKTOR + okno → VZOR

Funkce **`run.frame_sig(toks, i, modality, r) -> str`**. VZOR = **přesná gramatická šablona**
(jako mluvnický vzor „pán"): `r` sektorů vlevo · **pivot** · `r` sektorů vpravo · modalita.

- **Sektor = `run.slot(tok)`** = `UPOS` + nosné rysy `Case`/`Tense` (`"PRON:Nom"`, `"VERB:Past"`).
  Interpunkce v `PUNCT_KEEP` se vrací doslova (`,`, `:`, …).
- **Pivot nese pád** (`slot(toks[i])`), protože **pád v pivotu JE role** → VZOR rozliší
  Kdo=`Nom`/Komu=`Dat`/Co=`Acc`, a přitom generalizuje přes lexém (`svolal ≈ pozval`).
- **`r` (poloměr)** řídí rozsah okna. Měřeno: r=1 → ⌀ 6,2 členů/vzor, r=2 → ⌀ 1,5. Pokrytí povrchů
  dělá **kvantita přesných vzorů**, ne rozostření jednoho.
- Hranice: `^` (okno přesahuje před začátek), `$` (za konec).

Příklad: `„Kdo svolal svých dvanáct?"` → pivot `Kdo` → `^·PRON:Nom·VERB:Past·?` (r=1).

---

## 5. Krok 1d — VZOR → KURÁTORSKÁ OPRAVA

Funkce **`roles.curated_standardize(sent, r=2) -> list[(Token, role, curated:bool)]`**. Pro každý
sektor spočte VZOR; je-li v **`curated.json`**, vrátí **opravenou** roli, jinak `standard_role`.

- **Proč:** anotace ÚFAL je u některých oken nekonzistentní/chybná. Opravou u VZORu **jednou**
  se chyba spraví pro **všechny** budoucí výskyty (pákový efekt: 4 vklady → 85 oprav napříč korpusem).
- **Kdy kuratela (a kdy ne):** nečistota oken má tři příčiny — **zahozená předložka** (řeší
  precizace VZORu), **chyba ÚFALu** (řeší **kuratela**), **sémantika slova** (strop okna, potřebuje lexém).
- **Formát zápisu:** `{ "<VZOR>": {"role": "<náš klíč>", "note": "<proč>"} }`.

---

## 6. Slovníčky

### 6.1 UPOS — slovní druh

| tag | česky | příklad |
|---|---|---|
| NOUN | podstatné jméno | město, síla |
| PROPN | vlastní jméno | Ježíš, Kafarnaum |
| VERB | sloveso | odejít, učit |
| ADJ | přídavné jméno | galilejský |
| ADV | příslovce | nakonec, tam |
| PRON | zájmeno | on, je, kdo |
| DET | determinátor | ten, svůj, který |
| NUM | číslovka | dvanáct, 1937 |
| ADP | předložka | do, v, s, o |
| AUX | pomocné/sponové sloveso | byl (spona), jsem |
| CCONJ | souřadicí spojka | a, ale |
| SCONJ | podřadicí spojka | že, aby, když |
| PART | částice | ne, ať, by |
| INTJ | citoslovce | ó, hle |
| PUNCT | interpunkce | . , ? ! : |
| SYM / X | symbol / neznámé | %, cizí token |

### 6.2 Pád (`feats["Case"]`)

| UD | pád | otázka | typická role |
|---|---|---|---|
| Nom | 1. nominativ | kdo? co? | `who` / `what_subject` |
| Gen | 2. genitiv | koho? čeho? | `whose_of_what` (i směr z/do → `where`) |
| Dat | 3. dativ | komu? čemu? | `to_whom` |
| Acc | 4. akuzativ | koho? co? | `whom_what` |
| Voc | 5. vokativ | *(oslovení)* | mimo obsah |
| Loc | 6. lokál | o kom? o čem? | `where` / `about_whom_what` |
| Ins | 7. instrumentál | s kým? čím? | `with_whom_what` |

### 6.3 Standardizované role — NAŠE klíče (výstup fáze 1; deprel PRYČ)

**Přísudek** — `action` (slovesný děj) · `state` (jmenný se sponou) · `past_participle` (činné příčestí -l) · `passive_participle` (trpné -n/-t)

**Podmět** — `who` (osoba: Kdo?) · `what_subject` (věc/jev: Co?)

**Předmět** — `whom_what` (4. Koho/Co?) · `whose_of_what` (2. Koho/Čeho?) · `to_whom` (3. Komu/Čemu?) · `about_whom_what` (6. O kom/čem?) · `with_whom_what` (7. S kým/čím?)

**Příslovečné určení** — `where` (Kde/Kam/Odkud?) · `when` (Kdy?) · `how` (Jak?) · `how_much` (Kolik?) · `why` (Proč — příčina) · `for_what_purpose` (Za jakým účelem) · `on_what_condition` (Za jaké podmínky?) · `in_spite_of` (Navzdory čemu?)

**Rozvití** — `which_attribute` (přívlastek: Jaký/Který/Čí?) · `as_what_state` (doplněk: Jako jaký?)

**Strukturní** (glue — neodpovídají na otázku) — `preposition` · `conjunction` · `subordinator` · `auxiliary` · `copula` · `punctuation` · `name_part` · `fixed_expr`

---

## 7. Pracovní příklad (fáze 1 celá)

Věta z blobu `bible_lukas`: **„Odešel do galilejského města Kafarnaum a učil je v sobotu."**
Standardizovaný výstup (deprel pryč; poslední sloupec = vada k narovnání):

| # | sektor | UPOS | lemma | pád | role (náš klíč) | co je špatně |
|---|--------|------|-------|-----|-----------------|--------------|
| **Ø** | *(elidovaný podmět)* | — | *on* | Nom | `who` | **chybí** — pro-drop → **Ježíš** |
| 1 | Odešel | VERB | odejít | — | `action` | — |
| 2 | do | ADP | do | Gen | `preposition` | — |
| 3 | galilejského | ADJ | galilejský | Gen | `which_attribute` | — |
| 4 | města | NOUN | město | Gen | `where` | — |
| 5 | Kafarnaum | PROPN | Kafarnaum | Nom | `who` | **lže** — jméno města → `where` |
| 6 | a | CCONJ | a | — | `conjunction` | — |
| 7 | učil | VERB | učit | — | `action` | — |
| 8 | je | PRON | on | Acc | `whom_what` | zájmeno → doplnit „zástup" |
| 9 | v | ADP | v | Acc | `preposition` | — |
| 10 | sobotu | NOUN | sobota | Acc | `where` | **má být `when`** (v+akuz = čas) |
| 11 | . | PUNCT | . | — | `punctuation` | modalita „." |

Podrobný rozbor a klauzule viz `PRUCHOD.md`.

---

## 8. Měření (fakta k fázi 1)

- **VZOR → role je funkce?** Při **r=2** je **82 %** opakovaných oken role-konzistentních
  (r=1: 62 %, r=3: 89 %). ⇒ kurátorovatelné a stabilní.
- **Pokrytí** (tokeny v opakovaném čistém okně): r=2 ≈ **32 %** — roste s velikostí korpusu
  (úzké hrdlo NENÍ konzistence, ale kolik oken se opakuje).
- **Kuratela:** 4 ruční vklady opravily **85 chyb** ÚFALu napříč korpusem.
- **Předložka ve VZORu** (`ADP:lemma`): +3 pp čistoty — dílčí, ne hlavní příčina.

---

## 9. Rychlé spuštění

```bash
# vyžaduje UDPipe :8092
.venv/bin/python -c "import roles as R; \
  print(R.curated_standardize(R.udpipe('Odešel do galilejského města Kafarnaum a učil je v sobotu.')[0]))"
```
