# PRŮCHOD — krok po kroku jedna věta od blobu k odpovědi

Konkrétní, měřitelný průchod JEDNÉ reálné věty celým konceptem. Roste po krocích
(Krok 1, 2, …). U každého kroku: **co děláme teď** vs **jak to má být správně**.
Zkratky vysvětluje **Slovníček** na konci. Kotva pojmů: `SOULAD.md`.

Vzorová věta (blob `bible_lukas`):

> **„Odešel do galilejského města Kafarnaum a učil je v sobotu."**

---

## Krok 1 — věta z blobu → WORD_W_ATTR_ARRAY (anotace ÚFAL)

> Slovníček atomů (SOULAD §1): **WORD_PLAIN** (holé slovo) · **WORD_W_ATTR** (slovo+atributy, dict)
> · **WORD_W_ATTR_ARRAY** (věta/okno) · **SLOT** (`run.slot()`) · **SLOT_ARRAY** = VZOR (`run.frame_sig()`).
> „SEKTOR" jsme retirovali — pole = přípona `_ARRAY`.

**Co děláme teď:** UDPipe (offline, cache `annotations.pkl`) rozseká větu na **SEKTORY**
(`lemma / UPOS / pád / deprel / head`). `standardize()` pak každý sektor přeloží do **NAŠEHO
čitelného klíče** (deprel ven — `nsubj`/`obl`/`amod` jsou nečitelné). Obsah ale zatím **dědí
chyby parseru** (viz sloupec „co je špatně").

**Standardizovaný výstup 1. fáze** — jen naše klíče; **UPOS zůstává** (čitelný), **`deprel` pryč**.
Každý sektor dostane klíč katalogu — i předložka/spojka/interpunkce:

| # | sektor | UPOS | lemma | pád | role (náš klíč) | co je špatně |
|---|--------|------|-------|-----|-----------------|--------------|
| **Ø** | *(elidovaný podmět)* | — | *on* | Nom | `who` | **chybí** — pro‑drop; konatel OBOU přísudků = **Ježíš** (doplnit z těžiště) |
| 1 | Odešel | VERB | odejít | — | `action` | — |
| 2 | do | ADP | do | Gen | `preposition` | — |
| 3 | galilejského | ADJ | galilejský | Gen | `which_attribute` | — |
| 4 | města | NOUN | město | Gen | `where` | — |
| 5 | Kafarnaum | PROPN | Kafarnaum | Nom | `who` | **lže** — UDPipe ji chybně označil za podmět; je to jméno města → má být `where` |
| 6 | a | CCONJ | a | — | `conjunction` | — |
| 7 | učil | VERB | učit | — | `action` | — |
| 8 | je | PRON | on | Acc | `whom_what` | **holé zájmeno** — „je" = zástup; doplnit referent |
| 9 | v | ADP | v | Acc | `preposition` | — |
| 10 | sobotu | NOUN | sobota | Acc | `where` | **špatná role** — „v + akuzativ" = čas → má být `when` |
| 11 | . | PUNCT | . | — | `punctuation` | koncová → nese modalitu „." |

**`roles.decompose` reálně vrátí** (dvě klauzule, sdílený elidovaný podmět):

```
«odejít»: action=odejít · which_attribute=galilejský · where=město · who=Kafarnaum ✗
«učit»:   action=učit    · whom_what=on(je) · where=sobota ✗
```

**Tři vady, které tabulka odhalí** (a které musí narovnat Krok 2):

1. **`who = Kafarnaum`** — zděděná chyba `deprel` (nsubj); má být místo. Podmět je elidovaný → **Ježíš**.
2. **`where = sobota`** — vlastní mezera mapy: „v + akuzativ" je **časové** (`when`), ne místní (`where`). `role_prepositions` musí zohlednit i **pád**, ne jen předložku.
3. **`whom_what = on`** — nerozřešené zájmeno („je" = zástup), holá výplň → doplnit referent.

**Princip kroku 1: anotace je VSTUP, ne PRAVDA.** Nese pro‑drop (chybějící konatel),
chyby `deprel` (Kafarnaum) i mezery mapy (v+akuz.). Krok 1 to **vynese na světlo**,
nepropustí dál. **Katalog (`role_catalog`) JSOU naše klíče všude; `deprel` je jen surovina,
kterou na ně přepočítáváme.** Narovnání dělá Krok 2.

---

## Slovníček (vysvětlivky zkratek v tabulkách)

### UPOS — slovní druh (Universal POS)

| tag | česky | příklad |
|---|---|---|
| NOUN | podstatné jméno (obecné) | město, síla, sobota |
| PROPN | vlastní jméno | Ježíš, Kafarnaum, R.U.R. |
| VERB | sloveso | odejít, učit, kázat |
| ADJ | přídavné jméno | galilejský, český |
| ADV | příslovce | nakonec, tam |
| PRON | zájmeno | on, je, kdo, co |
| DET | determinátor (ukazovací/přivlastňovací…) | ten, svůj, který |
| NUM | číslovka | dvanáct, 1937 |
| ADP | předložka | do, v, s, o |
| AUX | pomocné/sponové sloveso | byl (spona), jsem |
| CCONJ | souřadicí spojka | a, ale, nebo |
| SCONJ | podřadicí spojka | že, aby, když |
| PART | částice | ne, ať, by |
| INTJ | citoslovce | ó, hle |
| PUNCT | interpunkce | . , ? ! |
| SYM / X | symbol / neznámé | %, cizí token |

### PÁD (Case) — 7 českých pádů, UD zkratky

| UD | pád | otázka | typická role |
|---|---|---|---|
| Nom | 1. nominativ | kdo? co? | podmět (`who`/`what_subject`) |
| Gen | 2. genitiv | koho? čeho? | přivlastnění, po předl. (do, z, od) (`whose_of_what`) |
| Dat | 3. dativ | komu? čemu? | nepřímý předmět / adresát (`to_whom`) |
| Acc | 4. akuzativ | koho? co? | přímý předmět (`whom_what`) |
| Voc | 5. vokativ | *(oslovení)* | Hospodine! (mimo obsah) |
| Loc | 6. lokál | o kom? o čem? | po předl. (v, na, o) — místo/téma (`where`/`about_whom_what`) |
| Ins | 7. instrumentál | s kým? čím? | nástroj, průvod (`with_whom_what`) |

> Pád je klíčový: **pád pivotu JE role** (Nom→who, Dat→to_whom, Acc→whom_what). Proto
> musí být ve VZORu vždy — viz `SOULAD.md`, princip vzor „pán".

### Standardizované role — NAŠE klíče (výstup 1. fáze; `deprel` PRYČ)

Interně počítáme z UDPipe (`deprel` + pád + předložka), ale **výstup je vždy náš čitelný
klíč** — nečitelné zkratky `nsubj`/`obl`/`amod` se ven nedostanou.

**Přísudek** — `action` (slovesný děj) · `state` (jmenný se sponou, „byl králem") · `past_participle` (činné příčestí ‑l) · `passive_participle` (trpné ‑n/‑t)

**Podmět** — `who` (osoba: Kdo?) · `what_subject` (věc/jev: Co?)

**Předmět** (dle pádu) — `whom_what` (4. Koho/Co?) · `whose_of_what` (2. Koho/Čeho?) · `to_whom` (3. Komu/Čemu?) · `about_whom_what` (6. O kom/čem?) · `with_whom_what` (7. S kým/čím?)

**Příslovečné určení** — `where` (Kde/Kam/Odkud?) · `when` (Kdy?) · `how` (Jak?) · `how_much` (Kolik?) · `why` (Proč — příčina) · `for_what_purpose` (Za jakým účelem — záměr) · `on_what_condition` (Za jaké podmínky?) · `in_spite_of` (Navzdory čemu?)

**Rozvití** — `which_attribute` (přívlastek: Jaký/Který/Čí?) · `as_what_state` (doplněk: Jako jaký?)

**Strukturní** (glue — neodpovídají na otázku) — `preposition` (předložka) · `conjunction` (souřadicí spojka) · `subordinator` (podřadicí spojka) · `auxiliary` (pomocné sloveso) · `copula` (spona být) · `punctuation` (interpunkce, koncová nese modalitu) · `name_part` (část víceslovného jména) · `fixed_expr` (ustálený výraz)
