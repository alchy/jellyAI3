# PRŮCHOD — krok po kroku jedna věta od blobu k odpovědi

Konkrétní, měřitelný průchod JEDNÉ reálné věty celým konceptem. Roste po krocích
(Krok 1, 2, …). U každého kroku: **co děláme teď** vs **jak to má být správně**.
Zkratky vysvětluje **Slovníček** na konci. Kotva pojmů: `SOULAD.md`.

Vzorová věta (blob `bible_lukas`):

> **„Odešel do galilejského města Kafarnaum a učil je v sobotu."**

---

## Krok 1 — věta z blobu → SEKTORY (anotace ÚFAL)

**Co děláme teď:** UDPipe (offline, cache `annotations.pkl`) rozseká větu na tokeny;
každý = **SEKTOR** s `lemma / UPOS / pád / čas / deprel / head`. A bereme to, jak přijde —
`roles.decompose` pak slepě věří `deprel`.

**Jak rozsekání dopadne** — všechny sloupce, poslední **„co je špatně"**. `deprel` = surovina
z ÚFAL (vstup); **`role = role_key(deprel + pád + předložka)`** = náš klíč (`role_catalog`
z `cs.json`, viz Slovníček). Katalog je jemnější než deprel — JEDEN deprel → víc rolí
(`obl` → `where`/`when`/…):

| # | sektor | UPOS | lemma | pád | deprel | role (náš klíč) | co je špatně |
|---|--------|------|-------|-----|--------|-----------------|--------------|
| **Ø** | *(elidovaný podmět)* | — | *on* | Nom | *chybí* | `who` | **chybí** — pro‑drop; konatel OBOU přísudků = **Ježíš** (doplnit z těžiště) |
| 1 | Odešel | VERB | odejít | — | root | `action` | — |
| 2 | do | ADP | do | Gen | case | — | — |
| 3 | galilejského | ADJ | galilejský | Gen | amod | `which_attribute` | — |
| 4 | města | NOUN | město | Gen | obl | `where` | — |
| 5 | Kafarnaum | PROPN | Kafarnaum | Nom | nsubj | `who` | **lže** — jméno města, ne konatel; má být `where` |
| 6 | a | CCONJ | a | — | cc | — | — |
| 7 | učil | VERB | učit | — | conj | `action` | — |
| 8 | je | PRON | on | Acc | obj | `whom_what` | **holé zájmeno** — „je" = zástup; doplnit referent |
| 9 | v | ADP | v | Acc | case | — | — |
| 10 | sobotu | NOUN | sobota | Acc | obl | `where` | **špatná role** — „v + akuzativ" = časové → má být `when` |
| 11 | . | PUNCT | . | — | punct | — | — |

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

### deprel — závislostní vztah (co sektor VE VĚTĚ dělá) → katalogová role

| deprel | česky | → role katalogu |
|---|---|---|
| root | kořen věty (hlavní přísudek) | `action` / `state` |
| nsubj | podmět | `who` / `what_subject` |
| nsubj:pass | podmět trpného rodu | `who` / `what_subject` |
| obj | přímý předmět (akuz.) | `whom_what` |
| iobj | nepřímý předmět (dat.) | `to_whom` |
| obl | příslovečné/nepřímé určení (s předložkou) | `where` / `when` / `with_whom_what` … (dle pádu+předl.) |
| cop | spona (být) | → přísudek jmenný (`state`) |
| amod | adjektivní přívlastek | `which_attribute` |
| nmod | jmenný přívlastek | `which_attribute` |
| appos / flat | apozice / víceslovné jméno | *(součást entity)* |
| case | předložka (case marker) | *(nese pád/směr)* |
| cc | souřadicí spojka | *(spojuje)* |
| conj | konjunkt (druhý člen/klauzule) | *(dědí roli řídícího)* |
| advmod | příslovečné určení (příslovce) | `how` / `where` / `when` (i vztažné „kde/kam") |
| acl / acl:relcl | vztažná/adnominální klauzule | *(dědí místo #59)* |
| advcl | příslovečná klauzule | `for_what_purpose` / `on_what_condition` / `in_spite_of` / `why` |
| xcomp | doplněk (druhotný přísudek) | `as_what_state` |
| punct | interpunkce | *(modalita věty)* |
