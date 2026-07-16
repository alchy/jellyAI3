# Faktový graf — výsledky

**Datum:** 2026-07-16 · **Větev:** `feature/fact-graph`

## Co je hotové

Reifikovaný **n-ární faktový graf**: každá slovesná událost = faktový uzel
(predikát + váha opakování), k němu role-hrany na účastníky (`subj/obj/time/loc/num/
pred`). Postaven z větných anotací (B1). `GraphAnswerer` (`--graph`) odpovídá
**2-skokem** (téma → fakt s max vahou → hodnota); jinak fallback na extraktivní.
Export do viewBase (faktové uzly + role-hrany). Default `mode` zůstává `extractive`
→ nulová regrese.

Jádro je hermeticky otestované (celá sada zelená). Reálný graf: **842 uzlů, 724
faktů** z 1402 vět.

## Živé odpovídání (`./jelly graph-ask`)

| Otázka | Odpověď (graf) | Verdikt |
|---|---|---|
| kdo napsal Babičku? | **Božena Němcová** | ✅ správně |
| kdy se narodila Božena Němcová? | **2. května 1818** | ✅ správně |
| kde se narodila Božena Němcová? | **Slezsku** | ✅ správně (z **téhož** faktu jako kdy) |
| kdy se narodil Karel Čapek? | 26. srpna 1935 | ❌ špatně (extrakční šum) |
| kdo je Rossum? | fallback (věta) | — (v grafu není sponová definice) |

**N-arita funguje:** „kdy" i „kde" u Němcové čerpají z jednoho narozovacího faktu
(`subj=Božena Němcová, time=2. května 1818, loc=Slezsku`) — přesně smysl reifikace.

## Chyba nalezená a opravená živě

„kdo napsal Babičku?" nejdřív vracelo „povídka": téma **Babička** (kniha) se
resolvovalo na uzel s nejvyšší vahou, což bylo obecné **babička** (žena). Oprava:
resolve tématu je **case-sensitive** a preferuje uzel pokrývající víc témat
(„Božena Němcová" > „Němcová"). Zajištěno hermetickým testem.

## Poctivá omezení (→ další práce)

Mechanismus i n-ární model se potvrdily; **slabina je kvalita extrakce z reálných
rozborů**:
- **Šum ve faktech** — u Čapkova narození vznikl fakt se špatným rokem (1935).
  Reflexivní „narodil se", složité věty a rozlišení podmětu produkují chyby; obecná
  podstatná jména jako podmět („dítě", „dcera", „syn") tvoří balast.
- **Tvary hodnot** — „Slezsku" místo nominativu „Slezsko" (atributy se neskloňují
  zpět; lze dořešit MorphoDiTou jako u V4a).
- **Řídkost** — jen věty s jasným podmětem+slovesem dají fakt.

Další směr: filtr podmětů (preferovat entity, zahodit obecná jména/zájmena), lepší
navázání data/místa na sloveso, reflexiva, subsumpce částečných faktů, a normalizace
tvarů. Teprve pak dává smysl uvažovat o *konsolidaci kódu k grafu* (nápad uživatele).

## Vizualizace

Export `to_networkx`/`to_json` posílá do viewBase i **faktové uzly** (typ `fact`,
popisek = predikát) jako waypointy — připraveno pro animaci **tras dotazu**
(téma → faktový uzel → hodnota), viz roadmapa.
