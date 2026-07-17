# Jmenný uzel × instance — jméno není entita (backlog #8)

## Koncept (zadavatel)

Jmenovka je štítek; entita je INSTANCE doložená svým okolím. Jmenný uzel +
instanční uzly spojené hranou „jmenuje se"; instance per provenienční
kontext (dokument/odstavec); srůstání jen při překryvu kontextových otisků.
Toyota = druhový uzel, vlastnosti VŽDY na instanci. Dotaz na jmenovku →
focus-offer instancí (dialogový mechanismus hotový).

## Klíčové měření (2026-07-18, falzifikace naivního srůstu)

Kontextový otisk (sdílení sousedů grafu) sám o sobě identitu NEROZLIŠÍ:

| pár | překryv | pokrytí fragmentu | skutečnost |
|-----|---------|-------------------|------------|
| Ježíš × Ježíš Nazaretský | 197 | 0.31 | táž osoba |
| Jan × Jan Křtitel | 29 | 0.28 | DVĚ osoby |
| Němce × Božena Němcová | 7 | 0.70 | manželé (!) |
| K. A. Čapek × Josef Čapek | 35 | — | bratři (!) |

Postavy jednoho příběhu sdílejí svět, ať jsou totožné, nebo ne. Rodinní
příslušníci sdílejí kmen jména I svět. Rodový odhad na skloněných tvarech
lže („Ježíše Krista" → Fem). ⇒ Slepé statistické slučování je vyloučeno.

## Fáze 1 — HOTOVO (commit této větve)

* **Extrakce `jmenovat`**: „X řečený/zvaný Y" (Mt 1,16 „Ježíš řečený
  Kristus") → fakt `jmenovat(subj=X, pred=Y:jméno)`. Těsné sousedství
  (nositel ≤3 vlevo, alias ≤2 vpravo) — výčtové věty nepárují vzdálená
  jména. Tabulky `alias_participles`, `name_predicate` v cs.json.
* **`instance.resolve_instances`**: srůst VÝHRADNĚ z textového tvrzení —
  střep, jehož všechna slova jsou kmenově slučitelná se jmény nositele ∪
  tvrzené aliasy, A jehož otisk sdílí ≥3 sousedy (korpus tvrzení
  potvrzuje), se přemapuje na kanon (nejkratší jméno). Výsledek: „Ježíše
  Krista" → „Ježíš"; Nazaretský čeká (bez tvrzení — poctivě odděleno,
  dialog nabízí).
* **`graph.name_families`**: kmen jména → osobní uzly (169 rodin) —
  podklad dialogové nabídky a budoucí jmenovky.
* **Identitní fallback**: díra pred/attr čte po `druh` i `jmenovat`
  (jméno je odpověď na „kdo je X").
* Sdílený `remap_nodes` (kanonizace i instance); fakty s jediným
  účastníkem po srůstu padají.

## Fáze 2 — OTEVŘENÁ

1. **Instance per odstavec**: uzly vznikají per provenienční okno
   (dokument/odstavec) a srůstají otiskem — řeší smíšené holé uzly („Jan"
   = evangelista ∪ Křtitel už při buildu, dnes neoddělitelné post-hoc).
2. **Rozpuštění dvou-osobových slepenců**: uzel, jehož kmeny pokrývají ≥2
   nezávislé osoby s vysokým překryvem k oběma („Áronovi Mojžíš"), se
   rozpustí do silnější komponenty.
3. **Jmenovka jako uzel**: typ `jméno`, hrana `jmenovat` ke každé
   instanci; dotaz na nejednoznačnou jmenovku → focus-offer rodiny.
4. **Kanonická id nominativem** (#1v2): „Šimona" → „Šimon" (dnes kanon
   nejkratším členem — genitivy v id jsou kosmetický dluh).
5. Toyota/druhové instance (vlastnosti na instanci) — až s Mnemos výukou
   pojmů (#7).
