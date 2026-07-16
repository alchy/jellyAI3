# Přesnost odpovědí (arc 4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Podklad: analytický rozbor dialogu z webu (16 otázek) — každý task odkazuje
> na dialogy, které opravuje. Metriky (etalon + coverage) po každém tasku.

**Goal:** Odstranit chybové módy odhalené dialogem: kontext-fallback u identity,
parse-quirk copular otázek, ztracené apoziční tituly, nestabilní/zašuměné
výčty, existenciální pro-drop šum, ano/ne hádání. Subjektivní kvalita se musí
potkat s metrikami — etalon dostane případy z dialogu vč. negativních.

**Baseline:** etalon 15/15 (100 %), GAP 2/3; coverage 31 %; graf 2755/4068;
252 testů.

## Global Constraints

- Etalon nikdy pod 100 % jádra; nové normativní případy se přidávají ve
  stejném tasku jako jejich fix. Universálnost (feats/struktura, žádné výčty).
- Větev `presnost-odpovedi`; commit + metriky po tasku.

---

### Task 1: Identita bez kontext-fallbacku + kanonický copular handler
(dialogy: Prager Morgenpost, Machar, robot ×2, Ludvík Němec, Helena)

- Kontextové patro jen pro díry `subj`/`obj` (identita pred/attr a ano/ne
  otázky s dírou None kontextem NEhádají → poctivé „nenašel").
- Pattern: spona s tázacím kořenem („**Co** je robot?" — parser dá root na
  „Co") → kanonicky `být` + jmenný člen jako known + díra pred/attr.
- `predicate=None` už není wildcard na vnější díře.
- Etalon: `Co je robot?→stroj`; `Kdo je Ludvík Němec?→nenašel` (reject Čapek);
  `Kdo byl Svatopluk Machar?→nenašel` (reject Morgenpost). Runner: pole
  `reject` (must-not-contain).

### Task 2: Apoziční účastník
(dialogy: „Jakou hru napsal KČ?", chudé „Co napsal?")

- „napsal hru **R.U.R.**" → appos dítě předmětu vstupuje do TÉHOŽ faktu jako
  další obj → `napsat(KAČ, hra, R.U.R.)`; „Jakou hru…" najde díru R.U.R.
- Etalon: `Jakou hru napsal Karel Čapek?→R.U.R.`; `Co napsal Karel Čapek?`
  expect + „R.U.R.".

### Task 3: Enumerace — stabilní, se stropem, bez junku
(dialogy: PM výčet 17 hodnot, „Kdo byl Karel?" 8 hodnot, nestabilita hra/válka)

- Tie-grouping podle základního skóre (váha+role/typ) BEZ aktivace; aktivace
  jen řadí uvnitř skupiny. Strop výčtu (5). Filtr jednoznakových hodnot.

### Task 4: Pro-drop rodová shoda + pred ≠ interrogativum
(dialog: „Kdo byl Karel?" — šum válka/obava/jaký)

- Pro-drop dosazení jen při shodě rodu slovesného tvaru (feats Gender) se
  jménem osoby: „**Byla** válka" (Fem/Neut) ≠ Čapek (Masc) → existenciál, ne
  elize. Sponový pred s `PronType=Int/Rel` („jaký") fakt nezakládá.
- Rebuild grafu; etalon: reject na „Kdo je Karel Čapek?" (jaký, obava, válka).

### Task 5: Ano/ne otázky + obl knowns
(dialogy: „Publikoval Karel…?", „Jak to souviselo s KČ?")

- Otázka bez tázacího lemmatu = zjišťovací → kontrola existence faktu
  („Ano…" / fallback), žádné kontextové hádání.
- Pattern sbírá obl/nmod účastníky jako known (role theme) — „s Karlem
  Čapkem". (Demonstrativní kotva „to"/„tomto" a cesta-v-grafu „jak souvisí
  X s Y" = budoucí arc — spreading activation.)

### Task 6: Viz odlišení kontextových hran + uzávěr

- Fakt `kontext` v exportu: label „souvislost", `kind="context"` na uzlu i
  hranách (viz styling — čárkovaně/slabě). Odpovídá dotazu uživatele na hranu
  KAČ—Ludvík Němec.
- Finální metriky (tabulka), docs, memory, merge do main, push, restart webu.

## Rollback / stopky

Etalon pod 100 % jádra nebo rozbitý test → stop a analýza. Ztráta gapů
(R.U.R./Válka GAP-FIXED) = regrese — kontext-gating nesmí zabít subj/obj patro.
