# Design V4a: bohatá analýza otázky + sponové odpovědi

**Datum:** 2026-07-16
**Autor:** Jindřich Němec + Claude
**Status:** Schváleno (fáze A; fáze B roadmapa)

## 1. Cíl a kontext

V3 (pravidlový answerer) odpovídá na Kdo/Co/Kde/Kdy/Kolik přes entity a role, ale:
- rozpoznává **tázací slova jen podle přesného tvaru** → „Jaký/Jaká/Jaké" i jiné
  formy padají na extraktivní;
- neumí **definiční a predikátové otázky** („Kdo je X?" → tautologie „Rossum";
  „kdo byl bratrem X?" → vybere podmět).

Fáze A je řeší **jednou bohatou analýzou otázky** (společný předek pro celý další
rozvoj) + **strategií sponového přísudku**. Je to zároveň základ pro fázi B
(konverzační pseudo-attention), která ten samý rozbor otázky použije k „zahřívání"
kontextu.

## 2. Klíčová rozhodnutí

| Oblast | Rozhodnutí |
|---|---|
| Rozpoznání typu otázky | **přes lemma** (jaký/který/čí + varianty), ne přes tvar |
| Analýza | jeden `QuestionAnalysis` (typ, sloveso, spona?, témata) — sdílený předek |
| Nová strategie | **sponový přísudek**: „X je/byl Y" → odpověď Y (nominální/adjektivní) |
| Nové typy | Jaký/Který (přísudek = vlastnost), Čí (osoba, přivlastnění) |
| Drobnost | filtr zájmen/spojek u „Co?" (aby nevzniklo „který") |
| Integrace | rozšíří `selection` + `TemplateAnswerer`; anotace se nemění (parse už je) |

## 3. Bohatá analýza otázky (`jellyai/answerer/question.py`)

`analyze_question(question, client) -> QuestionAnalysis`:
- Parse otázky přes UDPipe službu.
- **Typ přes lemma**: mapa lemma→typ — `kdo→Kdo`, `co→Co`, `kde/kam/odkud→Kde`,
  `kdy→Kdy`, `kolik→Kolik`, `jaký→Jaký`, `který→Který`, `čí→Čí`. Lemmatizace pokryje
  všechny tvary (Jaká/Jaké/Jakého → „jaký").
- **Sloveso**: lemma kořenového slovesa (pro výběr role).
- **Spona**: je otázka sponová? (kořen má `cop` potomka nebo hlavní sloveso je „být").
- **Témata**: obsahová lemmata (bez stopslov) — pro fázi B (zahřívání); ve fázi A
  se počítají, ale nepoužijí.

`QuestionAnalysis(qtype, verb_lemma, is_copula, topic_terms)`.

## 4. Sponová/predikátová strategie (`selection.py`)

V UD je „Božena Němcová byla spisovatelka" strukturováno: **přísudek** „spisovatelka"
je kořen, „byla" je `cop`, „Němcová" je `nsubj`. Odpověď na definiční/vlastnostní
otázku = ten přísudek (+ jeho přívlastky, např. „významná spisovatelka").

`select_predicate(annotation) -> Candidate|None`:
- Najdi kořen věty, který má potomka `cop` (sponu „být") — to je nominální/adjektivní
  přísudek.
- Vezmi přísudek + jeho přímé přívlastky (`amod`, `flat`) jako odpovědní frázi.
- Vrať kandidáta (form = fráze, lemma = nominativ přes lemmata).

**Dispatch** v `select_answer`: použij sponovou strategii, když
`is_copula` nebo `qtype in {Jaký, Který}`; jinak dosavadní entita-podle-role.
Pro „Co?" navíc **přeskoč zájmena/spojky** (UPOS PRON/DET/CCONJ) při výběru předmětu.

**Rozsah:** přímý případ „X je/byl Y → Y" (definice, vlastnost). Inverzní
„kdo byl bratrem X?" (najdi osobu v relaci) je těžší — fáze A ho zkusí přes
přísudek, ale garantuje jen přímý případ; zbytek fallback extraktivní.

## 5. Šablony (`templates.py`)
Přidat `Jaký`, `Který` (frame `{answer}`, přísudek — pro adjektiva ponechat tvar,
pro podstatná jména nominativ) a `Čí` (osoba, nominativ). Ostatní beze změny.

## 6. Tok dat (beze změny principu)

```
otázka → analyze_question (UDPipe): typ + sloveso + spona? + témata
retrieval → pasáž → anotace → select_answer:
   spona/Jaký? → přísudek   |   jinak → entita podle typu+role
   → nominativ (MorphoDiTa) → šablona → odpověď   (fallback extraktivní)
```

## 7. Ošetření chyb a hraniční případy
- Neznámé tázací slovo → fallback extraktivní (dnešek).
- Sponová věta bez jasného přísudku → fallback.
- „Jaký" bez adjektivního přísudku v pasáži → fallback.
- Anotace se nemění (dependency parse už obsahuje `cop`/`deprel`).

## 8. Testování (pytest, hermetické)
- **analyze_question:** „Jaká byla Němcová?" (přes FakeUfalClient parse) → typ „Jaký",
  is_copula True, sloveso „být". Varianty tvarů → stejný typ.
- **select_predicate:** anotace se sponou („Němcová byla významná spisovatelka") →
  přísudek „významná spisovatelka".
- **Co filtr:** předmět-zájmeno se přeskočí, vybere podstatné jméno.
- **TemplateAnswerer:** „kdo je X?" → definice (přísudek), ne tautologie; „Jaký je X?"
  → vlastnost; fallback když nic.

## 9. Fáze B — roadmapa (další spec + plán)

Konverzační **pseudo-attention** nad retrievalem:
- `ConversationContext`: `pojem → aktivace`, zahřívání z `topic_terms`, **dohasínání**
  (× faktor) každé kolo; při malém překryvu s horkým stavem silnější útlum.
- **Předehřátý retrieval**: skóre = `BM25(dotaz) + λ · match(pasáž, horký stav)`.
- **Vzdálenostní jádro místo tvrdých bloků** *(uživatelovo klíčové rozhodnutí)*:
  nalezená věta vyzařuje skóre do okolních vět s **útlumem podle vzdálenosti**
  (sever/jih), aby systém nezávisel na kvalitě formátování odstavců. **Soubor**
  zůstává spolehlivý hrubý předěl; uvnitř dokumentu se počítá spojitá vzdálenost.
- Zapojení do `repl` (stavové napříč dotazy); jednorázový `ask` netěží.

## 10. Mimo rozsah (YAGNI, fáze A)
- Konverzační aktivace a hierarchie/vzdálenost (to je fáze B).
- Otázky „Proč/Jak" (vedlejší věta příčiny/způsobu).
- OOV skloňování místních jmen (limit slovníku MorphoDiTy).
