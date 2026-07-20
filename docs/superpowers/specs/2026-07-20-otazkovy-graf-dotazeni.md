# Otázkový graf — dotažení konceptu: čtyři měřené experimenty (#57 pokračování)

> Zadání (user, brainstorm 2026-07-20 odpoledne): „dokázal bys koncept
> otázkového grafu vylepšit… otázkový graf, faktový graf, nabízí se
> i odpovědní graf." Rozhodnutí brainstormu: **odpovědní graf ZAPARKOVÁN**
> (až přijde, bude to efemérní reifikace tahu — zrcadlo větného grafu na
> výstupní straně; projekce do event logu #47 jako pozdější zlepšení).
> Prioritou je dotáhnout koncept otázkového grafu: vylepšení a **DRY vůči
> znalostnímu (faktovému) grafu**. Rozsah určují TESTY A MĚŘENÍ (zadání
> user): co neprokáže zlepšení nad základ, se nepřijímá.

---

## 1. Kontext: čtyři grafy (širší obraz, zčásti zaparkovaný)

Brainstorm potvrdil symetrii, do které koncept #57 zapadá:

| | efemérní (jeden tah) | trvalý |
|---|---|---|
| **vstup** | větný graf (tokeny + hypotézy + nároky) | otázkový graf (kompilát karet + schématu) |
| **výstup** | *odpovědní graf (ZAPARKOVÁN)* | faktový graf (znalost + aktivace) |

Dialog jako cyklus: větný graf osvětlí otázkový → vítěz + dekorace →
dotaz do faktového → výsledek → text → neosvětlené role a alternativy
= hrany zpět do otázkového grafu → další tah.

Odpovědní graf dnes existuje ve střepech (`graph_answerer.py:108–113`
`pick_focus`, `_last_values`, `_prev_trace`; `state.py` `PendingFocus`;
`last_resolution`/`last_overflow` čtené automatem, `automaton.py:321–340`)
— jeho reifikace je odložena. Až přijde, `DialogPosition` se zúplní na
obě strany (uzel otázkového grafu × odpovědní graf minulého tahu).
Tento dokument se odpovědního grafu dál nedotýká.

## 2. Mapa DRY nálezů (základ brainstormu)

**A. Klony karet podél skrytých dimenzí.** 19 dotazových karet, kostra
většiny táž: `TAZACI [?SE] PREDIKÁT ÚČASTNÍCI`. Dimenze čas × elipse ×
osoba plodí klony: q-otaz-minuly / -prezens / -minuly-prodrop /
q-kratke-sloveso; q-zjistovaci-minuly / -prezens; q-vyberova-minuly /
-prodrop. Asymetrie prozrazují ruční enumeraci: kombinace
„prézens-prodrop" („Kde bydlí?") chybí — nikdo ji nenapsal.

**B. Dekorace zapečené jako celé karty.** `q-rekl-adresatovi` je celá
karta jen kvůli dativu, ač dekorační model (spec #57, T3) říká, že
`role:adresat` se věší, nesoutěží. Totéž `q-kdo-sloveso-misto` (místní
předložka = Topos) a `q-prvni-osoba-minuly` (1. osoba = Mnemos).
Zásada #55 „pád účastníka nese roli" je obecná — dnes zapečená v klonech.

**C. Schéma predikátů se nese, ale nečte.** `QGraph.predicates`
(2 722 predikátů) se v `compile_qgraph` uloží a nikdy nepoužije.
Faktový graf ví, že `napsat` má subj:person a obj:dílo — otázkový graf
se ho neptá, takže nerozliší smysluplné „Kdo napsal X?" od prázdného
„Kde napsal X?".

**D. Hrany jsou kartézský součin.** Každá otázka → každý clarify uzel
(`qgraph.py:86`). Spec T4 přitom říká: hrany se mají odvozovat.

**E. Workeři vyjmenovaní ručně.** `compile_qgraph` staví tři worker
uzly natvrdo a `illuminate` volá experty přímo — duplicitně s pořadím
bran v `turn()`. Formální `claim()` (#26) je chybějící jazyk.

**F. Dědění dvakrát.** „Hrany místo klonů" (varianty v otázkovém grafu)
a oceán vrstev #41 (dědění po druh-hranách ve faktovém) jsou týž
mechanismus — defeasible inheritance s útlumem. Nabízí se sdílený
walker (výhled, viz §7).

## 3. Rámec experimentů (princip rozsahu)

**Základ** = stav větve `otazkovy-graf`: kompilát 26 uzlů, shadow
harness `benchmark/run_qgraph.py` se čtyřmi rovinami na 100 %
(dispatch dialog 11/11, etalon 41/41, stav 3/3, dekorace 41/41).

Každé vylepšení je samostatný experiment se třemi povinnými částmi:

1. **Hypotéza** — co se zlepší a proč;
2. **Měření** — nové gap řádky / scénáře NAPŘED (červené), plus parita
   základu;
3. **Kritérium přijetí** — parita drží (žádná rovina ani benchmark pod
   100 %) **a** aspoň jeden nový řádek zezelená.

Co kritérium nesplní, se nepřijímá — zůstane zapsaným nálezem (vzor:
váhy z telemetrie, změřená mrtvá větev). Žádný experiment nemění
matcher ani nezavádí interpret do JSONu (zákaz ATN trvá; dimenze jsou
enumerativní osy, ne program).

## 4. Experimenty

### E1 — Rodinné karty s dimenzemi (řeší A+B)

**Hypotéza:** 19 karet lze vyjádřit ~8 rodinami; karta rodiny deklaruje
kostru + dimenze (čas: minulý/prézens/krátké; elipse účastníka; osoba)
a kompilace decku ji deterministicky rozvine na konkrétní vzory —
enumerace zůstává (ukotvený match, pasti 9–11 drží), `teach` se píše
jednou per rodina. Dekorační prvky vzoru (`%{DATIV}` → role:adresat,
`%{PREDL_MISTA}` → topos:oblast, `%{PRVNI_OSOBA}` → mnemos:prvni-osoba)
přestanou plodit klony: karta je deklaruje jako volitelné prvky nesoucí
dekoraci. Chybějící kombinace (prézens-prodrop) vyplynou z rozvinutí.

**Měření:** (a) parita rozvinutí — množina rozvinutých vzorů je
NADMNOŽINOU dnešních vzorů 19 karet (nic se neztrácí; nové kombinace
jsou právě zisk) a dispatch na stávajících benchmarcích se nemění;
(b) nové gap řádky etalonu na tvary, které dnes karta nechytá
(kandidát: „Kde bydlí?" — prézens-prodrop; další z rozvinutí).

**Kritérium:** počet zdrojových karet klesne, ≥1 gap FIXED, benchmarky
i harness 100 %.

**Rizika:** kolize klíče decku (priorita × délka vzoru) mezi rozvinutými
vzory — klíč musí zůstat deterministický (harness počítá remízy dál);
dimenze nesmí přerůst v podmínky (jen osy enumerace).

**Empirie E1 (2026-07-20 večer, commity f840ffa–f49f34f) — PŘIJATO:**
zdrojové dotazové soubory **19 → 16** (rodiny `q-otaz.json`
a `q-zjistovaci.json` nahradily 5 klonů), rozvinutých karet 20
(+1 generovaná kombinace `q-otaz-prezens-prodrop`, ručně nikdy
nenapsaná). Gap scénář `kde-bydli-prodrop-prezens` („Kde bydlí?“ po
zápisu) byl červený (poctivé nenašel — nechytala ho ani poziční
šablona) a rozvinutím **GAP-FIXED** („Kde bydlí?“ → Petrovice).
Parita úplná: 568 testů, etalon 31/31, focus 12/12, dialog 45/45
(GAP 3/0), zápis 34/34, harness obě varianty 100 % (27 uzlů,
0 remíz). Korekční nálezy: (a) `q-vyberova-prodrop` NENÍ klon
`q-vyberova-minuly` — implicitní spona „být“ literálem je jiná
sémantika, do rodiny nepatří; (b) svinutí dekoračních karet
(q-rekl-adresatovi, q-kdo-sloveso-misto, q-prvni-osoba-minuly) se
odkládá — jejich `action` se liší strukturně (role theme,
user_subject) a osa s přepisem akce by byla program v datech (riziko
ATN); patří k úvaze E3/E4; (c) kartové odpovědní tahy jsou v dialog
rovině harnessu „mimo rozsah“ odjakživa (`used.patterns` plní jen
dialogové karty; dotazové karty měří etalonová rovina přes
`last_query_card`) — dispatch nové karty proto dokládá run_dialog
GAP-FIXED, ne shadow dialog rovina.

### E2 — claim() workerů (#26 srůst, řeší E)

**Hypotéza:** experti deklarují nároky formálním `claim()` rozhraním
(zárodky už existují: `chronos.claim_words`, `metron.compute`,
`focus_query_phrases`); kompilace staví worker uzly z registru claimů
— konec ručního výčtu v `compile_qgraph` i přímých volání v
`illuminate`.

**Měření:** dispatch 11/11 + 41/41 beze změny; test rozšiřitelnosti:
fiktivní expert registrovaný v testu → worker uzel vznikne a osvětlení
ho zvedne bez zásahu do compile/turn.

**Kritérium:** parita + test rozšiřitelnosti zelený. Splněné E2 je
brána budoucího přepnutí dispatch (samostatné rozhodnutí, bitová parita
dialog benchmarku).

**Empirie E2 (2026-07-20 večer, commity 992240d–7b1fda5) — PŘIJATO:**
registr `jellyai/iris/claims.py` (`ExpertClaim` + `default_claims()`);
`compile_qgraph` staví worker uzly z registru a `illuminate` volá
rozpoznávače jednotně — smazán ruční výčet tří expertů i přímé importy
(`compute`, `clock_answer`, `deaccent` + focus fráze). Parita úplná:
569 testů, benchmarky i harness obě varianty 100 % beze změny čísel
(tier klíče 3/2/1/0 zachovány). Test rozšiřitelnosti zelený: fiktivní
expert přidaný JEN claimem dostal worker uzel a osvětlení ho zvedlo
bez zásahu do compile/illuminate/turn. Brána budoucího přepnutí
dispatch je otevřena — samo přepnutí zůstává SAMOSTATNÉ rozhodnutí
(§6) s bitovou paritou dialog benchmarku.

**Empirie D (2026-07-20 večer, větev qgraph-dispatch) — PŘEPNUTO:**
brány přímých expertů (Metron → Chronos hodiny → meta-focus) v
`_turn()` nahrazeny smyčkou nad `qgraph.claims` seřazenou prioritou
claimů — pořadí bran je poprvé v DATECH; automat drží kompilát
(`IrisAutomaton.qgraph`). Neznámý worker propadá (test vetřelce).
Meta-focus se posunul před resume identity/pick — dialog benchmark
bitově beze změny (45/45), takže posun je neutrální. Parita úplná:
571 testů, všechny benchmarky i harness 100 %. Výroková polovina a
příkazy zůstávají ručnímu větvení — to je etapa #51, ne fáze D.

### E3 — Instance ze schématu predikátů (řeší C)

**Hypotéza:** schéma se rozšíří z množiny jmen na podpisy s rolemi
(čtou se z existujících faktů při kompilaci — žádná kurátorovaná
sémantika); instance (rodina × predikát) se materializují líně při
osvětlení; instance bez role díry nesvítí → místo marného hledání
chytrá clarifikace šablonou („o napsat vím kdo a co, kde nevím" —
jazyk jako data, text v cs.json).

**Měření:** parita 41/41; nové reject/gap řádky nesmyslných děr
(„Kde napsal KČ?"); dialog scénář chytré clarifikace.

**Kritérium:** nové řádky zelené; NESMÍ vzniknout tvrdé odmítání —
dialog > figly (vysvětlený terminál, ne zamítnutí; predikáty naučené
za běhu Mnemosem musí instance dostat také — slovník roste).

**Rizika:** kombinatorika (materializovat jen navštívené); čerstvě
naučený predikát nesmí propadnout (instanční vrstva musí vidět
aktuální slovník, ne jen kompilát).

**Empirie E3 (2026-07-20 večer, větev qgraph-dispatch) — PŘIJATO:**
`FactGraph.predicate_roles()` čte schéma ŽIVĚ z faktů (Mnemos
predikáty instance vidí), `instance_lit()` v qgraph.py dává verdikt
s vakuovým guardem (neznámý predikát/prázdné role = None, nesoudí se
— past 2). „Kde napsal R.U.R.?" → „O ději „napsat“ vím co, kdo; kde
nevím." (šablona `empty_role_answer` + `role_labels` v cs.json).
Oba gap řádky FIXED (etalon GAP 6/3, dialog GAP 4/0), parita úplná
(572 testů, harness 42/42 obě varianty). NÁLEZ: první realizace jako
KARTA v automatu prošla jen dialog benchmarkem — etalon měří čistý
answerer; verdikt je odpovědní pravidlo, ne dialogový akt → přesun do
answereru (šablona = data). Druhý nález: automat čte trace None jako
miss a přebíjel verdikt kartou assurance-fail — příznak
`last_empty_role` říká „jistota, ne nejistota“ a dialogové karty se
obcházejí.

### E4 — Odvozené hrany (řeší D; závisí na E3)

**Hypotéza:** zpřesňovací hrany se odvodí z typu aktu (overflow jen
u výčtových děr, identita jen u osob — tabulkou, ne kódem) a digging
hrany z rolí schématu predikátu (drill na time/loc jen kde fakty
predikátu role mívají). Kartézský součin zmizí. Nová schopnost:
proaktivní nabídka po odpovědi („mohu doplnit kde") kartou —
dialogový akt; nabídka, ne vnucování.

**Měření:** stav 3/3 s odvozenými hranami (parita); nové dialog
scénáře nabídek; počet hran jako diagnostika struktury.

**Kritérium:** parita + ≥1 scénář nabídky zelený.

## 5. Pořadí a závislosti

E1 a E2 jsou nezávislé (libovolné pořadí). E3 vyžaduje rozšíření
schématu o role. E4 staví na E3. Každý experiment se commituje
samostatně s měřením v commit zprávě; nepřijatý experiment se commitne
jako nález do spec (sekce Empirie), kód se neponechává.

## 6. Co se vědomě nedělá

- **Odpovědní graf** — zaparkován (viz hlavička; efemérní reifikace
  jako budoucí směr, projekce do event logu #47 jako zlepšení poté);
- **výroková polovina** (T6) — jiný graf, nekřížit předčasně;
- **váhy v dispatch** — mrtvá větev (0 remíz), harness je dál počítá;
- **přepnutí dispatch** — ✅ přímí experti přepnuti (fáze D, Empirie
  výše); zbývá výroková polovina a příkazy (#51 etapa 3).

## 7. DRY účet a výhledy

Po přijetí E1–E4 platí: sémantika děr se čte ze schématu faktového
grafu (ne z rukou), digging hrany z rolí faktů (ne z enumerace),
povrch češtiny žije v rodinných kartách (ne v klonech), nároky
v claim() deklaracích (ne v pořadí větví). Otázky zůstávají stínem
schématu faktů — DRY obou grafů je architektonický princip, ne úklid.

Výhledy (zapsané, neimplementované): sdílený walker dědění (varianty
rodin v otázkovém grafu ↔ oceán vrstev #41 ve faktovém — týž
mechanismus defeasible inheritance); efemérní odpovědní graf a jeho
projekce do event logu (#47); vícejazyčnost (uzly jazykově neutrální,
povrch = rodinné karty per jazyk — nový jazyk nemění graf).
