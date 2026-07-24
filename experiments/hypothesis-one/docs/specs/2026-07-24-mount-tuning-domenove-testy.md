# Ladění mountu (entity routing) + doménové testy

*2026-07-24, větev `mount-tuning` z `hypothesis-one`*

## Kontext

Commit `32e20e2` zapojil **líný entity routing** (`fact_store.route_docs` — dotaz
entita × predikát × odpovědní role vybere pár dokumentů z hlaviček fact shardů, které
se mountnou běžnou cestou). Byl commitnut se `entity_retrieval.enabled=true`, ale
**neměřen proti etalonu** (velký etalon vyžaduje UDPipe).

Nález při měření: malý `eval_answers.py` routing VŮBEC netestuje (jeho `stage_trace`
jde napřímo `select_files`, ne `route_docs`). Routing se cvičí jen přes `a.answer()`,
což dělá `eval_large.py`.

## Část A — ladění mountu

### Změřený problém

Velký etalon `eval_large` (145 otázek), routing ON vs OFF:

| mode | skóre | Seifert (narodil/Žižkov) | Kundera (otec) |
|------|-------|--------------------------|----------------|
| OFF (tf·idf loader) | 63/145 | ✓ answer | ✓ answer |
| **replace** (commit 32e20e2) | **61/145** | ✗ clarify | ✗ clarify |
| union | 63/145 | ✓ | ✓ |
| union_cap | 63/145 | ✓ | ✓ |

Příčina: `answering.answer()` skládal mount jako `hot = routed or select_files(lemmas)` —
routing **nahradil** lexikální mount. Užší sada dokumentů vyhladověla aktivační pole →
assurance spadla `answer → clarify`, aniž se změnila identita vítěze (Seifert: odpověď
Žižkov je STEJNÁ, jen shozená na doptání). Routing tak −2 bez jediného zisku (korpus
nemá kolize, kde by strukturální přesnost routingu předčila lexikální výběr).

### Řešení

`answering._compose_mount(routed, lexical)` — kompozice řízená configem
(`entity_retrieval.mode`), nic natvrdo:

- **replace** — routed nahradí lexikální (původní; riskuje vyhladovění pole).
- **union** — routed má prioritu, lexikální DOPLNÍ (důkaz jistý + pole zdravé). *Default.*
- **union_cap** — union oříznutý na `max(|routed|, |lexical|)` (nepřemountuje).

Interim default = `union` (bezpečné na velkém etalonu). **Napětí**: `replace` mountuje
čistě jen routed doc (dobré pro kolize, špatné proti vyhladovění), `union` vrací
lexikální rivaly (dobré proti regresi, u kolizí vrací špatného rivala). Který mode je
skutečně nejlepší, rozhodne až Část B (collision testy) — teprve tam má routing co
ukázat.

## Část B — doménové testy (`gold_domain.json` + `eval_domain.py`)

Korpus vyrostl (52 fact shardů, 16k faktů: 22 wiki + ~29 biblických knih). To umožňuje
testy, které CÍLENĚ stresují výběr souboru.

### Kategorie `collision` (routing scoreboard)

Lexikálně matoucí otázky, kde `select_files` může vybrat špatný soubor a routing má
vyhrát. Kromě odpovědi se ověřuje i **vítězný soubor** (`expect_doc`) — jádro „retrieval
po uzlech" je vybrat SPRÁVNÝ soubor mezi kolidujícími kandidáty. Zdroje kolizí:

- **bratři Čapkové** — Josef (1887 Hronov, †1945 Bergen-Belsen, malíř) × Karel (1890
  Malé Svatoňovice, †1938 Praha, spisovatel). Sdílené příjmení, obě data v obou článcích.
- **autor × dílo** — Bílá nemoc / R.U.R. / Válka s Mloky → Karel Čapek (past: bratr Josef).
- **sdílené křestní jméno** — „Kdy se narodil Čapek/Jaroslav?" → doptání (ambiguita).

### Kategorie `coverage` (per-entita regresní pokrytí)

Systematicky pro 22 wiki entit: rok narození/úmrtí, místo, dílo, povolání, taxonomie.
Ground truth **ověřen ručně proti raw textu** (`data/raw/wiki_*.txt`), ne proti extrakci
→ netestuje se extrakce sama sebou.

### Harness `eval_domain.py`

- `python3 eval_domain.py` — současný config, plný rozpad + propady (ANS/DOC).
- `python3 eval_domain.py --sweep` — porovná off/replace/union/union_cap (parse-cached),
  vypíše, KDE se módy liší → rozhodovací materiál pro finální `mode`.

Každý běh navíc zapíše **HTML scoreboard** do `docs/last-test.html` (modul `test_report.py`,
`write_scoreboard` — offline, laděné s `docs/style.css`, světlé i tmavé téma): velké skóre,
skóre po doménách (meter bary), a per-doména tabulky *otázka → odpověď systému → očekávaná
odpověď → mód* (+ vítězný soubor u collision). Reprezentativní vzorek (`sample=N` = všechny
propady + N passů) nebo vše (default). Stránka je odkázaná z `docs/index.html`, gitignored
(generovaný snímek, jako `baseline_large.json`).

Skórování mode-aware (jako eval_large): honest-negative/unsure PASS když NEodpoví
sebevědomě; clarify PASS když se doptá; answer PASS když answer + odpověď sedí; u
collision navíc `expect_doc`.

## Postup

1. ✅ `_compose_mount` + config `mode`, změřeno (union 63/145, parita s OFF).
2. ✅ `eval_domain.py` harness (single + sweep, expect_doc).
3. ✅ `gold_domain.json` — 12 collision + 59 coverage = 71 položek (builder z ověřených faktů).
4. ✅ `--sweep` nad doménovým etalonem → finální `mode = union`.
5. ✅ Regrese: malý etalon 18/25 beze změny, offline testy 6/6.

## Výsledky (měřeno)

`eval_domain.py --sweep`, 71 položek:

| mode | skóre | collision (odpověď/soubor) | liší se od OFF |
|------|-------|----------------------------|----------------|
| off | 65/71 (91 %) | 12/12 · 10/10 | — |
| replace | 64/71 | 12/12 · 10/10 | −1 (Seifert) |
| **union** | **65/71** | 12/12 · 10/10 | 0 |
| union_cap | 65/71 | 12/12 · 10/10 | 0 |

**Poctivý závěr o routingu:** na celém korpusu se módy liší v JEDINÉ položce (Seifert,
kde `replace` shodí answer→clarify). **Routing nepřinesl zisk ani na kolizích** — i bez
něj lexikální `select_files` vybere správného bratra (křestní jméno v dotazu už silně
váží správný soubor: „Josef Čapek" → wiki_josef_čapek). Kolize, jak jsou tu postavené,
nejsou pro lexikální výběr dost těžké. Routing je tedy nyní **neškodná pojistka** (union
= parita s OFF), ne zdroj přesnosti. Jeho cena („retrieval po uzlech", Díl VIII) je sázka
na budoucí měřítko/typy dotazů, kde lexikální výběr selže (např. dotaz bez entity v textu,
relace, kde odpovědní soubor lexikálně nematchuje) — ne na dnešním etalonu.

Rozhodnutí: `entity_retrieval.mode = union` (default). Splňuje „nechat zapnuté" (user)
bez regrese. `replace`/`union_cap` ponechány jako měřené alternativy v configu.

### Mezery odhalené doménovými testy (job scoreboardu)

- authorship-who 1/3 — „Kdo napsal Válku s Mloky/Bílou nemoc?" → Hašek/Vančura (dílo
  neprovázané s Čapkem; jen R.U.R. projde).
- taxonomy 5/6 — „Co je kůň domácí?" → „zvíře" místo „lichokopytník" (obecné slovo přebíjí taxon).
- temporal — „Kdy se narodil Bohumil Hrabal?" → 1893 (špatný rok); Nezval → 1900 (SPRÁVNĚ,
  ale shozeno na clarify = kalibrace assurance).
- spatial — „Kde se narodil Vladislav Vančura?" → „Slezsko" místo „Háj ve Slezsku" (částečná extrakce).

## Měřítko úspěchu

- ✅ Velký etalon: routing NEregreduje (union 63 = OFF baseline; replace bylo 61).
- ✅ Doménové testy: ground truth ověřený proti zdroji (raw wiki), ruční kurace,
  scoreboard 65/71 s konkrétními zdokumentovanými mezerami.
- ⚠️ Zisk routingu nad OFF se NEPROKÁZAL — kolize nejsou pro lexikální výběr dost těžké.
  Budoucí směr: testy, kde odpovědní soubor lexikálně NEMATCHUJE dotaz.
