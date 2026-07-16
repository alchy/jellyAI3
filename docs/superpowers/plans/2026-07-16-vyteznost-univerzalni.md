# Výtěžnost: univerzální reifikace + normalizace — Implementation Plan

> ✅ **DOKONČENO (2026-07-16).** Výsledek: etalon **15/15 (100 %)** + GAP-FIXED
> „Kdo napsal R.U.R.?" i „Kdo napsal Válku s mloky?"; vlajková rekurze „Kdo byl
> bratr autora, který napsal R.U.R.?" → Josef Čapek je normativní případ.
> Odchylka od plánu (pokyn uživatele — univerzální datový model, ne SELECT
> vzory): místo worklist extraktoru vznikla **kontextová asociace** (role ③
> aktivačního pole — fakt `kontext` váže entity věty na aktuální subjekt
> dokumentu) + **predikát jako preference** (přesný fakt → asociační patro)
> + **enumerativní odpovědi** (_match vrací všechny rovnocenné díry). Navíc:
> overtní zájmenný podmět blokuje pro-drop (čistí šum sponových vět).

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Navazuje na `2026-07-16-entity-resolution.md` (blocker vyřešen, etalon 12/13).

**Goal:** Maximalizovat výtěžnost z textu univerzálními mechanismy (žádné téma-specifické ohýbání — korpus se bude zásadně rozšiřovat): (B) reifikace libovolného sponového genitivu, (C) normalizace tečkovaných zkratek patternem, (D) poziční slučování jmenných fragmentů. Měřicí instance: rekurzivní dotaz „Kdo byl bratr autora, který napsal R.U.R.?".

**Architecture:** Jazykové tabulky konsolidované v `jellyai/lang` (jediný zdroj, aktivní jazyk = stav modulu). Extrakce reifikuje sponový genitiv bez slovníku. Normalizace tokenizace žije v `UfalClient` (jediný chokepoint — korpus i otázky). Resolver dostane druhý, poziční merge-pass.

**Tech Stack:** `.venv/bin/python`, pytest, pylint, `benchmark/run_etalon.py`.

## Global Constraints

- **Etalon nikdy pod 12/13**; každý task končí měřením. Cíl: gap „Kdo napsal R.U.R.?" → GAP-FIXED + nový core case flagship rekurze.
- **Univerzálnost**: žádný hardcode konkrétních jmen/titulů; tabulky jen jako jazyková data (`jellyai/lang/cs.json`).
- Determinismus; refactoring bez zpětné kompatibility je dovolen.
- Práce na větvi `vyteznost-univerzalni`.

---

### Task A: `jellyai/lang` = jediný zdroj jazykových tabulek

**Files:** Modify `jellyai/lang/__init__.py` (aktivní jazyk: `set_language`/`current`), `jellyai/lang/cs.json` (+`work_nouns`, `work_verbs`), `jellyai/graph/canon.py` (deleguje na `lang.current()`, vlastní `set_language` zmizí), `jellyai/graph/spread.py` + `jellyai/graph/recover.py` (čtou tabulky z `lang`), `jellyai/tasks.py` + `tests/test_lang.py` (import z `jellyai.lang`).

- [ ] Failing test: `lang.current()` vrací cs pravidla; `current()["work_nouns"]` obsahuje „drama"; po `set_language(<custom>)` se mění i `canon.cluster_key`.
- [ ] Implementace; celá suita + pylint + etalon 12/13 beze změny; commit.

### Task B: Univerzální reifikace sponového genitivu

**Files:** Modify `jellyai/graph/extract.py` (copula větev; `_REL_NOUNS` smazat), Test `tests/test_relations.py`.

Sponová věta „X je Y ⟨osoba-gen⟩" → **vždy** `Y(subj=X, obj=osoba)` (Y = libovolné podstatné jméno hlavy: bratr, drama, román…), **plus** identita `být(X, Y)` (zůstane zodpověditelné „Co je X?"), **plus** pro `Y ∈ lang.work_nouns` autorský fakt `napsat(subj=osoba, obj=X)`.

- [ ] Failing testy: drama-věta (dle reálné anotace R.U.R.) → fakty `drama(X, Karel Čapek)` + `napsat(Karel Čapek, X)` + `být(X, drama)`; stávající bratr/identita testy drží.
- [ ] Implementace; suita + pylint + **rebuild grafu + etalon** (čekám posun gapů: „Kdo napsal Válku s mloky?"); commit.

### Task C: Normalizace tečkovaných zkratek (pattern, ne výčet)

**Files:** Create `jellyai/normalize.py` + `tests/test_normalize.py`, Modify `jellyai/ufal_client.py` (`UfalClient.parse`/`entities` aplikují normalizaci; `FakeUfalClient` ne).

Tokenizace trhá „R.U.R." na `R/./U/./R/.` (lemma dokonce „U"→„United") a NER vrací paskvil `R.U`. Univerzální oprava na jediném chokepointu:

- `merge_abbreviations(sentences)`: běh ≥2 párů ⟨jednopísmenný token⟩⟨.⟩ → jeden token (form=lemma=„R.U.R.", upos PROPN, head/deprel prvního písmene, span přes celý běh); **přemapovat 1-based `head`** všech tokenů věty.
- `expand_abbreviation_entities(text, entities)`: entita, jejíž text je prefixem tečkované zkratky v `text` na téže pozici (regex `(?:\w\.){2,}` s Unicode písmenem), se roztáhne na celou zkratku; duplicitní/vnořené entity uvnitř spanu se zahodí.

- [ ] Failing testy na reálném tvaru (R/./U/./R/. s heady z anotace; entity `R.U`+fragmenty); head remap ověřen.
- [ ] Implementace + zapojení do `UfalClient`; suita + pylint.
- [ ] **Re-anotace korpusu** (`.venv/bin/python cli.py annotate`) + rebuild grafu + etalon — čekám „Kdo napsal R.U.R.?" GAP-FIXED (autorství z Task B + entita z Task C); commit.

### Task D: Poziční slučování jmenných fragmentů (resolver, 2. pass)

**Files:** Modify `jellyai/graph/graph.py` (`resolve_entities`), Test `tests/test_resolve_entities.py`.

Kratší víceslovné jméno se slije do delšího, jen když se kmeny zarovnají na **první (křestní) a poslední (příjmení)** pozici a kandidát je jednoznačný: `Karel Čapek` → `Karel Antonín Čapek` ✓; otec `Antonín Čapek` (křestní sedí na prostřední pozici) zůstává; holá jednoslovná jména se neslučují nikdy.

- [ ] Failing testy: merge KČ→KAČ; otec zůstává; dvojznačný kandidát zůstává; idempotence drží.
- [ ] Implementace (rozšíření `node_map` před přepisem — jeden rewrite); suita + rebuild + etalon; commit.

### Task E: Normativní případy + docs + merge

- [ ] Etalon: + `{"q": "Kdo byl bratr autora, který napsal R.U.R.?", "expect": ["Josef"], "cat": "rekurze"}`; gapy, které začaly procházet, povýšit na core.
- [ ] Finální: suita + pylint + etalon (vše zaznamenat); docs (spec status, memory); merge do `main`; commit.

## Rollback / stopky

Etalon < 12/13 nebo rozbitý existující test → stop a analýza (nikdy neopravovat test podle kódu). Neuniverzální řešení (hardcode tématu) = špatné řešení, i kdyby zvedlo skóre.
