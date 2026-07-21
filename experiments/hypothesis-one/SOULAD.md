# SOULAD — sdílené porozumění (hypothesis-two)

Referenční kotva, aby nedocházelo ke vzájemnému neporozumění. Tři části:
**terminologie** (co které slovo znamená), **fakty** (co reálně máme a s jakými čísly),
**principy** (proč to tak stavíme). Vše ostatní se poměřuje tímto dokumentem.

Souvisí: `docs/superpowers/specs/2026-07-21-hypothesis-one-synteticky-otazkovy-graf-design.md`
(návrhový spec), `STATE.md` (stav validace).

---

## 1. Terminologie

| pojem | význam | kde v kódu |
|---|---|---|
| **SOURCE FACT** (zdrojový fakt) | věta z korpusu; zdroj světla, který vrhá stíny-otázky | `annotations.pkl` |
| **SEKTOR** | jeden slot v okně (token); **ÚFAL ho atribuuje** (upos/lemma/pád/deprel) | `run.slot()` |
| **VZOR** | *single entita (token) + její r=X frame*; match-klíč | `run.frame_sig()` |
| **r = X** (poloměr) | šířka okna rámu; číselník tvrdosti shody | `CFG["radius"]` |
| **OCCURRENCE + FRAME** | index: token = Occurrence (lemma+rysy), Frame = signatura okna | `run.frame_sig` |
| **DÍRA** (hole) | tázaný slot = jedna katalogová role | `roles.decompose` |
| **KATALOG ROLÍ** (větné členy) | univerzální schéma odpovědi (who/where/whom_what/state…); klíče univerzální, popisy = data | `cs.json role_catalog` |
| **OTÁZKA** | kanonický pattern = VZOR **s dírou**; jedna na askable roli (skelet) | `roles` + `role_ask` |
| **DOTAZ** | povrchové formulace jedné OTÁZKY (přirozený jazyk); **generuje Ollama offline** | `ollama_iface.gen_questions` |
| **ODPOVĚĎ** | výplň díry; fragment (slot) NEBO věta (šablona); je to sama fakt (rekurze) | `ollama_iface.gen_answer` |
| **QUERY PATTERN** | frame **s dírou** = match objekt (= OTÁZKA VZOR) | `run.frame_sig` (díra) |
| **ANSWER PATTERN** | frame **s výplní**; QUERY PATTERN se do něj promítne (vyplní díru) | (projekce) |
| **STÍN** | otázka je stín faktu; vždy dopadne na nějaký VZOR-tvar | (princip) |
| **grafikon** | graf VZORŮ; hrany OTÁZKA↔DOTAZ(↔ODPOVĚĎ); hustota řízena r | `bundle.jsonl` |
| **bundle / katalog vazeb** | fakt → { stín per role }ⁿ (deterministicky) | `gen_bundle.py` |
| **kontext patro** | co-occurrence fakty `kontext(nejteplejší, entita)`; odpoví na díru slovesa bez přímého faktu (R.U.R.→Čapek) | orig. `graph.py:297` |
| **TĚŽIŠTĚ / ActivationField** | slábnoucí pole `warm/step/hottest`; drží nejteplejší entitu dokumentu | orig. `graph/activation.py` |
| **pro-drop koreference** | elidovaný podmět kopule doplněn z těžiště → identitní fakt (…byl Bůh → `být(Ježíš,Bůh)`) | orig. `graph/extract.py` |
| **identitní hlas** | hlasování přes `být`/`druh`/`jmenovat` pro „Kdo je X?" | orig. `_identity_vote` |
| **hub-brána** | práh větvení; řídké téma věř, hub (Ježíš 209 sousedů) → ptej se | orig. `context_hub_limit` |
| **source-attention** | dynamické mezitahové pole nad dokumenty; odliší, který korpus je teplý | orig. `_source_bonus`, `doc_links` |
| **etalon** | zlatá míra; každá změna měřena (parita + ≥1 nový řádek, jinak nepřijmout) | (projektový princip) |
| **zákon 3** | jazyk = data (JSON); univerzální klíče v kódu, české hodnoty v `cs.json` | `cs.json` |

**Pozor na dvojici OTÁZKA vs DOTAZ:** OTÁZKA = abstraktní VZOR-s-dírou (jeden), DOTAZ =
konkrétní povrch (mnoho). Kolik DOTAZŮ se slije do jedné OTÁZKY řídí **r** (viz Fakty).

---

## 2. Fakty (co reálně máme)

**Korpus:** 13 dokumentů — bible (genesis, exodus, žalmy, jan, lukáš, matouš) +
wiki (Karel Čapek, Josef Čapek, Božena Němcová, Jan Neruda, Bílá nemoc, R.U.R., Válka s mloky).
Anotace `data/annotations.pkl` (UDPipe, cache).

**Postaveno:**
- `run.py` — occurrence+frame index, `frame_sig` s konfigurovatelným r (r=1 bitově zpětně kompatibilní), tf·idf brána, sdílený `role_key`, viewBase export.
- `roles.py` — rozklad **per klauzule** do katalogu rolí; **#59** meziklauzulová dědičnost místa (relativní věta zdědí místo z řídící). Ověřeno: „Josef a Maria odešli do Jeruzaléma, kde se narodil Ježíš" → „Kdo se narodil v Jeruzalémě?" → `{who: Ježíš, where: Jeruzalém}`.
- `cs.json` — `role_catalog`, `role_ask`, `deprel_to_role`, `role_prepositions`, `clause_markers`, příčestí/relativní příslovce. **Sjednocené klíče** (jeden slovník: UD deprel je vstup, katalog je kanon).
- `gen_registry.py` → `registry.jsonl` — přímá vazba fakt↔synt.otázka (katalogové role).
- `gen_bundle.py` → `bundle.jsonl` — **deterministický** svazek stínů.
- `ollama_iface.py` — `gen_questions` (TAM), `gen_answer` (věta), `verify` (ZPĚT); qwen3.6, `think:false`.
- `gen_variants.py` — dávka parafrází + zpětná verifikace.

**Změřeno:**
- **VZOR / r**: r=1 ~18 členů na rám (zobecnění, mnoho DOTAZŮ→1 OTÁZKA); r=2 stínové dvojice vrcholí; r=3 ~1 člen (konkordance, 1 DOTAZ≈1 OTÁZKA).
- **bundle**: 10 642 faktů → **57 763 stínů** (⌀ 5,4 na fakt). Ale zatím **NE plně funkční**: 18 % odpovědí paskvil (`on`/`ten`/`který`), 29 % nízkohodnotné díry (`which_attribute`/`how`), 182 mis-segmentovaných obřích vět. **Bůh v katalogu JE** (`state → bůh`).
- **Ollama dávka K=50** (94 otázek): self-consistent **v1 68 % / v2 34 %**, verify ANO 62 %. Blízká parafráze ~2× spolehlivější než volná.

**ÚFAL realita:**
- **UDPipe** (:8092) — atribuuje sektory (deprel/upos/pád), nosná komponenta. ✔
- **MorphoDiTa** (:8093) — **měřeno rozbitá** (forma jako lemma, halucinace). ✘ nespoléhat.
- **NameTag** (:8091) — jména (lazy).

**Jak originál dojde k „Bůh" (princip, ověřeno v kódu):** NE přes co-occurrence (to je pro
identitu vypnuté), ale přes **identitní fakt `být(Ježíš,Bůh)` z pro-drop koreference**
(elidovaný podmět kopule v přímé řeči doplněn z těžiště) → `_identity_vote` → source-attention.
„Kdo napsal R.U.R.? → Čapek" je naopak **co-occurrence** (kontext patro, hub-brána).

---

## 3. Principy

1. **Otázka je stín faktu.** Vždy dopadne na nějaký VZOR-tvar; „žádná shoda" je kategorický omyl. „Minutí" = špatný zdroj světla, nebo nenahraný korpus (→ swap), nebo překryv stínů (→ váhy konverzace). Teprve když zdroj nemá kde vzít → *tápání ≠ terminál* → ptáme se.
2. **Fakt je n-ární.** Vrhá jeden stín na **každou** rolovou díru, ne jeden na fakt. Identitní kopule je **symetricky duální** (obě strany = díra).
3. **Odpověď má vlastní šablonu.** Odpověď-věta je sama fakt → indexuje se stejně → je rekurzivní/skládatelná. Fragment i věta z jedné šablony.
4. **Deterministické jádro + Ollama jen offline.** Runtime match je frame↔frame (žádné LLM v rozhodování). Ollama generuje DOTAZY (povrchy) **offline**, jako ÚFAL anotace. Jazyk = data.
5. **Dva světy se slévají, nestavíme od nuly.** Svět 1 (experiment) = **frame front-end**: porozumění otázce přes occurrence+frame katalog, nahrazuje ruční iris karty (= zjednodušení). Svět 2 (originál) = **graf back-end**: asociativní retrieval (= síla, „Bůh"). Šev = výstup `roles.py` → grafový retrieval.
6. **VZOR = single entita + r=X; sektory atribuuje ÚFAL.** r=X je **číselník závislosti OTÁZKA↔DOTAZ** (nízké r = široké zobecnění, vysoké r = přesná shoda).
7. **Asociaci nesmíme ztratit — „nesmíme ztratit Boha".** Rodí ji těžiště + koreference. Tři pojistky k naroubování (řez): **(a) těžišťová atribuce** (rodí identitu → Bůh), **(b) hub-brána** (trefit vs. zeptat se), **(c) identitní hlas**. Dynamické source-váhy dořeší mezitahovou dvojznačnost (Maria po Ježíši).
8. **Zákon 3.** Změna češtiny → jen JSON, nikdy kód. Univerzální klíče v kódu (UD deprel/pád/role), české hodnoty v `cs.json`.
9. **Etalon.** Každá změna měřena; přijmi jen paritu + aspoň jeden nový zelený řádek, jinak nepřijímej.

---

## 4. Cíl (jednou větou)

Vytvořit **dostatečně bohatý a FUNKČNÍ katalog syntetických vazeb FAKT(y) ↔ OTÁZKA(Y) ↔
ODPOVĚĎ(I)** — grafikon VZORŮ, kde frame front-end (jednoduchost) krmí graf back-end
(asociace) — **zjednodušit originál, ale neztratit jeho asociativní možnosti**.
