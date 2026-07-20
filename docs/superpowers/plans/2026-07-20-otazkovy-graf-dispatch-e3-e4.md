# Otázkový graf: přepnutí dispatch + E3 + E4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Učinit otázkový graf primárním dispatcherem přímých expertů
(fáze D), naučit ho číst schéma predikátů (E3 — chytrá clarifikace
prázdných děr) a odvodit hrany + proaktivní nabídky (E4). Spec:
`docs/superpowers/specs/2026-07-20-otazkovy-graf-dotazeni.md`.

**Architecture:** D: `IrisAutomaton` drží kompilát grafu a brány
přímých expertů řídí smyčka nad `qgraph.claims` (pořadí = data) místo
ručního pořadí v `_turn()`. E3: `FactGraph.predicate_roles()` +
`instance_lit()` v qgraph.py; automat po marném hledání s verdiktem
False vystřelí kartu `query.empty-role` (šablona, jazyk jako data).
E4: hrany otázka→clarify se odvozují z dat karet (díra × event),
proaktivní nabídka neosvětlených rolí odpovědního faktu kartou.

**Tech Stack:** beze změn (Python, pytest, JSON karty, benchmarky).

## Global Constraints

- Stejné jako v plánu `2026-07-20-otazkovy-graf-e1-e2.md`: větev
  (nyní main — pracovat na NOVÉ větvi `qgraph-dispatch`), heredoc
  commity, české uvozovky v JSON, celá sada před každým commitem
  (pytest + 5 benchmarků + run_qgraph obě varianty), žádný pokles.
- Dialog benchmark je PARITY GATE přepnutí — po fázi D musí být
  bitově beze změny (45/45, GAP 3/0, žádná změna textů odpovědí).
- Výroková polovina a příkazy zůstávají ručnímu větvení (#51, etapa 3
  — mimo rozsah tohoto plánu).
- Vakuová logika (past 2): prázdná množina rolí = ŽÁDNÝ verdikt,
  nikdy False.

---

### Task D1: Větev + claims-driven dispatch přímých expertů

**Files:**
- Modify: `jellyai/iris/automaton.py` (`__init__`, `_turn`, nový
  `_clock_response`, `_expert_turn`)
- Test: `tests/test_qgraph.py` (append)

**Interfaces:**
- Consumes: `compile_qgraph`, `QGraph.claims` (E2).
- Produces: `IrisAutomaton.qgraph` (kompilát; staví se v `__init__`);
  `_expert_turn(worker: str, text: str) -> IrisResponse | None`
  (mapa worker→handler: metron→`_metron_query`,
  chronos→`_clock_response`, iris→`_focus_query`; neznámý worker →
  None = propad). Smyčka v `_turn` nahrazuje DNEŠNÍ tři brány:
  metron gate, chronos gate a volání `_focus_query` (řádek ~299).

- [ ] **Step 1: Nová větev + baseline**

```bash
git checkout -b qgraph-dispatch
```
Run: celá sada. Expected: 570 testů, vše 100 % (main po merge).

- [ ] **Step 2: Červený test dispatch z dat**

Do `tests/test_qgraph.py`:

```python
def test_automat_dispatchuje_prime_experty_z_grafu():
    """Fáze D: pořadí bran přímých expertů je v DATECH (claims),
    ne v pořadí větví _turn(); neznámý worker bezpečně propadá."""
    from datetime import datetime
    from config import Config
    from jellyai.iris import IrisAutomaton
    from jellyai.iris.claims import ExpertClaim
    from jellyai.tasks import make_graph_answerer

    iris = IrisAutomaton(make_graph_answerer(Config()),
                         clock=lambda: datetime(2026, 7, 17, 12, 0))
    assert iris.qgraph.nodes["metron-vypocet"].worker == "metron"
    r = iris.turn("Kolik je hodin?")
    assert r.used["components"] == ["chronos"]
    r = iris.turn("Kolik je 1 plus 1?")
    assert "2" in r.text
    # claim s neznámým workerem se hlásí o všechno — musí PROPADNOUT
    vetrelec = ExpertClaim("vetrelec", "neexistuje", 9,
                           lambda text, now: True)
    iris.qgraph.claims = (vetrelec,) + tuple(iris.qgraph.claims)
    r = iris.turn("Kolik je hodin?")
    assert r.used["components"] == ["chronos"]
```

Run: `.venv/bin/python -m pytest tests/test_qgraph.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'qgraph'`.

- [ ] **Step 3: Implementace**

V `__init__` automatu (za načtení decku) přidej:

```python
        self.qgraph = compile_qgraph(
            self.deck, getattr(answerer, "_predicates", frozenset()))
```

(+ import `from jellyai.iris.qgraph import compile_qgraph` nahoru.)

Vyjmi chronos blok do metody:

```python
    def _clock_response(self, text):
        """Hodinová otázka — odpovídá Chronos sám (časová kotva)."""
        direct = clock_answer(text, self.clock())
        if direct is None:
            return None
        response = IrisResponse(
            text=direct, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": []})
        self.state.remember(text, response)
        return response
```

Přidej dispatcher:

```python
    def _expert_turn(self, worker, text):
        """Tah přímého experta podle worker atributu uzlu grafu."""
        handler = {"metron": self._metron_query,
                   "chronos": self._clock_response,
                   "iris": self._focus_query}.get(worker)
        return handler(text) if handler else None
```

V `_turn` NAHRAĎ metron gate + chronos gate (začátek metody) smyčkou:

```python
        # PŘÍMÍ EXPERTI dispatchem grafu (#57 fáze D): pořadí bran nesou
        # DATA (claims priorita), ne pořadí větví — konec pasti #51
        for claim in sorted(self.qgraph.claims,
                            key=lambda c: -c.priority):
            if not claim.recognize(text, self.clock()):
                continue
            handled = self._expert_turn(claim.worker, text)
            if handled is not None:
                return handled
```

a SMAŽ volání `_focus_query` u řádku ~299 (blok „META: stav
rozhovoru…" včetně komentáře). POZOR: meta se tím posouvá PŘED resume
identity/pick — parity gate (dialog benchmark) rozhodne, zda je to
neutrální; případný rozdíl je nález, ne k obcházení.

- [ ] **Step 4: Zelené testy + parity gate**

Run: `.venv/bin/python -m pytest -q` + celá sada benchmarků + harness.
Expected: 571+ testů PASS, dialog 45/45 bitově (žádná změna odpovědí),
vše ostatní beze změny čísel.

- [ ] **Step 5: Commit**

```bash
git add jellyai/iris/automaton.py tests/test_qgraph.py
git commit -F - <<'MSG'
feat(qgraph): #57 fáze D — dispatch přímých expertů řízen grafem

Brány metron/chronos/meta v turn() nahrazeny smyčkou nad
qgraph.claims (pořadí = data, konec pasti #51 pro přímé experty);
neznámý worker bezpečně propadá. Parity gate: dialog benchmark
bitově beze změny.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
MSG
```

---

### Task D2: Uzávěrka fáze D (docs)

**Files:**
- Modify: spec (poznámka do §6 + Empirie D), `docs/BACKLOG.md` (#57)

- [ ] **Step 1:** Do spec za Empirii E2 přidej `**Empirie D (přepnutí
dispatch přímých expertů):**` — co se přepnulo (3 brány → smyčka nad
claims), parity výsledek, co zůstává ručně (výroky/příkazy — #51
etapa 3). V §6 uprav odrážku „přepnutí dispatch" na „✅ přímí experti
přepnuti (fáze D); zbývá výroková polovina (#51)". BACKLOG #57: „FÁZE
D HOTOVA (dispatch přímých expertů z claims, parity bitová)".

- [ ] **Step 2:** Commit (`docs(qgraph): #57 fáze D — empirie`).

---

### Task E3-1: Schéma s rolemi + instance_lit

**Files:**
- Modify: `jellyai/graph/graph.py` (`FactGraph.predicate_roles`),
  `jellyai/iris/qgraph.py` (`instance_lit`)
- Test: `tests/test_qgraph.py` (append)

**Interfaces:**
- Produces: `FactGraph.predicate_roles(predicate) -> frozenset[str]`
  (role účastníků napříč fakty predikátu; prázdná = predikát nezná);
  `instance_lit(predicate, hole_role, roles_of) -> bool | None`
  v qgraph.py (None = žádný verdikt: neznámý predikát/prázdné role —
  vakuový guard; jinak `hole_role in roles`).

- [ ] **Step 1: Ověř tvar participantů**

Run: `grep -n "class Participant" jellyai/graph/*.py && grep -n "p.role\|participant.role" jellyai/answerer/graph_answerer.py | head -3`
Expected: Participant má atribut `role`; pokud se jmenuje jinak,
použij skutečné jméno v kroku 3.

- [ ] **Step 2: Červený test**

```python
def test_instance_ze_schematu_predikatu():
    """E3: instance svítí jen s rolí díry ve schématu; neznámý
    predikát nebo prázdné role = ŽÁDNÝ verdikt (vakuum, past 2)."""
    from config import Config
    from jellyai.iris.qgraph import instance_lit
    from jellyai.tasks import make_graph_answerer

    graph = make_graph_answerer(Config()).graph
    roles = graph.predicate_roles("napsat")
    assert "subj" in roles and "obj" in roles
    assert "loc" not in roles
    roles_of = graph.predicate_roles
    assert instance_lit("napsat", "obj", roles_of) is True
    assert instance_lit("napsat", "loc", roles_of) is False
    assert instance_lit("blafnout", "loc", roles_of) is None
```

Run + Expected: FAIL (`predicate_roles` neexistuje).

- [ ] **Step 3: Implementace**

`FactGraph.predicate_roles` (do graph.py k `facts_of`):

```python
    def predicate_roles(self, predicate):
        """Role, ve kterých fakty predikátu nesou účastníky (#57 E3).

        Schéma otázkového grafu se NEkurátoruje — čte se z faktů;
        prázdná množina = graf predikát nezná (žádný verdikt).
        """
        roles = set()
        for fact in self.facts.values():
            if fact.predicate == predicate:
                roles.update(p.role for p in fact.participants)
        return frozenset(roles)
```

`instance_lit` (qgraph.py, za `decorate`):

```python
def instance_lit(predicate, hole_role, roles_of):
    """Verdikt líné instance (rodina × predikát) nad schématem (E3).

    Returns:
        bool | None: True = role díry ve schématu je; False = fakty
        predikátu roli nikdy nenesou (hledání je marné — chytrá
        clarifikace); None = predikát neznám / bez rolí (vakuový
        guard, past 2 — nesoudit).
    """
    roles = roles_of(predicate) if predicate else frozenset()
    if not roles:
        return None
    return hole_role in roles if hole_role else None
```

- [ ] **Step 4:** pytest celý + commit
(`feat(qgraph): #57 E3 — schéma s rolemi (predicate_roles, instance_lit)`).

---

### Task E3-2: Červené řádky chytré clarifikace

**Files:**
- Modify: `benchmark/etalon.jsonl` (gap řádek),
  `benchmark/dialog.jsonl` (gap scénář)

- [ ] **Step 1:** Zjisti dnešní odpověď:
`curl` netřeba — spusť `.venv/bin/python -c` s answererem na otázku
„Kde napsal R.U.R.?" a vypiš text. Expected: terminál
(„V textu jsem odpověď nenašel." nebo podobný miss).

- [ ] **Step 2:** Přidej řádky (české uvozovky!). Etalon:

```json
{"q": "Kde napsal R.U.R.?", "expect": ["nevím"], "cat": "prázdná díra", "gap": "#57 E3: predikát napsat roli loc nikdy nenese — místo generického terminálu má odpovědět chytrá clarifikace (vím kdo a co, kde nevím)"}
```

Dialog scénář:

```json
{"name": "prazdna-dira-clarify", "gap": "#57 E3: chytrá clarifikace prázdné díry ze schématu predikátů", "turns": [{"u": "Kde napsal R.U.R.?", "expect": ["nevím"], "reject": ["V textu jsem odpověď nenašel"]}]}
```

- [ ] **Step 3:** Run etalon + dialog → oba nové řádky ČERVENÉ (gap
otevřený); jádra beze změny. Commit
(`test(qgraph): #57 E3 — červené řádky chytré clarifikace`).

---

### Task E3-3: Karta query-empty-role + zapojení

**Files:**
- Create: `jellyai/iris/patterns/cs/query-empty-role.json`
- Modify: `jellyai/lang/cs.json` (tabulka `role_labels`),
  `jellyai/iris/automaton.py` (hook po marné odpovědi)

**Interfaces:**
- Consumes: `instance_lit`, `predicate_roles` (E3-1);
  `answerer.last_pattern` (predicate, hole_role), `answerer.last_trace`
  (None = žádný fakt neodpověděl).
- Produces: event `query.empty-role` s kontextem
  `{predicate, missing, known}` (missing/known = česká slova rolí
  z `role_labels`).

- [ ] **Step 1:** Do cs.json přidej (k tabulkám answereru):

```json
  "role_labels": {"subj": "kdo", "obj": "co", "loc": "kde",
                  "time": "kdy", "num": "kolik"},
```

- [ ] **Step 2:** Karta `query-empty-role.json`:

```json
{
  "name": "query-empty-role",
  "trigger": {"event": "query.empty-role", "priority": 5},
  "dialog": "O ději „{predicate}“ vím {known}; {missing} nevím.",
  "action": {},
  "teach": "Chytrá clarifikace prázdné díry (#57 E3): otázka míří na roli, kterou fakty predikátu NIKDY nenesou (schéma z datového grafu, žádná kurátorovaná sémantika) — místo marného hledání a generického terminálu systém poctivě řekne, které role zná a kterou ne. Vakuový guard: neznámý predikát verdikt nedostane (past 2) a jede běžný terminál."
}
```

- [ ] **Step 3:** Hook v automatu — v `_turn` NA MÍSTĚ, kde se vrací
odpověď dotazu (větev `self._respond(answer, "answer", …)` po
`answerer.answer()`): PŘED `_respond` vlož:

```python
        empty = self._empty_role(text)
        if empty is not None:
            return empty
```

a metodu:

```python
    def _empty_role(self, text):
        """Chytrá clarifikace prázdné díry (#57 E3, karta query.empty-role)."""
        from jellyai.iris.qgraph import instance_lit
        pat = self.answerer.last_pattern
        if (pat is None or self.answerer.last_trace is not None
                or not getattr(pat, "predicate", None)
                or not getattr(pat, "hole_role", None)):
            return None
        verdict = instance_lit(pat.predicate, pat.hole_role,
                               self.answerer.graph.predicate_roles)
        if verdict is not False:
            return None
        labels = current().get("role_labels", {})
        roles = self.answerer.graph.predicate_roles(pat.predicate)
        known = ", ".join(sorted(labels[r] for r in roles if r in labels))
        missing = labels.get(pat.hole_role)
        card = self.deck.best("query.empty-role", {})
        if card is None or not known or missing is None:
            return None
        response = IrisResponse(
            text=card.dialog.format(predicate=pat.predicate,
                                    known=known, missing=missing),
            kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["graph-answerer"],
                  "patterns": [card.name]})
        self.state.remember(text, response)
        return response
```

Přesné místo vložení najdi: `grep -n "_respond(answer" automaton.py`
— hook patří před OBĚ místa (řádky ~353 a ~358)? NE — jen před
KONCOVÉ `_respond` (řádek ~358, cesta bez overflow dialogu); overflow
větev má fakty, last_trace není None, hook je tam neutrální.

- [ ] **Step 4:** Celá sada. Expected: etalon GAP-FIXED řádek
„Kde napsal R.U.R.?" (obsahuje „nevím"), dialog scénář GAP-FIXED,
vše ostatní beze změny. POZOR na reject „V textu jsem odpověď
nenašel" — nesmí zůstat. Commit
(`feat(qgraph): #57 E3 — chytrá clarifikace prázdné díry kartou`).

---

### Task E3-4: Uzávěrka E3

**Files:** spec (Empirie E3), BACKLOG (#57)

- [ ] **Step 1:** Empirie E3 do spec: verdikt instance_lit, čísla
(GAP-FIXED řádky, parita), vakuový guard, poznámka: schéma se čte
ŽIVĚ z grafu (Mnemos predikáty instance vidí — roles_of jde přes
graph, ne kompilát). BACKLOG: „E3 HOTOVO (…)". Commit.

---

### Task E4-1: Odvozené clarify hrany

**Files:**
- Modify: `jellyai/iris/qgraph.py` (compile_qgraph — derivace hran),
  clarify karty (metadata `fits` NEpotřeba — derivace z eventu a díry
  karty otázky, čistě z existujících dat)
- Test: `tests/test_qgraph.py` (append/uprav)

**Interfaces:**
- Produces: hrany otázka→clarify jen kde dávají smysl:
  `statement.*` clarify NIKDY z otázky; `data.overflow` jen z karet
  S dírou (action.query.hole); `resolve.ambiguous` a `focus.low`
  ze všech otázek. Kartézský součin končí.

- [ ] **Step 1: Červený test**

```python
def test_clarify_hrany_se_odvozuji():
    """E4: hrany otázka→clarify z dat karet (díra × event), ne
    kartézský součin — statement clarify z otázky nevede, overflow
    jen z výčtových děr."""
    qg = _graph()
    otaz = qg.nodes["q-otaz-minuly"]           # má díru
    cile = {e.target for e in otaz.edges if e.kind == "zpresneni"}
    assert "focus-offer-overflow" in cile
    assert "clarify-identity" not in cile       # statement-side
    zjist = qg.nodes["q-zjistovaci-prezens"]    # bez díry
    cile = {e.target for e in zjist.edges if e.kind == "zpresneni"}
    assert "focus-offer-overflow" not in cile   # existence nepřeteče
    assert "focus-offer-homonym" in cile
```

Expected: FAIL (dnes kartézský součin — clarify-identity všude).

- [ ] **Step 2:** V `compile_qgraph` nahraď blok hran:

```python
    for node in nodes.values():
        if node.kind != "otazka":
            continue
        # digging hrany se ODVOZUJÍ z dat (E4): statement clarify
        # z otázky nevede; overflow jen z karet s dírou (výčty)
        has_hole = bool(cards_by_name[node.card].action
                        .get("query", {}).get("hole"))
        node.edges = [
            QEdge("zpresneni", name) for name, event in clarify_events
            if not event.startswith("statement.")
            and (event != "data.overflow" or has_hole)]
```

kde `clarify_events` je seznam `(name, event)` clarify karet
a `cards_by_name` mapa jmen karet decku (postav při kompilaci).

- [ ] **Step 3:** Celá sada — stav 3/3 drží (overflow hrana
z q-co-vime, karta má díru), počet hran klesl (vypiš
`sum(len(n.edges) for n in ...)` před/po do commit zprávy). Commit
(`feat(qgraph): #57 E4 — odvozené clarify hrany (konec kartézského součinu)`).

---

### Task E4-2: Proaktivní nabídka neosvětlených rolí

**Files:**
- Create: `jellyai/iris/patterns/cs/answer-offer-roles.json`
- Modify: `jellyai/iris/automaton.py`, `benchmark/dialog.jsonl`
  (gap scénář NAPŘED)

**Interfaces:**
- Consumes: `answerer.last_trace["fact"]`, `graph.facts`,
  `role_labels` (E3-3), `answerer.last_pattern` (hole_role, known).
- Produces: event `answer.offer-roles`; nabídka se PŘIPOJUJE za text
  odpovědi (jako odpálené připomínky); follow-up „Kde?" obslouží
  EXISTUJÍCÍ drill (q-hola-otazka přes _prev_trace) — nabídka je
  čistě dialogový akt.

- [ ] **Step 1:** Zjisti reálné hodnoty pro scénář:
spusť answerer na „Kdy se narodil Karel Čapek?" a „Kde se narodil
Karel Čapek?" a poznamenej odpovědi (datum, místo). Scénář piš
s NAMĚŘENÝMI hodnotami.

- [ ] **Step 2:** Gap scénář (dialog.jsonl, hodnoty z kroku 1):

```json
{"name": "proaktivni-nabidka-roli", "gap": "#57 E4: po odpovědi s neosvětlenými rolemi faktu nabídnout doplnění (nabídka, ne vnucování — dialog > figly)", "turns": [{"u": "Kdy se narodil Karel Čapek?", "expect": ["<datum>", "Mohu doplnit"]}, {"u": "Kde?", "expect": ["<místo>"]}]}
```

Run dialog → červený. Commit gap řádku.

- [ ] **Step 3:** Karta `answer-offer-roles.json`:

```json
{
  "name": "answer-offer-roles",
  "trigger": {"event": "answer.offer-roles", "priority": 3},
  "dialog": "Mohu doplnit: {roles}.",
  "action": {},
  "teach": "Proaktivní nabídka (#57 E4): odpovědní fakt nese role, na které se nikdo neptal (neosvětlené role žhavého faktu — T4 spec). Systém je po odpovědi NABÍDNE (nevnucuje — dialog > figly); follow-up „Kde?“ obslouží drill holé otázky. Pozorovatel (theme) se nenabízí — není odpověď."
}
```

V automatu za úspěšnou odpověď (kde `_respond(answer, "answer", …)`
vrací response na KONCOVÉ cestě) vlož před return obohacení:

```python
            offer = self._offer_roles()
            if offer:
                answer.text = f"{answer.text}\n{offer}"
```

a metodu:

```python
    def _offer_roles(self):
        """Nabídka neosvětlených rolí odpovědního faktu (#57 E4)."""
        trace = self.answerer.last_trace
        if not isinstance(trace, dict) or trace.get("fact") is None:
            return None
        fact = self.answerer.graph.facts.get(trace["fact"])
        pat = self.answerer.last_pattern
        if fact is None or pat is None:
            return None
        labels = current().get("role_labels", {})
        asked = {getattr(pat, "hole_role", None)}
        asked.update(role for role, _ in getattr(pat, "known", ()) or ())
        unlit = sorted(labels[p.role] for p in fact.participants
                       if p.role in labels and p.role not in asked
                       and p.role != "theme")   # pozorovatel není nabídka
        if not unlit:
            return None
        card = self.deck.best("answer.offer-roles", {})
        if card is None:
            return None
        return card.dialog.format(roles=", ".join(dict.fromkeys(unlit)))
```

(Tvar `pat.known` ověř: `grep -n "known" jellyai/answerer/query.py
| head -5` — je-li to seznam dvojic (role, term), kód sedí; jinak
uprav destrukturaci podle skutečnosti.)

- [ ] **Step 4:** Celá sada. Expected: gap scénář GAP-FIXED (odpověď
nese „Mohu doplnit: …" a drill „Kde?" odpoví místem); dialog jádro
45/45 — POZOR: nabídka se připojí i k jiným odpovědím; pokud tím
padne existující scénář (expect/reject kolize), je to NÁLEZ — zvaž
guard (nabídka jen pro kind answer bez overflow/drill) a zapiš do
Empirie. Commit
(`feat(qgraph): #57 E4 — proaktivní nabídka rolí kartou`).

---

### Task E4-3: Uzávěrka E4 + celku

**Files:** spec (Empirie E4), BACKLOG (#57), paměť

- [ ] **Step 1:** Empirie E4 (hrany před/po, scénář, nálezy). BACKLOG
#57: „E4 HOTOVO; zbývá výroková polovina (#51) a odpovědní graf
(park)". Aktualizuj paměťové soubory (MEMORY.md + jellyai3-fact-graph)
o stav D+E3+E4.

- [ ] **Step 2:** Finální celá sada + commit + merge do main dle volby
uživatele (finishing-a-development-branch).
