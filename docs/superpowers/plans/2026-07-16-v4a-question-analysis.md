# V4a — bohatá analýza otázky + sponové odpovědi — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Kroky mají checkbox (`- [ ]`).
> **Realizace:** většinu kódu píše Claude sám.

**Goal:** Rozpoznávat typ otázky přes lemma (Jaký/Který/Čí + varianty) a odpovídat na sponové/predikátové otázky („X je/byl Y" → Y).

**Tech Stack:** Python 3.11, stávající V3 vrstva (UDPipe klient/služby, selection, templates). Bez nových závislostí.

## Global Constraints
- Python 3.11 venv; commity končí `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Hermetické testy přes `FakeUfalClient`; anotace se nemění (parse už obsahuje `cop`/`deprel`).
- Větev `feature/v4a-question-analysis`.

---

### T1 — Bohatá analýza otázky (`jellyai/answerer/question.py`)
**Files:** Create `jellyai/answerer/question.py`, Test `tests/test_question_analysis.py`

**Interfaces:** `QuestionAnalysis(qtype, verb_lemma, is_copula, topic_terms)`;
`analyze_question(question, client) -> QuestionAnalysis`.

- [ ] Test: „Jaká byla Němcová?" (FakeUfalClient parse) → qtype „Jaký", is_copula True;
  varianta „Jaké je X?" → taky „Jaký". „kdo napsal X?" → „Kdo", is_copula False.
- [ ] Implementace:
```python
from dataclasses import dataclass, field
from jellyai.answerer.selection import _clean_lemma

_QTYPE_BY_LEMMA = {
    "kdo": "Kdo", "co": "Co", "kde": "Kde", "kam": "Kde", "odkud": "Kde",
    "kdy": "Kdy", "kolik": "Kolik", "jaký": "Jaký", "který": "Který", "čí": "Čí",
}


@dataclass
class QuestionAnalysis:
    qtype: str = None
    verb_lemma: str = None
    is_copula: bool = False
    topic_terms: list = field(default_factory=list)


def analyze_question(question, client):
    qa = QuestionAnalysis()
    for sentence in client.parse(question):
        for tok in sentence:
            lemma = _clean_lemma(tok.get("lemma", "")).lower()
            upos = tok.get("upos", "")
            if qa.qtype is None and lemma in _QTYPE_BY_LEMMA:
                qa.qtype = _QTYPE_BY_LEMMA[lemma]
            if upos == "VERB" and qa.verb_lemma is None:
                qa.verb_lemma = _clean_lemma(tok.get("lemma", ""))
            if lemma == "být" or tok.get("deprel") == "cop":
                qa.is_copula = True
            if upos in ("NOUN", "PROPN", "ADJ") and lemma not in _QTYPE_BY_LEMMA:
                qa.topic_terms.append(_clean_lemma(tok.get("lemma", "")))
    return qa
```
- [ ] Commit `feat: bohatá analýza otázky (typ přes lemma, spona, témata)`.

---

### T2 — Sponová strategie + filtr zájmen (`jellyai/answerer/selection.py`)
**Files:** Modify `selection.py`, Test `tests/test_selection.py` (přidat)

**Interfaces:** `select_predicate(annotation, qtype) -> Candidate|None`; `select_answer`
získá parametr `is_copula=False` a dispatch.

- [ ] Test: anotace se sponou („Němcová byla významná spisovatelka" — root „spisovatelka",
  cop „byla", amod „významná") → `select_predicate` vrátí „významná spisovatelka".
- [ ] Test: „Co" s předmětem-zájmenem → přeskočí, vybere podstatné jméno.
- [ ] Implementace — přidat konstanty a funkci:
```python
_PREDICATE_MODIFIERS = {"amod", "flat", "compound"}
_SKIP_UPOS = {"PRON", "DET", "CCONJ", "ADP", "PUNCT"}


def select_predicate(annotation, qtype):
    """Sponový přísudek: kořen se sponou `cop` + jeho přívlastky."""
    for sent in annotation.get("sentences", []):
        cop_heads = [t["head"] for t in sent if t.get("deprel") == "cop"]
        for head_id in cop_heads:
            if not 0 < head_id <= len(sent):
                continue
            pred = sent[head_id - 1]
            phrase = [pred] + [t for t in sent
                               if t.get("head") == head_id and t.get("deprel") in _PREDICATE_MODIFIERS]
            phrase.sort(key=lambda t: t.get("start") or 0)
            return Candidate(form=" ".join(t["form"] for t in phrase),
                             lemma=_nominative(phrase), qtype=qtype)
    return None
```
- [ ] Upravit `_select_object` — přeskočit `_SKIP_UPOS`:
```python
            if tok.get("deprel") not in _OBJECT or tok.get("upos") in _SKIP_UPOS:
                continue
```
- [ ] Upravit `select_answer` — přidat `is_copula` a dispatch:
```python
def select_answer(qtype, verb_lemma, annotation, is_copula=False):
    if is_copula or qtype in ("Jaký", "Který"):
        pred = select_predicate(annotation, qtype)
        if pred:
            return pred
    if qtype == "Kdo" or qtype == "Čí":
        return _select_subject_entity(annotation, verb_lemma, qtype)
    if qtype == "Co":
        return _select_object(annotation, verb_lemma, qtype)
    if qtype == "Kde":
        return _select_entity(annotation, _GEO, qtype)
    if qtype == "Kdy":
        return _select_entity(annotation, _TIME, qtype)
    if qtype == "Kolik":
        return _select_number(annotation, qtype)
    return None
```
- [ ] Commit `feat: sponová strategie výběru přísudku + filtr zájmen u Co`.

---

### T3 — Šablony pro nové typy (`jellyai/templates.py`)
**Files:** Modify `templates.py`, Test `tests/test_templates.py` (přidat)

- [ ] Test: `target_case("Jaký") is None`; `target_case("Který") == "1"`; `fill("Jaký", "významná spisovatelka") == "významná spisovatelka"`.
- [ ] Přidat do `TEMPLATES`:
```python
    "Jaký": {"frame": "{answer}", "case": None},      # přísudek/adjektivum — ponech tvar
    "Který": {"frame": "{answer}", "case": _NOMINATIVE},
    "Čí": {"frame": "{answer}", "case": _NOMINATIVE},
```
- [ ] Commit `feat: šablony pro Jaký/Který/Čí`.

---

### T4 — Zapojení do TemplateAnswereru (`jellyai/answerer/template.py`)
**Files:** Modify `template.py`, Test `tests/test_template_answerer.py` (přidat)

- [ ] Test: „kdo je Rossum?" se sponovou anotací → přísudek (definice), ne „Rossum".
  „Jaká byla Němcová?" → „významná spisovatelka".
- [ ] Nahradit `_analyze_question` voláním `analyze_question` a předat `is_copula`:
```python
from jellyai.answerer.question import analyze_question
...
    def answer(self, question, retrieved):
        if not retrieved:
            return self.fallback.answer(question, retrieved)
        qa = analyze_question(question, self.client)
        if qa.qtype is None:
            return self.fallback.answer(question, retrieved)
        for passage, score in retrieved:
            annotation = self.annotations.get((passage.doc_id, passage.index))
            if not annotation:
                continue
            candidate = select_answer(qa.qtype, qa.verb_lemma, annotation,
                                      is_copula=qa.is_copula)
            if candidate is None:
                continue
            text = self._render(qa.qtype, candidate)
            if text.strip():
                return Answer(text=text, sources=[f"{passage.doc_id}#{passage.index}"],
                              score=float(score))
        return self.fallback.answer(question, retrieved)
```
(Odstranit starou funkci `_analyze_question` a `_QWORDS` z template.py — nahrazeny question.py.)
- [ ] Commit `feat: TemplateAnswerer používá bohatou analýzu + sponovou strategii`.

---

### T5 — Reálné ověření + poznámka do výsledků
**Files:** Modify `docs/superpowers/2026-07-16-v3-results.md` (nebo nový v4a), `README.md`

- [ ] Naživo přes wrapper: `./jelly template "Jaká byla Božena Němcová?"`,
  `./jelly template "kdo je Rossum?"` — porovnat s dřívějškem (má zmizet tautologie,
  „Jaký" má odpovídat). Anotace se nepřegenerovávají.
- [ ] Zapsat honest výsledky (co se zlepšilo, co zůstává).
- [ ] Commit `docs: V4a výsledky + README`.

## Self-Review
Spec pokryt: bohatá analýza (T1), sponová strategie + filtr (T2), šablony (T3),
integrace (T4), ověření (T5). Hermetické testy přes FakeUfalClient; anotace beze změny. ✔
Placeholder scan: konkrétní kód u všech tasků. ✔
Type consistency: `analyze_question -> QuestionAnalysis`; `select_answer(..., is_copula)`
volané v T4 sedí na signaturu z T2; `Candidate` konzistentní. ✔
