"""Univerzální princip odpovídání: **otázka = neúplný fakt s dírou**.

Otázku rozebereme na `Pattern` — predikát + známí účastníci (role→termín) + **díra**
(role a preferovaný typ, kam patří odpověď). Odpovídání je pak *match* proti grafu:
najdi fakt shodný ve známých rolích a vrať účastníka v roli díry, seřazeno aktivací.

Bez ručních qtype pravidel a bez slovníků: **vztahy jsou struktura** (relační jméno
+ genitivní osoba ve spone → predikát = to jméno), tázací slovo určuje díru. Jazykové
je jen mapování tázacích lemmat a deprelů — tenká vrstva, ne rozlitá logika.
"""

from dataclasses import dataclass, field

from jellyai.answerer.selection import _clean_lemma

# tázací lemma → (pevná role díry | None = vezmi z deprelu, preferovaný typ | None)
_HOLE = {
    "kdo": (None, "person"), "co": (None, None), "čí": (None, "person"),
    "kdy": ("time", "time"), "kde": ("loc", "geo"), "kam": ("loc", "geo"),
    "kolik": ("num", "number"), "jaký": ("attr", None), "který": ("attr", None),
}
_DEPREL_ROLE = {"nsubj": "subj", "nsubj:pass": "subj", "obj": "obj", "iobj": "obj"}
_CONTENT = {"NOUN", "PROPN", "NUM", "ADJ"}


_DATE_PARTS = {"rok", "měsíc", "den"}   # „v kterém roce" = 2-skokový drill, ne pattern


@dataclass
class SubQuery:
    """Vnořený pod-dotaz (rekurze): „bratr Karla Čapka" = najdi bratra Karla.
    `known` může obsahovat další SubQuery — hloubka je dynamická dle zanoření."""
    predicate: str
    known: list = field(default_factory=list)   # [(role, termín | SubQuery)]
    hole_role: str = "subj"


@dataclass
class Pattern:
    """Neúplný fakt z otázky: predikát, známé role→(termín|SubQuery) a díra."""
    predicate: str = None
    known: list = field(default_factory=list)   # [(role, termín | SubQuery)]
    hole_role: str = None
    hole_type: str = None
    date_part: str = None                       # „rok/měsíc/den" → 2-skokový drill


def _genitive_child(head_tok, sent):
    """Genitivní přívlastek (nmod) daného tokenu — pro „bratr **Karla Čapka**"."""
    hid = sent.index(head_tok) + 1
    for tok in sent:
        if tok.get("head") == hid and str(tok.get("deprel", "")).startswith("nmod"):
            return tok
    return None


def _rel_clause(head_tok, sent):
    """Vztažná věta (acl/acl:relcl) modifikující token — „autor **který napsal X**"."""
    hid = sent.index(head_tok) + 1
    for tok in sent:
        if tok.get("head") == hid and str(tok.get("deprel", "")).startswith("acl"):
            return tok
    return None


def _known_of(token, sent):
    """Rekurzivně: token → termín (list) NEBO **SubQuery** (vnořený pod-dotaz).

    Auto-trigger ze struktury, hloubka dynamická dle zanoření:
    * **vztažná věta** „autor který napsal R.U.R." → SubQuery(napsat, obj=R.U.R.),
    * **genitiv** „bratr Karla Čapka" → SubQuery(bratr, obj=Karel),
    * jinak list = víceslovná entita.
    """
    relcl = _rel_clause(token, sent)
    if relcl is not None:
        rid = sent.index(relcl) + 1
        known, hole_role = [], "subj"
        for tok in sent:
            if tok.get("head") != rid or tok is token:
                continue
            role = _DEPREL_ROLE.get(tok.get("deprel"))
            if tok.get("upos") == "DET" and tok.get("deprel") in ("nsubj", "nsubj:pass"):
                hole_role = role or "subj"      # relativum „který" = díra (=hlava)
            elif role and tok.get("upos") in _CONTENT \
                    and _clean_lemma(tok.get("lemma", "")).lower() not in _HOLE:
                known.append((role, _known_of(tok, sent)))
        return SubQuery(_clean_lemma(relcl.get("lemma", "")), known, hole_role)
    gen = _genitive_child(token, sent)
    if gen is not None and token.get("upos") == "NOUN" \
            and gen.get("upos") in ("PROPN", "NOUN"):
        return SubQuery(_clean_lemma(token.get("lemma", "")),
                        [("obj", _known_of(gen, sent))], "subj")
    return _entity_term(token, sent)


def _entity_term(head_tok, sent):
    """Termín entity = hlavní token + jeho `flat`/`nmod` části („Karel"+„Čapek")."""
    hid = sent.index(head_tok) + 1
    words = [_clean_lemma(head_tok.get("lemma", ""))]
    for tok in sent:
        if tok.get("head") == hid and tok.get("deprel") in ("flat", "flat:name"):
            words.append(_clean_lemma(tok.get("lemma", "")))
    return " ".join(w for w in words if w)


def question_pattern(question, client):
    """Rozebere otázku na `Pattern` (viz modul). Bere první větu rozboru."""
    for sent in client.parse(question):
        return _parse_sent(sent)
    return Pattern()


def _parse_sent(sent):
    """Z tokenů věty sestaví Pattern (predikát, známí účastníci, díra)."""
    date_part = next((_clean_lemma(t.get("lemma", "")).lower() for t in sent
                      if _clean_lemma(t.get("lemma", "")).lower() in _DATE_PARTS), None)
    cop = any(t.get("deprel") == "cop" or _clean_lemma(t.get("lemma", "")) == "být"
              for t in sent)
    verb = next((_clean_lemma(t.get("lemma", "")) for t in sent
                 if t.get("upos") == "VERB"), None)
    hole_role = hole_type = None
    for tok in sent:                       # díra = tázací slovo
        low = _clean_lemma(tok.get("lemma", "")).lower()
        if low in _HOLE:
            fixed, hole_type = _HOLE[low]
            hole_role = fixed or _DEPREL_ROLE.get(tok.get("deprel"))
            break

    known = []
    if cop:
        # přísudkové jméno (root, ne tázací) = entita; s genitivní osobou = relace
        root = next((t for t in sent if t.get("deprel") == "root"
                     and t.get("upos") in ("NOUN", "PROPN")
                     and _clean_lemma(t.get("lemma", "")).lower() not in _HOLE), None)
        if root is not None:
            gen = _genitive_child(root, sent)
            if gen is not None and gen.get("upos") in ("PROPN", "NOUN"):
                # „Kdo je bratr autora který napsal X?" → predikát „bratr", obj =
                # rekurzivně vyřešená osoba (i vnořený pod-dotaz)
                known.append(("obj", _known_of(gen, sent)))
                return Pattern(_clean_lemma(root.get("lemma", "")), known,
                               hole_role or "subj", hole_type, date_part)
            # „Kdo/Co je X?" → identita X (díra = pred); „Jaký je X?" → attr
            known.append(("subj", _entity_term(root, sent)))
            return Pattern("být", known,
                           "attr" if hole_role == "attr" else "pred", hole_type)

    for tok in sent:                       # známí účastníci ze slovesné věty
        low = _clean_lemma(tok.get("lemma", "")).lower()
        if low in _HOLE or tok.get("upos") not in _CONTENT:
            continue
        if tok.get("deprel") in ("flat", "flat:name"):
            continue                       # část víceslovné entity, řeší _entity_term
        role = _DEPREL_ROLE.get(tok.get("deprel"))
        if role:
            known.append((role, _known_of(tok, sent)))   # rekurzivně (vnořené pod-dotazy)
    return Pattern(verb, known, hole_role, hole_type, date_part)
