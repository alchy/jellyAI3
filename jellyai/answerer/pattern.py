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
from jellyai.lang import current

# tázací lemma → (pevná role díry | None = vezmi z deprelu, preferovaný typ | None)
_HOLE = {
    "kdo": (None, "person"), "co": (None, None), "čí": (None, "person"),
    "kdy": ("time", "time"), "kde": ("loc", "geo"), "kam": ("loc", "geo"),
    "kolik": ("num", "number"), "jaký": ("attr", None), "který": ("attr", None),
}
_DEPREL_ROLE = {"nsubj": "subj", "nsubj:pass": "subj", "obj": "obj", "iobj": "obj",
                "obl": "theme", "obl:arg": "theme"}   # obl účastník („s Karlem")
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
    """HOLÝ genitivní přívlastek tokenu — pro „bratr **Karla Čapka**".

    nmod s předložkou („Válka **s mloky**", „drama **o robotech**") genitivní
    vztah NENÍ — je to fráze uvnitř termínu; poznáme ji podle `case` (ADP)
    dítěte. Parser navíc genitiv občas označkuje jako `flat` — prozradí ho
    PÁDOVÁ NESHODA s hlavou („bratr[Nom] Karla[Gen]"): tvar rozhoduje.
    """
    hid = sent.index(head_tok) + 1
    head_case = head_tok.get("feats", {}).get("Case")
    for i, tok in enumerate(sent):
        if tok.get("head") != hid:
            continue
        deprel = str(tok.get("deprel", ""))
        genitive_flat = (deprel.startswith("flat")
                         and tok.get("feats", {}).get("Case") == "Gen"
                         and head_case not in (None, "Gen"))
        if not (deprel.startswith("nmod") or genitive_flat):
            continue
        has_adp = any(c.get("head") == i + 1 and c.get("upos") == "ADP"
                      for c in sent)
        if not has_adp:
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
    """Termín entity = hlavní token + `flat` části VE STEJNÉM PÁDU.

    Pádová neshoda flat dítěte („bratr[Nom] Karla[Gen]") značí mis-tagged
    genitiv — do termínu nepatří. Je-li hlava sama flat (genitivní člen),
    přiberou se její flat sourozenci v témže pádu („Karla + Čapka").
    """
    hid = sent.index(head_tok) + 1
    own_case = head_tok.get("feats", {}).get("Case")
    words = [_clean_lemma(head_tok.get("lemma", ""))]
    for tok in sent:
        if tok.get("head") == hid and tok.get("deprel") in ("flat", "flat:name"):
            case = tok.get("feats", {}).get("Case")
            if own_case and case and case != own_case:
                continue
            words.append(_clean_lemma(tok.get("lemma", "")))
    if str(head_tok.get("deprel", "")).startswith("flat"):
        pos = sent.index(head_tok)
        for tok in sent:
            if tok is not head_tok and tok.get("head") == head_tok.get("head") \
                    and str(tok.get("deprel", "")).startswith("flat") \
                    and sent.index(tok) > pos \
                    and tok.get("feats", {}).get("Case") == own_case:
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
    # eliptická otázka bez slovesa („Jaká rodina?") má sponovou sémantiku
    cop = cop or not any(t.get("upos") == "VERB" for t in sent)
    verb = next((_clean_lemma(t.get("lemma", "")) for t in sent
                 if t.get("upos") == "VERB"), None)
    hole_role = hole_type = None
    for tok in sent:                       # díra = tázací slovo
        low = _clean_lemma(tok.get("lemma", "")).lower()
        if low in _HOLE:
            fixed, hole_type = _HOLE[low]
            hole_role = fixed or _DEPREL_ROLE.get(tok.get("deprel"))
            break
        if tok.get("feats", {}).get("PronType") == "Int" \
                or low in current()["interrogative_adverbs"]:
            # nepodporované tázací slovo („proč", „jak") — otázka NENÍ
            # zjišťovací (jinak by „Proč přišel?" odpovědělo „Ano"); tagger
            # PronType u příslovcí often nedává → jazyková tabulka
            hole_role, hole_type = "theme", None
            break

    known = []
    if cop:
        # přísudkové jméno (root, ne tázací) = entita; s genitivní osobou = relace
        root = next((t for t in sent if t.get("deprel") == "root"
                     and t.get("upos") in ("NOUN", "PROPN")
                     and _clean_lemma(t.get("lemma", "")).lower() not in _HOLE), None)
        if root is None:
            # parser dal kořen tázacímu slovu („Co je robot?") — jmenný člen je
            # jinde ve větě; identita se přesto složí kanonicky (jinak by
            # predicate=None matchoval cokoli a odpovídal šumem)
            root = next((t for t in sent if t.get("upos") in ("NOUN", "PROPN")
                         and _clean_lemma(t.get("lemma", "")).lower() not in _HOLE),
                        None)
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
        if low in _HOLE or low in _DATE_PARTS or tok.get("upos") not in _CONTENT:
            continue                       # „v kterém ROCE" je drill, ne účastník
        if tok.get("deprel") in ("flat", "flat:name"):
            continue                       # část víceslovné entity, řeší _entity_term
        role = _DEPREL_ROLE.get(tok.get("deprel"))
        if role:
            known.append((role, _known_of(tok, sent)))   # rekurzivně (vnořené pod-dotazy)
            tok_id = sent.index(tok) + 1
            for child in sent:
                # předložkové VLASTNÍ JMÉNO pod známým („bratr … S KARLEM
                # ČAPKEM") je vlastní účastník; obecné jméno („Válku
                # S MLOKY") je fráze titulu a zůstává uvnitř termínu
                if child.get("head") == tok_id and child.get("upos") == "PROPN" \
                        and str(child.get("deprel", "")).startswith("nmod") \
                        and any(c.get("head") == sent.index(child) + 1
                                and c.get("upos") == "ADP" for c in sent):
                    known.append(("theme", _entity_term(child, sent)))
    return Pattern(verb, known, hole_role, hole_type, date_part)
