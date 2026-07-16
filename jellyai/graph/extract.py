"""Extrakce reifikovaných faktů z větné anotace.

Každá slovesná událost se stane jedním **faktem** (Fact) s predikátem a účastníky
v rolích (podmět/předmět/čas/místo). Fakt je pozdější uzel grafu — reifikace vztahu.
Uzel účastníka je pojmenovaná entita (kanonicky) nebo nominativní lemma tokenu.
"""

from dataclasses import dataclass

from jellyai.answerer.selection import _clean_lemma
from jellyai.graph.canon import name_gender
from jellyai.lang import current

_SUBJ = {"nsubj", "nsubj:pass"}
_OBJ = {"obj", "iobj"}
_ATTR = {"obl", "nmod"}
_ENTITY_TYPE = {"p": "person", "g": "geo", "t": "time", "i": "institution"}
_ATTR_ROLE = {"time": "time", "geo": "loc", "number": "num"}   # typ cíle → role
_SKIP_UPOS = {"PRON", "DET"}   # zájmena/určovatele = balast (a záminka pro pro-drop)


@dataclass(frozen=True)
class Participant:
    """Účastník faktu v konkrétní roli.

    Atributy:
        role (str): Role (subj/obj/time/loc/num/pred).
        node (str): Id uzlu účastníka.
        type (str): Typ uzlu (person/geo/time/number/concept/institution).
    """
    role: str
    node: str
    type: str


@dataclass(frozen=True)
class Fact:
    """Reifikovaná událost — pozdější faktový uzel.

    Atributy:
        predicate (str): Predikát (lemma slovesa, nebo „být" u spony).
        participants (tuple): Seřazená n-tice Participant (určuje identitu faktu).
    """
    predicate: str
    participants: tuple


def make_fact(predicate, participants):
    """Sestaví `Fact` s deterministicky seřazenými účastníky (kvůli identitě).

    Args:
        predicate (str): Predikát faktu.
        participants (list[Participant]): Účastníci (libovolné pořadí).

    Returns:
        Fact: Fakt s n-ticí účastníků seřazenou podle (role, node).
    """
    return Fact(predicate, tuple(sorted(participants, key=lambda p: (p.role, p.node))))


def _entity_type(entity):
    """CNEC typ entity (první písmeno) → typ uzlu."""
    return _ENTITY_TYPE.get(entity.get("type", "")[:1].lower(), "concept")


def _node_for(token, entities, canon=None):
    """Vrátí (id, typ) uzlu pro token: entita (kanonicky) nebo nominativní lemma.

    U osobních entit se id sjednotí přes `canon` (fragment „Karel" → „Karel Čapek"),
    aby se fakta téže osoby nerozpadla na víc uzlů.

    Args:
        token (dict): Token s start/end/lemma/upos.
        entities (list[dict]): Entity věty s offsety.
        canon (dict | None): Mapa osobní jméno → kanonický (nejdelší) tvar.

    Returns:
        tuple[str, str] | None: (id, typ), nebo None když token nemá lemma.
    """
    start, end = token.get("start"), token.get("end")
    if start is not None and end is not None:
        best = None
        for e in entities:
            es, ee = e.get("start"), e.get("end")
            if es is not None and es <= start and end <= ee:
                # pylint: disable=unsubscriptable-object
                if best is None or (ee - es) > (best["end"] - best["start"]):
                    best = e            # nejdelší obklopující entita (celé „13. ledna 1890")
        if best is not None:
            text, typ = best["text"], _entity_type(best)
            if typ == "person" and canon:
                text = canon.get(text, text)
            return text, typ
    lemma = _clean_lemma(token.get("lemma", ""))
    if not lemma:
        return None
    return lemma, ("number" if token.get("upos") == "NUM" else "concept")


def _children(sent, head_id):
    """Tokeny věty s `head` == head_id (1-based)."""
    return [t for t in sent if t.get("head") == head_id]


def _pronoun_person(tok, context):
    """Anafora: osobní zájmeno 3. osoby → nejteplejší rodově shodná osoba.

    Zobecněný pro-drop: rod zájmena (feats `Gender`) se páruje s rodem jména
    (`name_gender` — tvar příjmení, jazyková data) přes kandidáty aktivačního
    pole (nejteplejší první). Demonstrativa (`PronType=Dem`, „to"), reflexiva
    (`Reflex=Yes`, „se") a 1./2. osoba osobu nevážou nikdy.

    Args:
        tok (dict | None): Zájmenný token (s feats).
        context (list[tuple] | None): Kandidáti [(id, typ), …] dle jasu.

    Returns:
        tuple | None: (id, typ) navázané osoby, nebo None.
    """
    if tok is None:
        return None
    feats = tok.get("feats", {})
    if feats.get("PronType") != "Prs" or feats.get("Reflex") == "Yes" \
            or feats.get("Person") not in (None, "3"):
        return None
    gender = feats.get("Gender")
    if gender not in ("Masc", "Fem"):
        return None
    for candidate in context or ():
        if candidate[1] == "person" and name_gender(candidate[0]) == gender:
            return candidate
    return None


def _relation_person(children, sent, entities, canon):
    """Osoba v HOLÉM genitivním přívlastku („bratr **Karla Čapka**").

    Osoba s předložkou („drama **o Karlu Čapkovi**") vztah/autorství nenese —
    je to „o kom", ne „čí". Holý genitiv poznáme podle chybějícího `case`
    (ADP) dítěte u nmod tokenu.
    """
    for tok in children:
        if tok.get("deprel", "").startswith("nmod") and tok.get("upos") not in _SKIP_UPOS:
            tok_id = sent.index(tok) + 1
            if any(c.get("head") == tok_id and c.get("upos") == "ADP" for c in sent):
                continue
            node = _node_for(tok, entities, canon)
            if node is not None and node[1] == "person":
                return node
    return None


def _first(tokens, deprels):
    """První token s `deprel` z množiny, nebo None."""
    return next((t for t in tokens if t.get("deprel") in deprels), None)


def _verb_head(index, sent):
    """Vrátí 1-based id nejbližšího slovesa-předka tokenu (i sebe), nebo None.

    Zanořený atribut (datum/číslo pod přívlastkem) se tak přiřadí slovesu, které ho
    skutečně řídí — a v souvětí zůstane u svého slovesa, ne u sousedního.

    Args:
        index (int): 0-based index tokenu ve větě.
        sent (list[dict]): Věta (tokeny s `head`, 1-based; 0 = kořen).

    Returns:
        int | None: 1-based id slovesa-předka, nebo None.
    """
    seen = set()
    cur = index
    while 0 <= cur < len(sent) and cur not in seen:
        seen.add(cur)
        if sent[cur].get("upos") == "VERB":
            return cur + 1
        head = sent[cur].get("head", 0)
        if not head:
            return None
        cur = head - 1
    return None


def _copular_facts(sent, head_id, subj_tok, subj, entities, canon):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Fakty sponové věty „X je Y" — univerzální reifikace genitivu + identita.

    Y s osobním genitivem → `Y(X, osoba)` pro **libovolné** Y (bratr, drama,
    román…): vztah je struktura věty, ne položka slovníku. Žánrové Y
    (`work_nouns` z jazykových dat) přidá autorství `napsat(osoba, X)`. Vždy
    vznikne i identita/vlastnost `být(X, Y)` („Kdo/Jaký je X?"). Kompenzuje
    parser-quirk, kdy kořenem spony je adjektivum („je satirický sci-fi román
    Karla Čapka") a jmenný přísudek visí jako druhý nsubj za sponou — slovosled
    dělí podmět|přísudek.

    Args:
        sent (list[dict]): Věta (tokeny).
        head_id (int): 1-based id hlavy sponové věty.
        subj_tok (dict | None): Token skutečného podmětu (kvůli odlišení).
        subj (tuple | None): (id, typ) podmětu (i pro-drop náhrada).
        entities (list[dict]): Entity věty.
        canon (dict | None): Kanonizace osobních jmen.

    Returns:
        list[Fact]: Fakty věty (reifikovaný vztah, autorství, identita).
    """
    head = sent[head_id - 1]
    children = _children(sent, head_id)
    rel_head, rel_children = head, children
    if head.get("upos") == "ADJ":
        cop_tok = _first(children, {"cop"})
        nominal = next(
            (t for t in children
             if t.get("deprel") in _SUBJ and t.get("upos") == "NOUN"
             and t is not subj_tok and sent.index(t) > sent.index(cop_tok)),
            None)
        if nominal is not None:
            rel_head = nominal
            rel_children = _children(sent, sent.index(nominal) + 1)
    facts = []
    rel = _clean_lemma(rel_head.get("lemma", ""))
    other = _relation_person(rel_children, sent, entities, canon)
    if subj and rel and other is not None and rel_head.get("upos") == "NOUN":
        facts.append(make_fact(rel, [
            Participant("subj", subj[0], subj[1]),
            Participant("obj", other[0], other[1]),
        ]))
        if rel in current()["work_nouns"]:
            facts.append(make_fact("napsat", [
                Participant("subj", other[0], other[1]),
                Participant("obj", subj[0], subj[1]),
            ]))
    pred = _node_for(rel_head, entities, canon)
    if pred and subj:
        role = "attr" if rel_head.get("upos") == "ADJ" else "pred"
        facts.append(make_fact("být", [
            Participant("subj", subj[0], subj[1]),
            Participant(role, pred[0], pred[1]),
        ]))
    return facts


def extract_facts(annotation, default_subject=None, canon=None, context=None):
    # pylint: disable=too-many-locals,too-many-branches
    """Vytáhne z anotace věty seznam reifikovaných faktů.

    Pro každý sloveso-token vznikne jeden n-ární fakt (podmět + předmět +
    atributy). **Elidovaný podmět** (pro-drop: „Narodil se 1890") dostane
    `default_subject` (hlavní entita dokumentu). **Osobní zájmeno** v roli
    podmětu/předmětu se rozváže anaforou (`_pronoun_person`) na rodově shodnou
    osobu z `context`; demonstrativa a nerozvázaná zájmena osobu NEdědí —
    overtní podmět není elize. Fakt bez dalšího účastníka než podmět se zahazuje.

    Args:
        annotation (dict): {"entities": [...], "sentences": [[token,...],...]}.
        default_subject (tuple | None): (id, typ) náhradního podmětu pro pro-drop.
        canon (dict | None): Kanonizace osobních jmen (sjednocení fragmentů).
        context (list[tuple] | None): Kandidáti anafory [(id, typ), …] dle jasu.

    Returns:
        list[Fact]: Nalezené fakty (mohou se opakovat — agreguje graf).
    """
    facts = []
    entities = annotation.get("entities", [])
    for sent in annotation.get("sentences", []):
        # atributy (čas/místo/číslo/téma) — i zanořené — přiřaď nejbližšímu
        # slovesu-předku
        attrs_by_verb = {}
        for i, tok in enumerate(sent):
            node = _node_for(tok, entities, canon)
            if node is None:
                continue
            role = _ATTR_ROLE.get(node[1])
            if role is None:
                # konceptové příslovečné určení („uvažovat o souvislosti")
                # se nezahazuje — role „theme"; jen obl substantiva
                if node[1] == "concept" and tok.get("upos") in ("NOUN", "PROPN") \
                        and str(tok.get("deprel", "")).startswith("obl"):
                    role = "theme"
                else:
                    continue
            vh = _verb_head(i, sent)
            if vh is not None:
                attrs_by_verb.setdefault(vh, set()).add(Participant(role, node[0], node[1]))

        for head_id in range(1, len(sent) + 1):
            head = sent[head_id - 1]
            children = _children(sent, head_id)
            subj_tok = _first(children, _SUBJ)
            pronoun_subj = (subj_tok is not None
                            and subj_tok.get("upos") in _SKIP_UPOS)
            subj_node = None
            if subj_tok is not None and not pronoun_subj:
                subj_node = _node_for(subj_tok, entities, canon)
            elif pronoun_subj:
                subj_node = _pronoun_person(subj_tok, context)   # anafora on/ona

            # sponová věta: (podmět)–být–(přísudek). Rozlišíme **identitu**
            # (podstatné jméno → role „pred": „je spisovatelka") od **vlastnosti/
            # stavu** (přídavné jméno → role „attr": „je nemocná") — „Kdo je"
            # čerpá z identity, „Jaký je" z vlastnosti, takže se nepletou.
            if _first(children, {"cop"}):
                # overtní zájmenný podmět NENÍ pro-drop elize: buď se rozváže
                # anaforou (on/ona), nebo („Je TO lepra") osobu nedědí vůbec
                subj = subj_node or (None if pronoun_subj else default_subject)
                facts.extend(_copular_facts(sent, head_id, subj_tok, subj,
                                            entities, canon))
                continue

            if head.get("upos") != "VERB":
                continue
            verb = _clean_lemma(head.get("lemma", ""))
            extra = []
            obj = _first(children, _OBJ)
            if obj is not None:
                o = (_node_for(obj, entities, canon)
                     if obj.get("upos") not in _SKIP_UPOS
                     else _pronoun_person(obj, context))    # anafora „ji/mu"
                if o:
                    extra.append(Participant("obj", o[0], o[1]))
            extra.extend(sorted(attrs_by_verb.get(head_id, ()), key=lambda p: (p.role, p.node)))
            if not extra:
                continue
            # nerozvázaný overtní zájmenný podmět („To vedlo…") osobu nedědí
            subj = subj_node or (None if pronoun_subj else default_subject)
            if subj is None:
                continue
            facts.append(make_fact(verb, [Participant("subj", subj[0], subj[1])] + extra))
    return facts
