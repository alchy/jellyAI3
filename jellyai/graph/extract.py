"""Extrakce reifikovaných faktů z větné anotace.

Každá slovesná událost se stane jedním **faktem** (Fact) s predikátem a účastníky
v rolích (podmět/předmět/čas/místo). Fakt je pozdější uzel grafu — reifikace vztahu.
Uzel účastníka je pojmenovaná entita (kanonicky) nebo nominativní lemma tokenu.
"""

from dataclasses import dataclass

from jellyai.answerer.selection import _clean_lemma

_SUBJ = {"nsubj", "nsubj:pass"}
_OBJ = {"obj", "iobj"}
_ATTR = {"obl", "nmod"}
_ENTITY_TYPE = {"p": "person", "g": "geo", "t": "time", "i": "institution"}
_ATTR_ROLE = {"time": "time", "geo": "loc", "number": "num"}   # typ cíle → role


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


def _node_for(token, entities):
    """Vrátí (id, typ) uzlu pro token: entita (kanonicky) nebo nominativní lemma.

    Args:
        token (dict): Token s start/end/lemma/upos.
        entities (list[dict]): Entity věty s offsety.

    Returns:
        tuple[str, str] | None: (id, typ), nebo None když token nemá lemma.
    """
    start, end = token.get("start"), token.get("end")
    if start is not None and end is not None:
        for e in entities:
            if e.get("start") is not None and e["start"] <= start and end <= e["end"]:
                return e["text"], _entity_type(e)
    lemma = _clean_lemma(token.get("lemma", ""))
    if not lemma:
        return None
    return lemma, ("number" if token.get("upos") == "NUM" else "concept")


def _children(sent, head_id):
    """Tokeny věty s `head` == head_id (1-based)."""
    return [t for t in sent if t.get("head") == head_id]


def _first(tokens, deprels):
    """První token s `deprel` z množiny, nebo None."""
    return next((t for t in tokens if t.get("deprel") in deprels), None)


def extract_facts(annotation):
    """Vytáhne z anotace věty seznam reifikovaných faktů.

    Pro každý sloveso-token s podmětem vznikne jeden n-ární fakt (podmět + předmět +
    atributy). Fakt bez dalšího účastníka než podmět se zahazuje.

    Args:
        annotation (dict): {"entities": [...], "sentences": [[token,...],...]}.

    Returns:
        list[Fact]: Nalezené fakty (mohou se opakovat — agreguje graf).
    """
    facts = []
    entities = annotation.get("entities", [])
    for sent in annotation.get("sentences", []):
        for head_id in range(1, len(sent) + 1):
            head = sent[head_id - 1]
            children = _children(sent, head_id)
            subj = _first(children, _SUBJ)
            if subj is None:
                continue
            subj_node = _node_for(subj, entities)
            if subj_node is None:
                continue
            if head.get("upos") != "VERB":
                continue
            verb = _clean_lemma(head.get("lemma", ""))
            parts = [Participant("subj", subj_node[0], subj_node[1])]
            obj = _first(children, _OBJ)
            if obj:
                o = _node_for(obj, entities)
                if o:
                    parts.append(Participant("obj", o[0], o[1]))
            if len(parts) > 1:
                facts.append(make_fact(verb, parts))
    return facts
