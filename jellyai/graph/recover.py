"""Role ② neighbor-spreadingu — doplnění chybějících titulů do grafu.

Z reálných anotací najde **horké spany, které NER minul** (`entity_candidates`
nad autoregresním spreadem) a přidá je jako uzly typu **„dílo"** propojené
s podmětem věty přes „work" sloveso (napsat/vydat…). Tím se do grafu dostane
i titul, který tokenizace/NER neustála — a otázka „Kdo napsal <titul>?" pak
najde autora. Běží jako **post-pass** po `build_graph` (nezasahuje do extrakce).
"""

from jellyai.lang import current
from jellyai.graph.spread import entity_candidates
from jellyai.graph.extract import (make_fact, Participant, _SUBJ, _SKIP_UPOS,
                                   _node_for, _clean_lemma)


def _sentence_subject(sent, entities):
    """Podmět věty jako (id, typ) — osobní `nsubj`, jinak nejdelší osobní entita."""
    for tok in sent:
        if tok.get("deprel") in _SUBJ and tok.get("upos") not in _SKIP_UPOS:
            node = _node_for(tok, entities)
            if node is not None and node[1] == "person":
                return node
    persons = [e for e in entities if e.get("type", "")[:1].lower() == "p"]
    if persons:
        best = max(persons, key=lambda e: e.get("end", 0) - e.get("start", 0))
        return (best["text"], "person")
    return None


def _work_verb(sent, before_index):
    """Lemma „work" slovesa nalevo od titulu (to, které titul uvádí)."""
    verb = None
    for i, tok in enumerate(sent):
        if i >= before_index:
            break
        lemma = _clean_lemma(tok.get("lemma", ""))
        if tok.get("upos") == "VERB" and lemma in current()["work_verbs"]:
            verb = lemma
    return verb


def recover_entities(annotations, graph):
    """Doplní do grafu horké chybějící tituly jako uzly „dílo" + autorský fakt.

    Args:
        annotations (dict): Anotace ((doc_id, idx) → {entities, sentences}).
        graph (FactGraph): Graf k doplnění (mění se in-place).

    Returns:
        list[str]: Doplněné tituly (v pořadí přidání).
    """
    added = []
    for annotation in annotations.values():
        entities = annotation.get("entities", [])
        person_spans = [(e.get("start"), e.get("end")) for e in entities
                        if e.get("type", "")[:1].lower() == "p"]
        for sent in annotation.get("sentences", []):
            cands = entity_candidates(sent, set(graph.nodes))
            if not cands:
                continue
            subject = _sentence_subject(sent, entities)
            if subject is None:
                continue
            forms = [t.get("form", "") for t in sent]
            for title in cands:
                if title not in forms:
                    continue
                index = forms.index(title)
                if _in_person(sent[index], person_spans):
                    continue          # skloňované jméno uvnitř osobní entity, ne titul
                verb = _work_verb(sent, index)
                if verb is None:
                    continue
                graph.add_fact(make_fact(verb, [
                    Participant("subj", subject[0], subject[1]),
                    Participant("obj", title, "dílo")]))
                added.append(title)
    return added


def _in_person(tok, person_spans):
    """True, když token leží uvnitř nějaké osobní entity (skloňované jméno)."""
    start, end = tok.get("start"), tok.get("end")
    if start is None or end is None:
        return False
    return any(ps is not None and ps <= start and end <= pe
               for ps, pe in person_spans)
