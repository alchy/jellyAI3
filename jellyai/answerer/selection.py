"""Výběr odpovědní entity z anotované pasáže podle typu otázky a role.

Tady se láme „netrefí vedle". NameTag řekne *typ* (osoba/místo/čas), UDPipe *roli*
(podmět `nsubj`, předmět `obj`). Když je ve větě víc osob, vybereme tu, která je
**podmětem slovesa z otázky** — ne jen „nějakou osobu poblíž". Nominativní tvar
odpovědi bereme z **lemmat** (lemma podstatného jména je 1. pád), takže „Babičku"
→ „Babička" bez dalšího skloňování.
"""

from dataclasses import dataclass

_PERSON, _GEO, _TIME = "p", "g", "t"
_SUBJECT = {"nsubj", "nsubj:pass"}
_OBJECT = {"obj", "iobj", "dobj"}


@dataclass
class Candidate:
    """Vybraná odpověď.

    Atributy:
        form (str): Povrchový tvar odpovědi, jak stojí v textu.
        lemma (str): Nominativní (základní) tvar — spojená lemmata.
        qtype (str): Typ otázky.
    """
    form: str
    lemma: str
    qtype: str


def _clean_lemma(lemma):
    """Ořízne technické přípony lemmatu (např. „robot-2", „Praha_;G")."""
    for sep in ("-", "_", "`"):
        i = lemma.find(sep)
        if i > 0:
            lemma = lemma[:i]
    return lemma


def _tokens_in_span(entity, tokens):
    """Vrátí tokeny věty, které leží uvnitř znakového rozsahu entity."""
    return [t for t in tokens
            if t.get("start") is not None
            and t["start"] >= entity["start"] and t["end"] <= entity["end"]]


def _nominative(tokens):
    """Spojí lemmata tokenů do nominativního tvaru."""
    return " ".join(_clean_lemma(t["lemma"]) for t in tokens)


def _entities_of(annotation, first_letter):
    """Vrátí entity daného CNEC typu (podle prvního písmene)."""
    return [e for e in annotation.get("entities", [])
            if e.get("type", "")[:1].lower() == first_letter]


def _entity_candidate(entity, sentence, qtype):
    """Sestaví kandidáta z entity (form = text entity, lemma = nominativ)."""
    toks = _tokens_in_span(entity, sentence)
    lemma = _nominative(toks) if toks else entity["text"]
    return Candidate(form=entity["text"], lemma=lemma or entity["text"], qtype=qtype)


def _sentence_of(sentences, entity):
    """Najde větu, do níž entita spadá (podle rozsahu tokenů)."""
    for sent in sentences:
        if _tokens_in_span(entity, sent):
            return sent
    return sentences[0] if sentences else []


def _select_subject_entity(annotation, verb_lemma, qtype):
    """Kdo: osoba, přednostně podmět slovesa z otázky; jinak první osoba."""
    persons = _entities_of(annotation, _PERSON)
    if not persons:
        return None
    fallback = None
    for sent in annotation.get("sentences", []):
        for tok in sent:
            if tok.get("deprel") not in _SUBJECT:
                continue
            entity = next((e for e in persons if _tokens_in_span(e, [tok])), None)
            if not entity:
                continue
            cand = _entity_candidate(entity, sent, qtype)
            head = tok.get("head", 0)
            if verb_lemma and 0 < head <= len(sent) \
                    and _clean_lemma(sent[head - 1]["lemma"]) == verb_lemma:
                return cand            # nejlepší: podmět právě toho slovesa
            fallback = fallback or cand
    if fallback:
        return fallback
    entity = persons[0]
    return _entity_candidate(entity, _sentence_of(annotation["sentences"], entity), qtype)


def _select_object(annotation, verb_lemma, qtype):
    """Co: předmět slovesa z otázky (nemusí být pojmenovaná entita)."""
    fallback = None
    for sent in annotation.get("sentences", []):
        for tok in sent:
            if tok.get("deprel") not in _OBJECT:
                continue
            cand = Candidate(form=tok["form"], lemma=_clean_lemma(tok["lemma"]), qtype=qtype)
            head = tok.get("head", 0)
            if verb_lemma and 0 < head <= len(sent) \
                    and _clean_lemma(sent[head - 1]["lemma"]) == verb_lemma:
                return cand
            fallback = fallback or cand
    return fallback


def _select_entity(annotation, first_letter, qtype):
    """Kde/Kdy: první entita daného typu (místo/čas)."""
    ents = _entities_of(annotation, first_letter)
    if not ents:
        return None
    entity = ents[0]
    return _entity_candidate(entity, _sentence_of(annotation["sentences"], entity), qtype)


def _select_number(annotation, qtype):
    """Kolik: první číslovka (UPOS NUM)."""
    for sent in annotation.get("sentences", []):
        for tok in sent:
            if tok.get("upos") == "NUM":
                return Candidate(form=tok["form"], lemma=tok["form"], qtype=qtype)
    return None


def select_answer(qtype, verb_lemma, annotation):
    """Vybere kandidáta na odpověď podle typu otázky a rolí v anotaci.

    Args:
        qtype (str): Typ otázky (Kdo/Co/Kde/Kdy/Kolik).
        verb_lemma (str | None): Lemma hlavního slovesa otázky (pro výběr role).
        annotation (dict): Anotace pasáže {"entities": ..., "sentences": ...}.

    Returns:
        Candidate | None: Vybraná odpověď, nebo None když nic nesedí.
    """
    if qtype == "Kdo":
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
