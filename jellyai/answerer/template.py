"""Pravidlový answerer — odpověď složí z retrievalu, rolí a šablon (V3).

Sešívá dohromady všechno předchozí: otázku rozebere (typ + sloveso), v nalezené
anotované pasáži vybere správnou entitu (podle typu a role), převede ji do
kanonického tvaru a vloží do šablony. Fakta jsou z textu, gramatika ze šablon,
tvary z lemmat/MorphoDiTy — nic si nevymýšlí. Když nic nesedí, poctivě spadne na
extraktivní answerer (radši věta z textu než mlčení).
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.selection import select_answer, _clean_lemma
from jellyai import templates

# tázací slovo → typ otázky
_QWORDS = {"kdo": "Kdo", "co": "Co", "kde": "Kde", "kdy": "Kdy", "kolik": "Kolik"}


def _analyze_question(question, client):
    """Z otázky zjistí typ (Kdo/Co/…) a lemma hlavního slovesa.

    Args:
        question (str): Dotaz uživatele.
        client: ÚFAL klient (kvůli syntaktickému rozboru otázky).

    Returns:
        tuple[str|None, str|None]: (typ otázky, lemma slovesa).
    """
    qtype, verb_lemma = None, None
    for sentence in client.parse(question):
        for tok in sentence:
            low = tok.get("form", "").lower()
            if qtype is None and low in _QWORDS:
                qtype = _QWORDS[low]
            if verb_lemma is None and tok.get("upos") == "VERB":
                verb_lemma = _clean_lemma(tok.get("lemma", ""))
    return qtype, verb_lemma


class TemplateAnswerer(Answerer):
    """Answerer skládající odpověď pravidly (retrieval + role + šablona)."""

    def __init__(self, client, annotations, fallback):
        """Vytvoří answerer.

        Args:
            client: ÚFAL klient (rozbor otázky, případně skloňování).
            annotations (dict): (doc_id, index) → anotace pasáže (z `annotate`).
            fallback (Answerer): Answerer pro případ, že nic nesedí (extraktivní).
        """
        self.client = client
        self.annotations = annotations or {}
        self.fallback = fallback

    def _render(self, qtype, candidate):
        """Převede kandidáta do cílového tvaru a vloží do šablony."""
        case = templates.target_case(qtype)
        if case is None:
            answer = candidate.form           # data/čísla beze změny
        else:
            answer = candidate.lemma          # nominativ = lemma (V1)
        return templates.fill(qtype, answer)

    def answer(self, question, retrieved):
        """Složí odpověď; když to nejde, deleguje na fallback.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list[tuple[Passage, float]]): Pasáže a skóre z retrieveru.

        Returns:
            Answer: Odpověď + zdroj, nebo výsledek fallbacku.
        """
        if not retrieved:
            return self.fallback.answer(question, retrieved)
        qtype, verb_lemma = _analyze_question(question, self.client)
        if qtype is None:
            return self.fallback.answer(question, retrieved)
        for passage, score in retrieved:
            annotation = self.annotations.get((passage.doc_id, passage.index))
            if not annotation:
                continue
            candidate = select_answer(qtype, verb_lemma, annotation)
            if candidate is None:
                continue
            text = self._render(qtype, candidate)
            if text.strip():
                return Answer(text=text, sources=[f"{passage.doc_id}#{passage.index}"],
                              score=float(score))
        return self.fallback.answer(question, retrieved)


def explain():
    """Vrátí lidský popis bloku TemplateAnswerer pro výukovou vrstvu.

    Returns:
        str: Popis pravidlového skládání odpovědi.
    """
    return (
        "TemplateAnswerer rozebere otázku (typ + sloveso), v nalezené anotované "
        "pasáži vybere entitu podle typu (NameTag) a role (UDPipe podmět/předmět), "
        "převede ji do kanonického tvaru a vloží do šablony. Fakta z textu, "
        "gramatika ze šablon. Když nic nesedí, spadne na extraktivní answerer."
    )
