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


def _to_nominative(phrase, client):
    """Převede odpovědní frázi do 1. pádu se zachováním rodu/čísla.

    Naivní „vezmi lemma" rozbíjí u víceslovných jmen shodu („Božený Němcová").
    Tady místo toho každé skloňovatelné slovo analyzujeme MorphoDiTou, vezmeme jeho
    tag, přepneme pád na 1. a necháme MorphoDiTu vygenerovat správný tvar. Slova bez
    pádu (slovesa, číslice…) necháme být.

    Args:
        phrase (str): Odpovědní fráze, jak stojí v textu (často v šikmém pádě).
        client: ÚFAL klient (MorphoDiTa analyze + generate).

    Returns:
        str: Fráze v 1. pádě; při neúspěchu původní fráze.
    """
    tokens = client.analyze(phrase)
    if not tokens:
        return phrase
    out = []
    for tok in tokens:
        tag = tok.get("tag", "")
        lemma = tok.get("lemma", tok.get("form", ""))
        # skloňovatelné slovo (podst./příd. jméno, zájmeno, číslovka) v šikmém pádě
        if len(tag) >= 5 and tag[0] in "NAPC" and tag[4] in "234567":
            nom_tag = tag[:4] + "1" + tag[5:]
            forms = client.generate(lemma, nom_tag)
            out.append(forms[0] if forms else tok.get("form", ""))
        else:
            out.append(tok.get("form", ""))
    return " ".join(w for w in out if w).strip()


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
            answer = candidate.form                       # data/čísla beze změny
        elif case == "1":
            # skloň celou frázi do 1. pádu se shodou; fallback na lemma-join
            answer = _to_nominative(candidate.form, self.client) or candidate.lemma
        else:
            answer = candidate.lemma
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
