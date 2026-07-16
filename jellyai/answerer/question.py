"""Bohatá analýza otázky — společný předek pro celý další rozvoj.

Z jednoho UDPipe rozboru otázky vytáhne všechno, co answerer (i budoucí konverzační
aktivace) potřebuje: typ otázky **podle lemmatu** (takže „Jaká/Jaké/Jakého" spadnou
pod „Jaký"), lemma hlavního slovesa, jestli je otázka **sponová** („X je/byl Y") a
obsahová témata. Rozpoznávání přes lemma je klíč — díky němu přestanou tvary
tázacích slov padat na extraktivní.
"""

from dataclasses import dataclass, field

from jellyai.answerer.selection import _clean_lemma

# lemma tázacího slova → typ otázky (varianty tvarů řeší lemmatizace)
_QTYPE_BY_LEMMA = {
    "kdo": "Kdo", "co": "Co",
    "kde": "Kde", "kam": "Kde", "odkud": "Kde",
    "kdy": "Kdy", "kolik": "Kolik",
    "jaký": "Jaký", "který": "Který", "čí": "Čí",
}


@dataclass
class QuestionAnalysis:
    """Výsledek rozboru otázky.

    Atributy:
        qtype (str | None): Typ otázky (Kdo/Co/Kde/Kdy/Kolik/Jaký/Který/Čí).
        verb_lemma (str | None): Lemma hlavního slovesa (pro výběr role).
        is_copula (bool): Zda je otázka sponová („X je/byl Y").
        topic_terms (list): Obsahová lemmata (pro budoucí konverzační aktivaci).
    """
    qtype: str = None
    verb_lemma: str = None
    is_copula: bool = False
    topic_terms: list = field(default_factory=list)


def analyze_question(question, client):
    """Rozebere otázku a vrátí typ, sloveso, sponu a témata.

    Args:
        question (str): Dotaz uživatele.
        client: ÚFAL klient (UDPipe rozbor).

    Returns:
        QuestionAnalysis: Struktura s rozborem.
    """
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
