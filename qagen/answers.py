"""Výběr kandidátních odpovědí z věty a přiřazení typu otázky.

Odpověď na faktickou otázku bývá pojmenovaná entita (osoba, místo, instituce,
čas) nebo číslo. Tenhle modul takové spány z věty vytáhne a rovnou k nim přiřadí
tázací slovo — z „starý Rossum" (osoba) se stane odpověď na otázku typu „Kdo".
Jinými slovy: hledáme ve větě to nejzajímavější, na co se dá zeptat.
"""

from dataclasses import dataclass

# První písmeno CNEC typu → tázací slovo. (CNEC = Czech Named Entity Corpus.)
_ENTITY_QTYPE = {"p": "Kdo", "g": "Kde", "i": "Co", "t": "Kdy"}


@dataclass
class Candidate:
    """Kandidát na (odpověď, typ otázky) v rámci jedné věty.

    Atributy:
        answer (str): Text odpovědi (spán ve větě).
        qtype (str): Typ otázky (Kdo/Co/Kde/Kdy/Kolik).
        start (int): Znakový offset odpovědi ve větě.
        end (int): Znakový offset za koncem (exkluzivní).
    """
    answer: str
    qtype: str
    start: int
    end: int


def candidates(sentence, tagger, config):
    """Najde ve větě kandidátní odpovědi a jejich typy otázek.

    Z pojmenovaných entit (NameTag) odvodí Kdo/Co/Kde/Kdy, z číslovek
    (MorphoDiTa POS „C") odvodí Kolik. Vynechá odpovědi rovné celé větě (neměly by
    kontext), deduplikuje a omezí počet na `max_answers_per_sentence`.

    Args:
        sentence (str): Věta k analýze.
        tagger (Tagger): Zdroj entit a tokenů.
        config (QagenConfig): Nastavení (povolené typy, max odpovědí).

    Returns:
        list[Candidate]: Kandidáti v pořadí entity → čísla.
    """
    raw = []
    for e in tagger.entities(sentence):
        qtype = _ENTITY_QTYPE.get(e.type[:1].lower())
        if qtype:
            raw.append(Candidate(e.text, qtype, e.start, e.end))
    for t in tagger.tokens(sentence):
        if t.pos == "C":  # číslovka
            raw.append(Candidate(t.text, "Kolik", t.start, t.end))

    result = []
    seen = set()
    stripped = sentence.strip()
    for c in raw:
        if c.qtype not in config.types:
            continue
        if c.answer.strip() == stripped:  # odpověď = celá věta → nemá kontext
            continue
        key = (c.answer, c.start)
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
        if len(result) >= config.max_answers_per_sentence:
            break
    return result
