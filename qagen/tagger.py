"""Rozhraní pro morfologicko-entitní analýzu češtiny.

Celá QA pipeline potřebuje vědět o větě dvě věci: kde jsou pojmenované entity
(kdo/co/kde/kdy — kandidáti na odpovědi) a jaké mají slova slovní druh a lemma.
Tuto znalost dodává ÚFAL (NameTag + MorphoDiTa), ale schováváme ji za jednoduché
rozhraní `Tagger`. Díky tomu zbytek pipeline neví nic o konkrétní knihovně a jde
testovat s `FakeTagger` — bez stahování stovek MB modelů jen kvůli testům.

Reálná implementace `UfalTagger` přijde zvlášť; importy ÚFAL jsou v ní schválně
až uvnitř metod, aby se tenhle modul dal naimportovat i bez nainstalovaného ÚFAL.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Entity:
    """Pojmenovaná entita nalezená ve větě (kandidát na odpověď).

    Atributy:
        text (str): Doslovný text entity.
        type (str): CNEC typ; rozhoduje první písmeno (p=osoba, g=místo,
            i=instituce, t=čas).
        start (int): Znakový offset začátku ve větě.
        end (int): Znakový offset za koncem (exkluzivní).
    """
    text: str
    type: str
    start: int
    end: int


@dataclass
class Token:
    """Jedno slovo s morfologickou anotací.

    Atributy:
        text (str): Tvar slova ve větě.
        lemma (str): Základní tvar (lemma).
        pos (str): Slovní druh — první písmeno PDT tagu (N=podst., C=číslovka…).
        start (int): Znakový offset začátku ve větě.
        end (int): Znakový offset za koncem (exkluzivní).
    """
    text: str
    lemma: str
    pos: str
    start: int
    end: int


class Tagger(Protocol):
    """Kontrakt, který qagen potřebuje: entity a otagované tokeny věty.

    Je to strukturální rozhraní (Protocol) — stačí mít obě metody, dědit nemusíš.
    """

    def entities(self, text: str) -> list:
        """Vrátí pojmenované entity ve větě (list[Entity])."""
        ...

    def tokens(self, text: str) -> list:
        """Vrátí otagované tokeny věty (list[Token])."""
        ...


class FakeTagger:
    """Testovací tagger — vrací předem připravená data podle přesného textu.

    Umožňuje testovat celou pipeline deterministicky a offline, bez ÚFAL modelů.
    Neznámou větu považuje za „nic nenalezeno" a vrátí prázdný seznam.

    Args:
        entities (dict | None): mapa text věty → list[Entity].
        tokens (dict | None): mapa text věty → list[Token].
    """

    def __init__(self, entities=None, tokens=None):
        self._entities = entities or {}
        self._tokens = tokens or {}

    def entities(self, text):
        """Vrátí nakonzervované entity pro danou větu (nebo prázdný seznam)."""
        return self._entities.get(text, [])

    def tokens(self, text):
        """Vrátí nakonzervované tokeny pro danou větu (nebo prázdný seznam)."""
        return self._tokens.get(text, [])


def _pdt_pos(tag):
    """Vrátí slovní druh = první znak PDT tagu (N/A/C/V…).

    Args:
        tag (str): Plný poziční PDT tag (např. „NNIP1-----A----").

    Returns:
        str: První znak tagu, nebo prázdný řetězec pro prázdný tag.
    """
    return tag[:1] if tag else ""


class UfalTagger:
    """Reálný tagger nad ÚFAL MorphoDiTa (POS/lemma) a NameTag (NER).

    Modely se načtou při vytvoření. Analýza probíhá po jedné větě, znakové offsety
    jsou relativní k předané větě (shodně s FakeTagger, aby byl blok zaměnitelný).
    Importy ÚFAL jsou schválně uvnitř metod, aby šel modul naimportovat i bez
    nainstalovaného ÚFAL (kvůli hermetickým testům zbytku pipeline).

    Args:
        morphodita_model (str): Cesta k modelu MorphoDiTa (.tagger).
        nametag_model (str): Cesta k modelu NameTag (.ner).

    Raises:
        FileNotFoundError: Když se některý model nepodaří načíst.
    """

    def __init__(self, morphodita_model, nametag_model):
        from ufal.morphodita import Tagger as MorphoTagger
        from ufal.nametag import Ner
        self._morpho = MorphoTagger.load(morphodita_model)
        if self._morpho is None:
            raise FileNotFoundError(f"Nelze načíst MorphoDiTa model: {morphodita_model}")
        self._ner = Ner.load(nametag_model)
        if self._ner is None:
            raise FileNotFoundError(f"Nelze načíst NameTag model: {nametag_model}")

    def tokens(self, text):
        """Otaguje větu MorphoDiTou a vrátí tokeny s lemma + POS a offsety.

        Args:
            text (str): Věta k analýze.

        Returns:
            list[Token]: Tokeny věty v pořadí výskytu.
        """
        from ufal.morphodita import Forms, TaggedLemmas, TokenRanges
        forms, lemmas, ranges = Forms(), TaggedLemmas(), TokenRanges()
        tokenizer = self._morpho.newTokenizer()
        tokenizer.setText(text)
        out = []
        while tokenizer.nextSentence(forms, ranges):
            self._morpho.tag(forms, lemmas)
            for i in range(len(lemmas)):
                start = ranges[i].start
                length = ranges[i].length
                out.append(Token(
                    text=text[start:start + length],
                    lemma=lemmas[i].lemma,
                    pos=_pdt_pos(lemmas[i].tag),
                    start=start,
                    end=start + length,
                ))
        return out

    def entities(self, text):
        """Najde ve větě pojmenované entity NameTagem a vrátí je se offsety.

        Args:
            text (str): Věta k analýze.

        Returns:
            list[Entity]: Entity věty (text, CNEC typ, znakové offsety).
        """
        from ufal.nametag import Forms, TokenRanges, NamedEntities
        forms, ranges, ents = Forms(), TokenRanges(), NamedEntities()
        tokenizer = self._ner.newTokenizer()
        tokenizer.setText(text)
        out = []
        while tokenizer.nextSentence(forms, ranges):
            self._ner.recognize(forms, ents)
            for ent in ents:
                first = ent.start
                last = ent.start + ent.length - 1
                start = ranges[first].start
                end = ranges[last].start + ranges[last].length
                out.append(Entity(
                    text=text[start:end],
                    type=ent.type,
                    start=start,
                    end=end,
                ))
        return out
