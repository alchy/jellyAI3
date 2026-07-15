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


class UfalTagger:
    """Reálný tagger nad ÚFAL NameTag (rozpoznávání pojmenovaných entit).

    Pozn.: záměrně **jen NameTag**, ne MorphoDiTa. `ufal.morphodita` a
    `ufal.nametag` jsou dva SWIG moduly sdílející C++ typ `std::vector<std::string>`
    a v jednom procesu se perou — ten použitý druhý spadne. NameTag sám pokryje
    naše typy odpovědí: osoby/místa/instituce/čas (Kdo/Kde/Co/Kdy) i číselné
    entity (Kolik). MorphoDiTa (lemmatizace) se dá vrátit později přes izolaci do
    subprocesu, ale pro V2a ji nepotřebujeme.

    Model se načte při vytvoření. Analýza probíhá po jedné větě, znakové offsety
    jsou relativní k předané větě (shodně s FakeTagger, aby byl blok zaměnitelný).
    Import ÚFAL je schválně uvnitř metod, aby šel modul naimportovat i bez
    nainstalovaného ÚFAL (kvůli hermetickým testům zbytku pipeline).

    Args:
        nametag_model (str): Cesta k modelu NameTag (.ner).

    Raises:
        FileNotFoundError: Když se model nepodaří načíst.
    """

    def __init__(self, nametag_model):
        from ufal.nametag import Ner
        self._ner = Ner.load(nametag_model)
        if self._ner is None:
            raise FileNotFoundError(f"Nelze načíst NameTag model: {nametag_model}")

    def tokens(self, text):
        """NameTag POS/lemma neposkytuje — vrací prázdný seznam.

        Čísla (Kolik) chodí u UfalTaggeru z číselných entit (viz answers.py),
        ne z POS. Metoda tu je kvůli dodržení rozhraní Tagger.

        Args:
            text (str): Věta (nepoužije se).

        Returns:
            list: Vždy prázdný seznam.
        """
        return []

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
