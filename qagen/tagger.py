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
