"""Porty knihovny — úzká rozhraní (Protocol), kam sedne i neuronová síť.

Primární abstrakce nejsou fasády, ale malé porty: každá fáze pipeline má úzký
kontrakt. Rozhraní jsou **strukturální** (`typing.Protocol`), takže stávající bloky
je splňují bez dědičnosti — a pozdější NN implementace stačí, když má stejné metody.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Rozdělí text na tokeny."""
    def tokenize(self, text: str) -> list: ...


@runtime_checkable
class QuestionAnalyzer(Protocol):
    """Rozebere otázku (typ, téma, sloveso…)."""
    def analyze(self, question: str): ...


@runtime_checkable
class FactExtractor(Protocol):
    """Z anotace věty vytáhne fakty."""
    def extract(self, annotation: dict) -> list: ...


@runtime_checkable
class Composer(Protocol):
    """Ze sady faktů složí čitelný text (víc než jednoslovná odpověď)."""
    def compose(self, question: str, facts: list) -> str: ...


@runtime_checkable
class CorpusPort(Protocol):
    """České korpusové nástroje (rozbor/entity/morfologie)."""
    def parse(self, text: str) -> list: ...
    def entities(self, text: str) -> list: ...
    def analyze(self, text: str) -> list: ...
    def generate(self, lemma: str, tag: str) -> list: ...
