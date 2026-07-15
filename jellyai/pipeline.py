"""Pipeline — pospojování bloků do jednoho celku „otázka → odpověď".

Jednotlivé bloky (loader, chunker, retriever, answerer) jsou schválně malé a
nezávislé. Pipeline je dirigent: postaví je do řady a předává mezi nimi data,
takže uživatel má jediné jednoduché rozhraní `ask(otázka)` a nemusí řešit, co se
děje uvnitř. Vyměnit answerer za jiný (třeba generativní ve V2) znamená vyměnit
jeden argument — zbytek stojí, jak stál.
"""

from jellyai.loader import load_documents
from jellyai.chunker import chunk
from jellyai.retriever import Retriever
from jellyai.answerer.extractive import ExtractiveAnswerer


class QAPipeline:
    """Spojí Retriever a Answerer do jednoho QA celku."""

    def __init__(self, retriever, answerer):
        """Vytvoří pipeline z hotového retrieveru a answereru.

        Args:
            retriever (Retriever): Už postavený index pasáží.
            answerer (Answerer): Blok skládající odpověď (např. ExtractiveAnswerer).
        """
        self.retriever = retriever
        self.answerer = answerer

    def ask(self, question):
        """Odpoví na dotaz: vyhledá pasáže a nechá answerer složit odpověď.

        Args:
            question (str): Dotaz uživatele v češtině.

        Returns:
            Answer: Odpověď se zdroji a skóre.
        """
        retrieved = self.retriever.search(question)
        return self.answerer.answer(question, retrieved)

    @classmethod
    def from_corpus(cls, directory, config):
        """Postaví celou pipeline z adresáře s vyčištěnými texty.

        Pohodlná tovární metoda: načte dokumenty, rozseká je na pasáže, postaví
        nad nimi retriever a připojí extraktivní answerer — vše podle konfigurace.
        Ušetří volajícímu ruční skládání čtyř bloků.

        Args:
            directory (str): Adresář s vyčištěnými `.txt` (typicky data/processed).
            config (Config): Konfigurace všech bloků.

        Returns:
            QAPipeline: Připravená pipeline s postaveným indexem.
        """
        docs = load_documents(directory)
        passages = []
        for doc in docs:
            passages.extend(chunk(doc, config.chunker))
        retriever = Retriever(config.retriever).build(passages)
        answerer = ExtractiveAnswerer(config.answerer)
        return cls(retriever, answerer)


def explain():
    """Vrátí lidský popis bloku Pipeline pro výukovou vrstvu.

    Returns:
        str: Několikavětý popis role pipeline a metody from_corpus.
    """
    return (
        "QAPipeline spojí Retriever a Answerer do jednoho celku: ask(otázka) "
        "vyhledá relevantní pasáže a nechá Answerer složit odpověď. "
        "from_corpus() postaví celou pipeline z adresáře s vyčištěnými texty."
    )
