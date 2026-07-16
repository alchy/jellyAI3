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


def _make_answerer(config):
    """Vybere answerer podle `config.answerer.mode` (pluggable blok).

    Pro "template" (V3) sestaví TemplateAnswerer s ÚFAL klientem, načtenými
    anotacemi a extraktivním fallbackem. Importy V3 jsou líné, aby extraktivní
    cesta (V1) nezáležela na ÚFAL/HTTP vrstvě.

    Args:
        config (Config): Konfigurace s `answerer.mode`, `services`.

    Returns:
        Answerer: ExtractiveAnswerer, nebo TemplateAnswerer pro mode="template".
    """
    if config.answerer.mode == "template":
        import os
        from jellyai.answerer.template import TemplateAnswerer
        from jellyai.ufal_client import UfalClient
        from jellyai.annotate import load_annotations
        annotations = {}
        if os.path.exists(config.services.annotations_path):
            annotations = load_annotations(config.services.annotations_path)
        return TemplateAnswerer(UfalClient(config.services), annotations,
                                ExtractiveAnswerer(config.answerer))
    return ExtractiveAnswerer(config.answerer)


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
        answerer = _make_answerer(config)
        return cls(retriever, answerer)

    @classmethod
    def from_index(cls, index_path, config):
        """Postaví pipeline z už uloženého indexu — bez opětovné stavby.

        Protahovací zkratka k :meth:`from_corpus`: retriever se místo počítání
        načte hotový z disku (tam je ta drahá práce), answerer se doplní z
        konfigurace. Díky tomu interaktivní prompt naskočí okamžitě, i kdyby
        korpus narostl.

        Args:
            index_path (str): Cesta k uloženému indexu (viz Retriever.save).
            config (Config): Konfigurace (použije se pro answerer).

        Returns:
            QAPipeline: Připravená pipeline s načteným indexem.
        """
        retriever = Retriever.load(index_path)
        answerer = _make_answerer(config)
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
