"""Fasáda Jelly — tenký „composition root" nad porty.

Není to monolit: složí **výchozí** bloky (retriever, korpus, answerer) a nechá je
**injektovat/vyměnit** (i za NN). Vlastní životní cyklus korpusu, umí debug a
pojmenované session. Kdo chce vnitřek, jde granulárně přes bloky/porty.
"""

from jellyai.logs import get_logger, set_debug as _set_debug
from jellyai.errors import JellyError
from jellyai.session import save_session, load_session


class Jelly:
    """Rychlá cesta i composition root pro skládačku portů."""

    def __init__(self, config=None, *, debug=False, retriever=None,
                 answerer=None, corpus=None):
        """Vytvoří fasádu s výchozími nebo injektovanými bloky.

        Args:
            config (Config | None): Konfigurace (default `Config()`).
            debug (bool): Zapne ladicí výpis.
            retriever/answerer/corpus: Injektované porty (None = výchozí bloky).
        """
        from config import Config
        self.config = config or Config()
        self.log = get_logger()
        self.debug = False
        self.set_debug(debug)
        self._retriever = retriever
        self._answerer = answerer
        self.corpus = corpus
        self._graph = None

    def set_debug(self, on):
        """Zapne/vypne ladicí výpis knihovny."""
        self.debug = on
        _set_debug(on)

    def ask(self, question, *, debug=None, temperature=0.0):
        """Odpoví na dotaz vybraným answererem; loguje, když debug.

        Args:
            question (str): Dotaz.
            debug (bool | None): Přebije globální debug pro tento dotaz.
            temperature (float): Teplota shody (0 = nejlepší, 1 = široce).

        Returns:
            Answer: Odpověď (s trasou a případnými alternativami).
        """
        if debug is not None:
            self.set_debug(debug)
        self.log.debug("otázka: %s (temperature=%s)", question, temperature)
        answerer = self._require_answerer()
        try:
            return answerer.answer(question, [], temperature=temperature)
        except TypeError:
            return answerer.answer(question, [])   # answerer bez parametru temperature

    def _require_answerer(self):
        """Vrátí answerer (injektovaný), nebo srozumitelnou chybu."""
        if self._answerer is None:
            raise JellyError("Answerer není nastavený — předej `Jelly(answerer=…)` "
                             "nebo použij granulární bloky.")
        return self._answerer

    def gravity(self):
        """Aktuální těžiště konverzace (nejteplejší uzel), nebo None."""
        context = getattr(self._answerer, "context", None)
        return context.hottest() if context is not None else None

    def trajectory(self):
        """Historie konverzace (trajektorie těžiště)."""
        return list(getattr(self._answerer, "history", []))

    def reset(self):
        """Začne nový rozhovor (vymaže těžiště a historii answereru)."""
        if hasattr(self._answerer, "reset"):
            self._answerer.reset()

    def save_session(self, name):
        """Uloží pojmenovanou session (váhy + historie)."""
        return save_session(name, self._answerer,
                            graph_path=self.config.graph.graph_path)

    def load_session(self, name):
        """Načte pojmenovanou session (pokračuje od posledních vah)."""
        return load_session(name, self._answerer)

    def close(self):
        """Složí korpusové služby, pokud běží."""
        if self.corpus is not None and hasattr(self.corpus, "stop"):
            self.corpus.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
