"""Adaptér `GraphView` nad viewBase — jediné místo, kde se viewBase importuje.

Líný import (jádro zůstává viewBase-free). Vlastní **životní cyklus webserveru**:
`serve()` nastartuje, `stop()` složí; context manager. Prompt řeší přes viewBase
`ControlWindow` (textové pole + on_submit). Graf jde stavět/měnit v kódu
(`add_node`/`update_node`/`flow`) i naplnit z `FactGraph` (`from_graph`).
"""

from jellyai.errors import JellyError


def _fact_id(fact_node):
    """Krátké čitelné id faktového uzlu (predikát + otisk klíče)."""
    return "fact:" + fact_node.predicate + ":" + str(abs(hash(fact_node.id)) % 100000)


class ViewBaseView:
    """viewBase adaptér: build/modify v kódu + prompt + živé aktualizace."""

    def __init__(self, title="jellyAI3"):
        """Vytvoří plátno viewBase (líný import).

        Args:
            title (str): Titulek okna.

        Raises:
            JellyError: Když viewBase není nainstalovaný (akční hláška).
        """
        try:
            import viewbase as vb
        except ImportError as exc:
            raise JellyError(
                "viewBase není nainstalovaný — nainstaluj ho: pip install viewbase "
                "(je volitelný, jádro jellyAI3 ho nepotřebuje).") from exc
        self._vb = vb
        self._canvas = vb.Canvas(title=title)
        self._handle = None

    def from_graph(self, graph):
        """Naplní plátno uzly a hranami faktového grafu (entity + faktové uzly).

        Args:
            graph (FactGraph): Zdrojový graf.

        Returns:
            ViewBaseView: self (pro řetězení).
        """
        for node in graph.nodes.values():
            self.add_node(node.id, label=node.id, kind=node.type)
        for fact in graph.facts.values():
            fid = _fact_id(fact)
            self.add_node(fid, label=fact.predicate, kind="fact")
            for participant in fact.participants:
                self.add_edge(fid, participant.node, role=participant.role)
        return self

    def add_node(self, node_id, **meta):
        """Přidá uzel (metadata → barva/label ve viewBase)."""
        self._canvas.add_node(node_id, **meta)

    def add_edge(self, src, dst, **meta):
        """Přidá hranu."""
        self._canvas.add_edge(src, dst, **meta)

    def update_node(self, node_id, **attrs):
        """Živě změní uzel (velikost/barva/label) — push přes WebSocket."""
        self._canvas.update_node(node_id, **attrs)

    def flow(self, path):
        """Animuje světelné částice po cestě uzlů (trasa dotazu)."""
        if len(path) >= 2:
            self._canvas.flow(path[0], path[-1], path=path)

    def on_prompt(self, callback):
        """Napojí textový prompt (ControlWindow) na callback(text)."""
        window = self._vb.ControlWindow()
        window.string("dotaz")
        self._canvas.open_window(
            window, on_submit=lambda values: callback(values["dotaz"]))

    def serve(self, open_browser=True):
        """Nastartuje webserver (neblokující); uloží handle.

        Returns:
            ViewBaseView: self.
        """
        self._handle = self._vb.serve(self._canvas, open_browser=open_browser,
                                      block=False)
        return self

    def stop(self):
        """Složí webserver."""
        if self._handle is not None:
            self._handle.stop()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
