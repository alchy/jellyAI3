"""Adaptér `GraphView` nad viewBase — jediné místo, kde se viewBase importuje.

Líný import (jádro zůstává viewBase-free). Vlastní **životní cyklus webserveru**:
`serve()` nastartuje, `stop()` složí; context manager. Prompt řeší přes viewBase
`ControlWindow` (textové pole + on_submit). Graf jde stavět/měnit v kódu
(`add_node`/`update_node`/`flow`) i naplnit z `FactGraph` (`from_graph`).
"""

from jellyai.errors import JellyError
from jellyai.viz.detail import node_detail_rows, fact_detail_rows

# barvy typů uzlů (paleta jellyAI3) — viewBase vyžaduje define_type před add_node
_TYPE_STYLE = {
    "person": {"color": "#d1702a"},
    "geo": {"color": "#3f7d54"},
    "time": {"color": "#b0832f"},
    "number": {"color": "#b0832f"},
    "concept": {"color": "#43678a"},
    "institution": {"color": "#8a4a17"},
    "fact": {"color": "#8a949f", "size": 0.7},
}


class ViewBaseView:
    """viewBase adaptér: build/modify v kódu + prompt + živé aktualizace."""

    def __init__(self, title="jellyAI3", *, theme="cyber"):
        """Vytvoří plátno viewBase (líný import).

        Args:
            title (str): Titulek okna.
            theme (str | dict): Téma viewBase — výchozí „cyber" (tmavé); „modern"
                je světlé, nebo vlastní dict.

        Raises:
            JellyError: Když viewBase není nainstalovaný (akční hláška).
        """
        try:
            import viewbase as vb  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise JellyError(
                "viewBase není nainstalovaný — nainstaluj ho: pip install viewbase "
                "(je volitelný, jádro jellyAI3 ho nepotřebuje).") from exc
        self._vb = vb
        self._canvas = vb.Canvas(title=title, theme=theme)
        self._handle = None
        self._terminal_id = None

    def from_graph(self, graph):
        """Naplní plátno uzly a hranami faktového grafu (entity + faktové uzly).

        Args:
            graph (FactGraph): Zdrojový graf.

        Returns:
            ViewBaseView: self (pro řetězení).
        """
        self._canvas.detail_window()               # klik na uzel → box se všemi meta
        for kind in {node.type for node in graph.nodes.values()} | {"fact"}:
            self._canvas.define_type(kind, **_TYPE_STYLE.get(kind, {"color": "#8a949f"}))
        for node in graph.nodes.values():
            # meta = co o uzlu držíme (typ, váha, fakty) → naplní detailní okno
            self.add_node(node.id, label=node.id, type=node.type,
                          **dict(node_detail_rows(graph, node.id)))
        for index, fact in enumerate(graph.facts.values()):
            fid = f"fact:{fact.predicate}:{index}"     # unikátní id (bez kolizí)
            self.add_node(fid, label=fact.predicate, type="fact",
                          **dict(fact_detail_rows(fact)))
            for participant in fact.participants:
                self.add_edge(fid, participant.node, role=participant.role)
        return self

    def add_node(self, node_id, **meta):
        """Přidá/aktualizuje uzel (idempotentní)."""
        self._canvas.ensure_node(node_id, **meta)

    def add_edge(self, src, dst, **meta):
        """Přidá hranu (idempotentní — duplicity nevadí)."""
        self._canvas.ensure_edge(src, dst, **meta)

    def update_node(self, node_id, **attrs):
        """Živě změní uzel (velikost/barva/label) — push přes WebSocket."""
        self._canvas.update_node(node_id, **attrs)

    def flow(self, path):
        """Animuje světelné částice po cestě uzlů (trasa dotazu). Chyby nekritické."""
        if len(path) < 2:
            return
        try:
            self._canvas.flow(path[0], path[-1], path=path)
        except Exception:  # pylint: disable=broad-exception-caught
            pass          # když uzly nejsou hranou spojené, jen nic neanimujeme

    def on_prompt(self, callback):
        """Napojí textový prompt (ControlWindow) na callback(text)."""
        window = self._vb.ControlWindow("prompt", title="Dotaz")
        window.string("dotaz", "Dotaz", maxlength=200)
        self._canvas.open_window(
            window, on_submit=lambda values: callback(values.get("dotaz", "")))

    def open_terminal(self, on_input):
        """Otevře konzolové okno; on_input(řádek) dostane, co uživatel napsal.

        Odpověď se do konzole píše přes `write` — uživatel ji vidí v prohlížeči
        (ne na stdoutu serveru).
        """
        window = self._vb.TerminalWindow("konzole", title="Dotaz", prompt="❓ ")
        self._terminal_id = window.window_id
        self._canvas.open_terminal(
            window, on_input=lambda event: on_input(getattr(event, "line", "")))

    def write(self, text):
        """Připíše text do konzolového okna (musí být otevřené přes open_terminal)."""
        if self._terminal_id is not None:
            self._canvas.terminal_write(self._terminal_id, text)

    def serve(self, open_browser=True, block=True):
        """Nastartuje webserver. `block=True` drží proces (standalone), jinak handle.

        Args:
            open_browser (bool): Otevřít prohlížeč.
            block (bool): True = blokuje (drží server); False = vrátí a uloží handle.

        Returns:
            ViewBaseView: self.
        """
        self._handle = self._vb.serve(self._canvas, open_browser=open_browser,
                                      block=block)
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
