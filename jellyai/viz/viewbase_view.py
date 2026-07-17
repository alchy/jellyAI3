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
    "dílo": {"color": "#c74f7a", "size": 1.2},   # titul doplněný role ② (neighbor-spread)
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

    def from_graph(self, graph, labeler=None):
        """Naplní plátno uzly a hranami faktového grafu (entity + faktové uzly).

        Args:
            graph (FactGraph): Zdrojový graf.
            labeler (callable | None): Volitelná úprava titulku uzlu (např.
                nominativizace „Hronově"→„Hronov" morfologií); id zůstává.

        Returns:
            ViewBaseView: self (pro řetězení).
        """
        self._canvas.detail_window()               # klik na uzel → box se všemi meta
        for kind in {node.type for node in graph.nodes.values()} | {"fact"}:
            self._canvas.define_type(kind, **_TYPE_STYLE.get(kind, {"color": "#8a949f"}))
        for node in graph.nodes.values():
            label = labeler(node) if labeler else node.id
            # meta = co o uzlu držíme (typ, váha, fakty) → naplní detailní okno
            self.add_node(node.id, label=label, type=node.type,
                          **dict(node_detail_rows(graph, node.id)))
        for index, fact in enumerate(graph.facts.values()):
            fid = f"fact:{fact.predicate}:{index}"     # unikátní id (bez kolizí)
            # kontextová asociace není přímý vztah → titulек „souvislost"
            label = "souvislost" if fact.predicate == "kontext" else fact.predicate
            self.add_node(fid, label=label, type="fact",
                          **dict(fact_detail_rows(fact)))
            for participant in fact.participants:
                self.add_edge(fid, participant.node, role=participant.role)
        return self

    def focus(self, node_id):
        """Vystředí kameru na uzel (obdoba kliknutí) — např. na nejaktivnější."""
        try:
            self._canvas.focus(node_id)
        except Exception:  # pylint: disable=broad-exception-caught
            pass          # neznámý uzel kameru nehýbe

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
        """Animuje světelné částice po trase (topic → answer). viewBase si cestu
        mezi konci **sám najde** (BFS), takže stačí předat konce. Chyby nekritické."""
        if len(path) < 2:
            return
        try:
            self._canvas.flow(path[0], path[-1], count=3, speed=0.6)
        except Exception:  # pylint: disable=broad-exception-caught
            pass          # když mezi konci nevede cesta, jen nic neanimujeme

    def packet(self, path):
        """Vyšle jeden „paket" (částici) po trase (topic→answer); viewBase si cestu
        mezi konci sám najde (BFS). Diskrétní tok jako v packet-flow příkladu —
        barvu/velikost bere z tématu (aditivní blending → burst na trase září)."""
        if not path or len(path) < 2:
            return
        try:
            self._canvas.flow(path[0], path[-1], count=1)
        except Exception:  # pylint: disable=broad-exception-caught
            pass          # když mezi konci nevede cesta, paket zahodíme

    def every(self, seconds, callback):
        """Zaregistruje periodickou úlohu (viewBase `every`) — volat před `serve`."""
        self._canvas.every(seconds)(callback)

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

    def open_docs_panel(self):
        """Otevře druhé konzolové okno — živý panel nejaktivnějších dokumentů
        (attention nad soubory). Aktualizuje se `write_docs` po každém dotazu."""
        window = self._vb.TerminalWindow("dokumenty", title="📄 Aktivní dokumenty",
                                         prompt="")
        self._docs_id = window.window_id
        self._canvas.open_terminal(window, on_input=lambda event: None)

    def write_docs(self, ranked):
        """Vypíše do panelu top dokumenty s jejich aktivací (attention).

        Args:
            ranked (list[tuple[str, float]]): (dokument, jas) seřazené sestupně.
        """
        if getattr(self, "_docs_id", None) is None:
            return
        if not ranked:
            self._canvas.terminal_write(self._docs_id, "— (žádná aktivace)")
            return
        lines = [f"{i}. {doc:24} {'█' * round(score) or '·'} {score:.2f}"
                 for i, (doc, score) in enumerate(ranked, 1)]
        self._canvas.terminal_write(self._docs_id, "\n".join(lines))

    def open_nodes_panel(self):
        """Otevře AKTIVAČNÍ OKNO — seznam uzlů seřazený podle jasu (největší
        nahoře), bez dialogu. Aktualizuje se `write_nodes` po každém tahu."""
        window = self._vb.TerminalWindow("aktivace", title="⚡ Aktivační okno",
                                         prompt="")
        self._nodes_id = window.window_id
        self._canvas.open_terminal(window, on_input=lambda event: None)

    def write_nodes(self, ranked):
        """Vypíše do aktivačního okna uzly s jasem (sestupně).

        Args:
            ranked (list[tuple[str, float]]): (uzel, jas) seřazené sestupně.
        """
        if getattr(self, "_nodes_id", None) is None:
            return
        if not ranked:
            self._canvas.terminal_write(self._nodes_id, "— (žádná aktivace)")
            return
        lines = [f"{i:2}. {node:28} {'█' * min(10, round(score)) or '·'} "
                 f"{score:.2f}"
                 for i, (node, score) in enumerate(ranked, 1)]
        self._canvas.terminal_write(self._nodes_id, "\n".join(lines))

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
