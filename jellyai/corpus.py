"""Korpusové nástroje jako spravovatelná služba (start/stop).

Přehledné rozhraní nad ÚFAL službami (UDPipe/NameTag/MorphoDiTa): explicitní
životní cyklus (`start`/`stop`, context manager), aby bylo vidět, kdy služby žijí.
Vnitřně je to `UfalClient` (líný spawn + atexit) — jen s čitelnou správou.
"""

from jellyai.ufal_client import UfalClient

# název nástroje → (atribut portu v ServicesConfig, atribut cesty k modelu)
_TOOLS = {"nametag": ("nametag_port", "nametag_model"),
          "udpipe": ("udpipe_port", "udpipe_model"),
          "morpho": ("morpho_port", "morphodita_model")}


class CorpusTools(UfalClient):
    """ÚFAL nástroje se start/stop a jako context manager."""

    def start(self, *tools):
        """Explicitně nastartuje uvedené nástroje (jinak se spustí líně).

        Args:
            *tools (str): „udpipe" / „nametag" / „morpho"; prázdné = všechny.

        Returns:
            CorpusTools: self (pro řetězení).
        """
        for name in (tools or _TOOLS.keys()):
            port_attr, model_attr = _TOOLS[name]
            self._ensure(name, getattr(self.config, port_attr),
                         getattr(self.config, model_attr))
        return self

    def stop(self):
        """Složí všechny běžící služby (alias `close`)."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
