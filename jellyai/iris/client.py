"""Klient Iris služby — web i CLI dotazují automat VÝHRADNĚ přes REST.

Jeden vstupní bod (spec §5): klient se nejdřív zkusí připojit na UŽ BĚŽÍCÍ
službu (typicky nastartovanou ručně nebo jiným procesem — nepřebírá její
životní cyklus); když neběží, nastartuje vlastní subprocess jako ÚFAL služby
(`_ServiceHandle`) a na konci ho složí.
"""

import json
import urllib.request

from jellyai.ufal_client import _ServiceHandle, _post


class IrisClient:
    """HTTP klient k `services/iris_service.py` (líné připojení/start)."""

    def __init__(self, services_config, graph_path):
        """Vytvoří klienta.

        Args:
            services_config (ServicesConfig): Adresa, port `iris_port`, timeout.
            graph_path (str): Cesta ke grafu (--model při vlastním startu).
        """
        self.config = services_config
        self.graph_path = graph_path
        self._handle = None
        self._connected = False

    def _ensure(self):
        """Připojí se na běžící službu, nebo ji líně nastartuje."""
        if self._connected or self._handle is not None:
            return
        url = f"http://{self.config.host}:{self.config.iris_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=1):
                self._connected = True     # cizí instance — životní cyklus její
                return
        except Exception:  # noqa: BLE001 — neběží → nastartujeme vlastní
            pass
        self._handle = _ServiceHandle(
            "services/iris_service.py", self.graph_path,
            self.config.host, self.config.iris_port,
            self.config.startup_timeout)

    def query(self, question, temperature=0.0):
        """POST /query — jeden tah konverzace (odpověď/dialog + metadata)."""
        self._ensure()
        return _post(self.config.host, self.config.iris_port, "/query",
                     {"question": question, "temperature": temperature})

    def reset(self):
        """POST /reset — nový rozhovor (vymaže dialog i aktivační pole)."""
        self._ensure()
        return _post(self.config.host, self.config.iris_port, "/reset", {})

    def schema(self):
        """GET /schema — publikovaný slovník jazyka a karet."""
        self._ensure()
        url = f"http://{self.config.host}:{self.config.iris_port}/schema"
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.load(resp)

    def close(self):
        """Složí vlastní službu (cizí běžící instance se nedotýká)."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
