"""Klient Iris služby — web i CLI dotazují automat VÝHRADNĚ přes REST.

Jeden vstupní bod (spec §5): klient se nejdřív zkusí připojit na UŽ BĚŽÍCÍ
službu (typicky nastartovanou ručně nebo jiným procesem — nepřebírá její
životní cyklus); když neběží, nastartuje vlastní subprocess jako ÚFAL služby
(`_ServiceHandle`) a na konci ho složí.
"""

import json
import urllib.request

from jellyai.buildinfo import git_sha
from jellyai.ufal_client import _ServiceHandle, _post


def version_warning(local, remote):
    """Neshoda verzí klient × služba (#40) — text varování, nebo None.

    Křičí se jen při SKUTEČNÉ neshodě dvou známých SHA; „unknown"
    (mimo git) poctivě nesoudí.
    """
    if "unknown" in (local, remote) or local == remote:
        return None
    return (f"⚠️  VERZE NESOUHLASÍ: služba Iris běží na {remote}, "
            f"klient je {local} — služba drží starý kód, restartuj ji "
            f"(kill :8084 + start, viz HANDOVER §1.7).")


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
                self._handshake()          # verzovací handshake (#40)
                return
        except Exception:  # noqa: BLE001 — neběží → nastartujeme vlastní
            pass
        self._handle = _ServiceHandle(
            "services/iris_service.py", self.graph_path,
            self.config.host, self.config.iris_port,
            self.config.startup_timeout)

    def _handshake(self):
        """Zaloguje verzi PŘIPOJENÉ (cizí) instance; při neshodě křičí —
        třída deploy-bolesti „napojeno na starou instanci" (#40)."""
        url = f"http://{self.config.host}:{self.config.iris_port}/version"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                remote = json.load(resp)
        except Exception:  # noqa: BLE001 — starší build bez /version
            print("[iris] připojeno na běžící službu BEZ /version "
                  "(starší build?) — zvaž restart", flush=True)
            return
        print(f"[iris] připojeno na službu {remote.get('sha')} "
              f"(start {remote.get('started')})", flush=True)
        warning = version_warning(git_sha(), remote.get("sha", "unknown"))
        if warning:
            print(warning, flush=True)

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
