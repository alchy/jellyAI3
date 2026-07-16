"""Klient a správce životního cyklu ÚFAL služeb.

`ufal.nametag`, `ufal.udpipe` a `ufal.morphodita` se v jednom procesu perou
(sdílený SWIG typ `vector<string>`). Řešení: každý běží jako **samostatný proces**
s malým HTTP API na localhostu. Tenhle modul ty procesy líně nastartuje, počká,
až naběhnou (`/health`), mluví s nimi přes `urllib` a na konci je složí. Pro testy
existuje `FakeUfalClient`, který vrací nakonzervovaná data bez sítě a modelů.
"""

import atexit
import json
import subprocess
import sys
import time
import urllib.request

from jellyai.normalize import merge_abbreviations, expand_abbreviation_entities

# Název služby → skript, který ji spouští.
_SERVICE_SCRIPTS = {
    "nametag": "services/nametag_service.py",
    "udpipe": "services/udpipe_service.py",
    "morpho": "services/morpho_service.py",
}


def _post(host, port, path, payload):
    """Pošle JSON POST na službu a vrátí rozparsovanou odpověď.

    Args:
        host (str): Adresa služby.
        port (int): Port služby.
        path (str): Endpoint (např. "/entities").
        payload (dict): Tělo požadavku.

    Returns:
        dict: Rozparsovaná JSON odpověď.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{host}:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


class _ServiceHandle:
    """Nastartuje službu jako subprocess a pohlídá její životní cyklus.

    Args:
        script (str): Cesta ke skriptu služby.
        model (str): Cesta k modelu, který si služba načte.
        host (str): Adresa (jen localhost).
        port (int): Port služby.
        timeout (float): Max. čekání na naběhnutí (s).

    Raises:
        RuntimeError: Když služba do timeoutu nenaběhne nebo spadne.
    """

    def __init__(self, script, model, host, port, timeout):
        self.port = port
        self.proc = subprocess.Popen(
            [sys.executable, script, "--port", str(port), "--model", model]
        )
        atexit.register(self.close)
        self._wait_health(host, port, timeout)

    def _wait_health(self, host, port, timeout):
        """Čeká, dokud služba neodpoví na /health (nebo dokud nevyprší timeout)."""
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Služba na portu {port} spadla (exit {self.proc.returncode})"
                )
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1):
                    return
            except Exception as exc:  # noqa: BLE001 - dokud nenaběhne, zkoušíme dál
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"Služba na portu {port} nenaběhla do {timeout}s ({last_error})")

    def close(self):
        """Ukončí proces služby (nejdřív slušně, pak natvrdo)."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()


class UfalClient:
    """HTTP klient k ÚFAL službám; služby startuje líně a drží po dobu procesu."""

    def __init__(self, config):
        """Vytvoří klienta nad konfigurací služeb.

        Args:
            config (ServicesConfig): Adresa, porty, cesty modelů, timeout.
        """
        self.config = config
        self._handles = {}

    def _ensure(self, name, port, model):
        """Zajistí, že služba `name` běží (líně ji spustí při první potřebě)."""
        if name not in self._handles:
            self._handles[name] = _ServiceHandle(
                _SERVICE_SCRIPTS[name], model, self.config.host, port,
                self.config.startup_timeout,
            )
        return self._handles[name]

    def entities(self, text):
        """Vrátí pojmenované entity věty (NameTag, normalizované zkratky).

        Returns:
            list[dict]: [{text, type, start, end}].
        """
        self._ensure("nametag", self.config.nametag_port, self.config.nametag_model)
        raw = _post(self.config.host, self.config.nametag_port,
                    "/entities", {"text": text})["entities"]
        return expand_abbreviation_entities(text, raw)

    def parse(self, text):
        """Vrátí syntaktický rozbor (UDPipe) s normalizovanými zkratkami.

        Returns:
            list[list[dict]]: Věty; token = {form, lemma, upos, head, deprel, start, end}.
        """
        self._ensure("udpipe", self.config.udpipe_port, self.config.udpipe_model)
        raw = _post(self.config.host, self.config.udpipe_port,
                    "/parse", {"text": text})["sentences"]
        return merge_abbreviations(raw)

    def analyze(self, text):
        """Vrátí morfologickou analýzu (MorphoDiTa): tokeny s lemma+tag.

        Returns:
            list[dict]: [{form, lemma, tag}].
        """
        self._ensure("morpho", self.config.morpho_port, self.config.morphodita_model)
        return _post(self.config.host, self.config.morpho_port,
                     "/analyze", {"text": text})["tokens"]

    def generate(self, lemma, tag):
        """Vygeneruje tvary lemmatu odpovídající tagu (MorphoDiTa skloňování).

        Args:
            lemma (str): Základní tvar.
            tag (str): Cílový (i wildcard) PDT tag.

        Returns:
            list[str]: Vygenerované tvary.
        """
        self._ensure("morpho", self.config.morpho_port, self.config.morphodita_model)
        return _post(self.config.host, self.config.morpho_port,
                     "/generate", {"lemma": lemma, "tag": tag})["forms"]

    def close(self):
        """Složí všechny spuštěné služby."""
        for handle in self._handles.values():
            handle.close()
        self._handles = {}


class FakeUfalClient:
    """Testovací klient — vrací nakonzervovaná data podle vstupu, bez sítě/modelů.

    Args:
        entities (dict|None): text → list entit.
        parse (dict|None): text → list vět (token dictů).
        analyze (dict|None): text → list tokenů.
        generate (dict|None): (lemma, tag) nebo lemma → list tvarů.
    """

    def __init__(self, entities=None, parse=None, analyze=None, generate=None):
        self._entities = entities or {}
        self._parse = parse or {}
        self._analyze = analyze or {}
        self._generate = generate or {}

    def entities(self, text):
        return self._entities.get(text, [])

    def parse(self, text):
        return self._parse.get(text, [])

    def analyze(self, text):
        return self._analyze.get(text, [])

    def generate(self, lemma, tag):
        if (lemma, tag) in self._generate:
            return self._generate[(lemma, tag)]
        return self._generate.get(lemma, [lemma])

    def close(self):
        pass
