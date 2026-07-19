"""Identita buildu — git SHA pro verzovací handshake služeb (BACKLOG #40).

Třída deploy-bolesti „napojeno na starou instanci": služba drží starý kód
i graf a web se tiše připojí. Handshake: služba SHA vystaví (/version),
klient ho při připojení porovná se svým a při neshodě křičí.
"""

import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_sha():
    """Krátké SHA HEAD repa; „unknown" mimo git (tarball nasazení).

    Returns:
        str: Např. „2234e49", nebo „unknown".
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=_ROOT, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"
