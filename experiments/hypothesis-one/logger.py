#!/usr/bin/env python3
"""Sdílený logovací wrapper všech programových částí experimentu.

Zatím píše jen do konzole, ale VŽDY přes tento wrapper — díky tomu se dá výstup
později přesměrovat (do souboru, do služby) beze změny jediného volajícího.

Formát řádku:  "[s] DDMMYYHHMM : zpráva"
  s = úroveň:  i = info,  d = debug,  ! = error,  * = ostatní
  DDMMYYHHMM = den, měsíc, rok(2), hodina, minuta (razítko okamžiku logu)
"""
from datetime import datetime

_LEVELS = ("i", "d", "!", "*")


def logger(severity, message):
    """Zaloguje `message` na úrovni `severity` na konzoli v jednotném formátu.

    Neznámá úroveň spadne do „*" (ostatní), takže volání nikdy nespadne. Časové
    razítko se bere z okamžiku volání (lokální čas).

    Args:
        severity (str): "i" | "d" | "!" | "*".
        message (str): text zprávy.

    Příklad:
        >>> logger("i", "korpus hotov: 11856 vět")
        [i] 2207261431 : korpus hotov: 11856 vět
    """
    sev = severity if severity in _LEVELS else "*"
    stamp = datetime.now().strftime("%d%m%y%H%M")
    print(f"[{sev}] {stamp} : {message}", flush=True)


if __name__ == "__main__":
    for s in ("i", "d", "!", "*", "x"):
        logger(s, f"ukázka úrovně {s!r}")
