"""Logování knihovny — správně: default ticho (NullHandler), debug přidá výstup.

Knihovna nemá konfigurovat globální logging hostitele. Proto logger „jellyai" má
`NullHandler` (mlčí), a `set_debug(True)` mu přidá `StreamHandler` a zapne DEBUG —
uživatel pak vidí, co knihovna dělá.
"""

import logging

_LOGGER_NAME = "jellyai"
_debug_handler = None


def get_logger():
    """Vrátí logger knihovny (s NullHandlerem, default ticho)."""
    log = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in log.handlers):
        log.addHandler(logging.NullHandler())
    return log


def set_debug(on):
    """Zapne/vypne ladicí výpis knihovny.

    Args:
        on (bool): True přidá StreamHandler + DEBUG; False ho odebere.
    """
    global _debug_handler
    log = get_logger()
    if on and _debug_handler is None:
        _debug_handler = logging.StreamHandler()
        _debug_handler.setFormatter(logging.Formatter("jellyai: %(message)s"))
        log.addHandler(_debug_handler)
        log.setLevel(logging.DEBUG)
    elif not on and _debug_handler is not None:
        log.removeHandler(_debug_handler)
        _debug_handler = None
        log.setLevel(logging.WARNING)
