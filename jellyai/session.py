"""Perzistence pojmenované konverzace — JSON (čitelné pro uživatele).

Stav rozhovoru = váhy těžiště (`ActivationField`) + historie + odkaz na graf. Uloží
se jako `data/sessions/<name>.json`, takže se dá pojmenovat, načíst a **pokračovat
od posledních vah**. Graf zůstává zvlášť (velký, numpy) — session ho jen referuje.
"""

import json
import os

from jellyai.graph.activation import ActivationField

_DIR = "data/sessions"


def _path(name, directory):
    """Sestaví cestu k souboru session."""
    return os.path.join(directory or _DIR, f"{name}.json")


def save_session(name, answerer, graph_path=None, directory=None):
    """Uloží stav konverzace answereru do JSON.

    Args:
        name (str): Jméno session.
        answerer: Objekt s `.context` (ActivationField) a `.history`.
        graph_path (str | None): Cesta ke grafu, který session používá.
        directory (str | None): Cílový adresář (default data/sessions).

    Returns:
        str: Cesta k uloženému souboru.
    """
    path = _path(name, directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"name": name, "graph_path": graph_path,
            "weights": answerer.context.to_dict(), "history": answerer.history}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def load_session(name, answerer, directory=None):
    """Načte stav konverzace do answereru (pokračuje od posledních vah).

    Args:
        name (str): Jméno session.
        answerer: Objekt s `.context` a `.history` (přepíší se).
        directory (str | None): Adresář (default data/sessions).

    Returns:
        dict: Načtená data session (vč. `graph_path`).
    """
    with open(_path(name, directory), encoding="utf-8") as handle:
        data = json.load(handle)
    answerer.context = ActivationField.from_dict(data["weights"])
    answerer.history = data["history"]
    return data
