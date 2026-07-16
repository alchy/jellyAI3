"""Výuková vrstva — jedno místo, kde se každý blok umí představit.

Celá knihovna má být i učební pomůcka, takže sem sbíráme popisy jednotlivých
bloků. Uživatel se přes `cli.py explain <blok>` doptá „a co dělá tohle?", aniž
by musel otevírat zdrojáky. Je to takový průvodce výstavou, který u každého
exponátu řekne pár vět — jen bez toho nudného hlasu.
"""

from jellyai import loader, chunker, retriever, pipeline
from jellyai.answerer import extractive, template

# Mapa název bloku -> funkce, která vrátí jeho popis. Přidat nový blok = přidat
# sem jeden řádek.
_BLOCKS = {
    "loader": loader.explain,
    "chunker": chunker.explain,
    "retriever": retriever.explain,
    "answerer": extractive.explain,
    "template": template.explain,
    "pipeline": pipeline.explain,
}


def explain_block(name):
    """Vrátí popis jednoho bloku podle jména.

    Args:
        name (str): Název bloku (jeden z :func:`list_blocks`).

    Returns:
        str: Několikavětý popis daného bloku.

    Raises:
        KeyError: Když blok s tímto jménem neexistuje; hláška nabídne dostupné.
    """
    if name not in _BLOCKS:
        raise KeyError(
            f"Neznámý blok: {name}. Dostupné: {', '.join(sorted(_BLOCKS))}"
        )
    return _BLOCKS[name]()


def list_blocks():
    """Vrátí seřazený seznam názvů bloků, které umí popsat sebe sama.

    Returns:
        list[str]: Názvy bloků abecedně (pro nápovědu a validaci vstupu).
    """
    return sorted(_BLOCKS)
