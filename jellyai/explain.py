"""Výuková vrstva — jedno místo, kde se každý blok umí představit.

Celá knihovna má být i učební pomůcka, takže sem sbíráme popisy jednotlivých
bloků. Uživatel se přes `cli.py explain <blok>` doptá „a co dělá tohle?", aniž
by musel otevírat zdrojáky. Je to takový průvodce výstavou, který u každého
exponátu řekne pár vět — jen bez toho nudného hlasu.
"""

from jellyai import loader, chunker, retriever, pipeline
from jellyai.answerer import extractive, template

def _explain_graph():
    """Popis bloku faktového grafu (hlavní směr projektu)."""
    return (
        "Faktový graf: korpus se rozloží na REIFIKOVANÁ fakta — každá slovesná "
        "událost je uzel (predikát + váha opakování) s role-hranami na účastníky "
        "(podmět/předmět/čas/místo). Entity se kanonizují (pádové varianty → "
        "jeden uzel se základním tvarem, ostatní tvary jsou aliasy). Odpovídání "
        "je pak match neúplného faktu s dírou: otázka Kdo napsal Babičku? = "
        "napsat(?, Babička) → díra subj → Božena Němcová. Postav grafem: "
        "./jelly annotate && ./jelly graph; dotazy: ./jelly graph-ask.")


def _explain_iris():
    """Popis automatu Iris (zaostření aktivace + dialog)."""
    return (
        "Iris: stavový automat ZAOSTŘENÍ — cílem je rozsvítit správné uzly "
        "grafu (aktivace), odpověď z nich padá přirozeně. Chování řídí JSON "
        "pattern-karty (trigger→dialog→akce; logika nikdy fixně v kódu): při "
        "nejistotě (QueryAssurance pod prahem karty) se ptá — nabídne kandidáty "
        "a rozsvítí je, volba uživatele otázku přehraje zaostřenou. Subsystémy: "
        "Chronos (časová kotva: dnes/včera/tento měsíc → intervaly, hodinové "
        "otázky) a Mnemos (paměť: konstatování → časově ukotvený fakt v grafu, "
        "deník data/memory.jsonl přežívá restart). Běží jako REST služba "
        "(:8084 — /query, /graphql, /schema); web ./jelly web s ní mluví "
        "výhradně přes API (tři okna: dialog / aktivace uzlů / dokumenty).")


# Mapa název bloku -> funkce, která vrátí jeho popis. Přidat nový blok = přidat
# sem jeden řádek.
_BLOCKS = {
    "loader": loader.explain,
    "chunker": chunker.explain,
    "retriever": retriever.explain,
    "answerer": extractive.explain,
    "template": template.explain,
    "pipeline": pipeline.explain,
    "graph": _explain_graph,
    "iris": _explain_iris,
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
