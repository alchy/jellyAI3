"""Živé propojení answereru s grafovou vizualizací (GraphView).

Po každém dotazu promítne stav answereru do view: **aktivaci nodů** (konverzační
těžiště → velikost) a **trasu** dotazu (téma → hodnota) jako `flow` po hranách.
Uživatel tak vidí, jak se graf rozsvěcuje a jak dotaz běží.
"""


def _trace_path(trace):
    """Z trasy poslední odpovědi sestaví cestu uzlů pro `flow` (téma → hodnota)."""
    path = [trace.get("topic"), trace.get("answer")]
    return [node for node in path if node]


def reflect(view, answerer):
    """Promítne stav answereru (těžiště + trasu) do grafové vizualizace.

    Args:
        view (GraphView): Cílová vizualizace.
        answerer: Objekt s `.context` (ActivationField) a `.turn.trace`.
    """
    context = getattr(answerer, "context", None)
    if context is not None:
        for node_id, weight in context.scores.items():
            view.update_node(node_id, size=1.0 + weight)
    turn = getattr(answerer, "turn", None)
    trace = turn.trace if turn is not None else None
    if trace:
        path = _trace_path(trace)
        if len(path) >= 2:
            view.flow(path)
