"""Presenter — zaostřená data ven, žádná forma odpovědi.

Cíl Iris je kvalita AKTIVOVANÝCH DAT, ne stylizace textu (tu řeší budoucí
vrstva). Presenter proto jen čte aktivační pole answereru a vrací seřazené
seznamy: aktivační okno uzlů (pro UI okno 2) a aktivní dokumenty (okno 3).
Řazení je deterministické (jas sestupně, pak abecedně).
"""


def activation_window(answerer, limit=12):
    """Aktivační okno: uzly seřazené podle jasu (největší nahoře).

    Args:
        answerer (GraphAnswerer): Zdroj konverzačního pole (`context.scores`).
        limit (int): Nejvíc řádků okna.

    Returns:
        list[tuple[str, float]]: [(id uzlu, jas)], jas sestupně.
    """
    scores = answerer.context.scores
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(node, round(jas, 4)) for node, jas in ranked[:limit]]


def docs_window(answerer, limit=5):
    """Aktivní dokumenty: attention nad zdroji, seřazené sestupně."""
    scores = answerer.source_context.scores
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc, round(jas, 4)) for doc, jas in ranked[:limit]]
