"""05 — Vyměň blok (dependency injection do fasády).

Fasáda Jelly je jen tenké dráty: každý port (retriever, answerer, korpus) jde
injektovat vlastní implementací — sem později sedne i NN.
Spusť: python examples/05_swap_a_block.py
"""

import jellyai
from jellyai.answerer.base import Answer


class YesAnswerer:
    """Ukázkový vlastní port answereru — vždy řekne „ano"."""

    context = None
    history = []

    def answer(self, question, retrieved, **kwargs):
        return Answer(text="ano", sources=[], score=1.0)


jelly = jellyai.Jelly(answerer=YesAnswerer())
print(jelly.ask("funguje injektování?").text)   # → ano
