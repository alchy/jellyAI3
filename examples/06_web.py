"""06 — Web: graf ve viewBase nahoře + prompt dole.

Při dotazu se do grafu živě promítne aktivace nodů (těžiště) a trasa (flow).
Potřebuje: pip install viewbase  a  ./jelly annotate && ./jelly graph
Spusť: python examples/06_web.py   (nebo ./jelly web)
"""

from config import Config
from cli import cmd_web

# terminál i web volají tutéž `ask`; tady spustíme web s viewBase
cmd_web(Config())
