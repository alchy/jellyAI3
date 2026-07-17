"""Iris — stavový automat pro ZAOSTŘENÍ aktivace uzlů grafu.

Jméno podle clony oka/objektivu: Iris řídí, kolik světla (aktivace) a KAM
dopadne. Cíl automatu je jediný — správné uzly mají svítit, špatné ne;
kvalitní odpověď z dobře zaostřeného pole pak padá přirozeně. Forma odpovědi
(kompozice textu) sem NEpatří — tu řeší jiná, budoucí vrstva.

Stavba (viz spec `docs/superpowers/specs/2026-07-17-ql-automat.md`):

* `patterns`  — pattern-karty: chování automatu jako data (1 JSON = 1 vzor);
* `assurance` — QueryAssurance: číselná jistota zaostření (řídí přechody);
* `automaton` — jádro: tah uživatele → odpověď NEBO dialogové doostření;
* `state`     — FocusState: aktivační pole + rozpracovaný dialog;
* `presenter` — zaostřená data (seřazené uzly) + metadata, žádná forma.

Princip „dialog > figly": pod prahem jistoty se automat PTÁ (a rozsvěcí
kandidáty), místo aby hádal další heuristikou.
"""

from jellyai.iris.patterns import PatternDeck, PatternCard
from jellyai.iris.assurance import assurance
from jellyai.iris.automaton import IrisAutomaton, IrisResponse
from jellyai.iris.subsystems.chronos import TimeInterval, resolve_temporal

__all__ = ["PatternDeck", "PatternCard", "assurance",
           "IrisAutomaton", "IrisResponse",
           "TimeInterval", "resolve_temporal"]
