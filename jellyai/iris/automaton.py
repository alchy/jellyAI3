"""Jádro automatu Iris: tah uživatele → odpověď, NEBO dialogové doostření.

Jeden tah (`turn`) projde vždy stejnou cestou:

1. **Rozpracovaný dialog?** Když minulý tah nabídl kandidáty a text vypadá
   jako volba, otázka se PŘEHRAJE zaostřená — nejednoznačný termín se nahradí
   vybraným kandidátem („Kdo je Čapek?" + volba „Josef Čapek" → „Kdo je Josef
   Čapek?"). Volba se navíc rozsvítí v poli (aktivace nese zaostření dál).
2. **Odpověď answereru** (plugin pseudo-QL) + **QueryAssurance** z evidence
   rozlišení: kvalita jmenných pater / rovnocenní soupeři / jas vítěze.
3. **Rozhodnutí**: jistá odpověď → vrať ji („answer"); odpověď postavená na
   hádání mezi rovnocennými kandidáty → událost `resolve.ambiguous` → karta
   s nabídkou zaostření („dialog"); žádná odpověď, ale rozlišený kandidát →
   událost `focus.low` → upřímný terminál s nejbližšími kandidáty; jinak
   poctivé „nenašel" fallbacku.

Chování NENÍ v kódu — o textech a akcích rozhodují pattern-karty
(`patterns/<jazyk>/*.json`); automat jen hlásí události a vykonává akce
karet (dialog > figly).
"""

import re

from dataclasses import dataclass, field

from jellyai.graph.canon import deaccent
from jellyai.iris.assurance import assurance
from jellyai.iris.patterns import PatternDeck
from jellyai.iris.presenter import activation_window, docs_window
from jellyai.iris.state import FocusState, PendingFocus

_PICK_WARMTH = 2.0        # jas vybraného kandidáta (volba = silné zaostření)


@dataclass
class IrisResponse:
    """Výsledek jednoho tahu automatu.

    Atributy:
        text (str): Věta pro dialogové okno (odpověď NEBO řeč automatu).
        kind (str): „answer" (odpověď z grafu/fallbacku) | „dialog" (řeč
            automatu — otázka na doostření či upřímný terminál).
        assurance (float): Jistota zaostření tahu (0–1).
        activation_window (list): [(uzel, jas)] sestupně — UI okno 2.
        docs_window (list): [(dokument, jas)] sestupně — UI okno 3.
        used (dict): Metadata: {"components": [...], "patterns": [...]} —
            které komponenty a pattern-karty se na tahu podílely.
        clarify (dict | None): U dialogu {"prompt", "candidates"}.
        trace (dict | None): Trasa grafu (téma → fakt → hodnota).
        sources (list): Zdroje odpovědi.
        alternatives (list): Fuzzy souvislosti (teplota > 0).
    """
    text: str
    kind: str
    assurance: float
    activation_window: list = field(default_factory=list)
    docs_window: list = field(default_factory=list)
    used: dict = field(default_factory=dict)
    clarify: dict = None
    trace: dict = None
    sources: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)


class IrisAutomaton:
    """Stavový automat zaostření nad jedním answererem (jedna konverzace)."""

    def __init__(self, answerer, deck=None, threshold=0.55):
        """Vytvoří automat.

        Args:
            answerer (GraphAnswerer): Odpovídací plugin (drží graf i pole).
            deck (PatternDeck | None): Balíček pattern-karet; None = vestavěné
                karty češtiny (`patterns/cs/`).
            threshold (float): Práh QueryAssurance — pod ním se vede dialog.
        """
        self.answerer = answerer
        if deck is None:
            deck = PatternDeck.for_language("cs")
            deck.load()
        self.deck = deck
        self.threshold = threshold
        self.state = FocusState()

    def reset(self):
        """Nový rozhovor: vymaže rozpracovaný dialog i pole answereru."""
        self.state = FocusState()
        self.answerer.reset()

    def turn(self, text, temperature=0.0):
        """Jeden tah konverzace (viz docstring modulu).

        Args:
            text (str): Vstup uživatele (otázka NEBO volba kandidáta).
            temperature (float): Teplota answereru (fuzzy souvislosti).

        Returns:
            IrisResponse: Odpověď nebo dialogové doostření + metadata.
        """
        used_patterns = []
        pick = self._resume_pick(text)
        if pick is not None:
            chosen, text = pick
            used_patterns.append(self.state.pending.card)
            self.state.pending = None
            self.answerer.context.warm(chosen, _PICK_WARMTH)
        # jas PŘED tahem: „zaostřil už uživatel?" — tah sám si téma rozsvítí,
        # takže čtení po odpovědi by jistotu falešně zvedlo
        before = dict(self.answerer.context.scores)
        answer = self.answerer.answer(text, [], temperature=temperature)
        res = self.answerer.last_resolution
        if res is not None:
            glow = before.get(res["winner"], 0.0)
            assur = assurance(res["quality"], len(res["rivals"]), glow)
            candidates = [res["winner"]] + list(res["rivals"])
        else:
            assur = 1.0 if answer.trace else 0.0
            candidates = []
        context = {"assurance": assur, "candidates": candidates,
                   "term": res["term"] if res else text}

        if answer.trace and (not candidates[1:] or assur >= self.threshold):
            return self._respond(answer, "answer", assur, used_patterns)
        if answer.trace:
            # odpověď stojí na hádání mezi rovnocennými kandidáty → nabídka
            card = self.deck.match("resolve.ambiguous", context)
            if card is not None:
                return self._dialog(card, context, text, used_patterns)
            return self._respond(answer, "answer", assur, used_patterns)
        if candidates:
            # rozlišeno, ale bez odpovědního faktu → upřímný terminál
            card = self.deck.match("focus.low", context)
            if card is not None:
                return self._dialog(card, context, text, used_patterns,
                                    await_pick=False)
        return self._respond(answer, "answer", assur, used_patterns)

    def _resume_pick(self, text):
        """Rozpozná volbu kandidáta z rozpracovaného dialogu.

        Volba se páruje volně (bez diakritiky, po slovech): „josef", „Josefa
        Čapka" i „ano" (= první nabídnutý). Nesedí-li nic, text je NOVÁ
        otázka a rozpracovaný dialog se zahodí.

        Returns:
            tuple[str, str] | None: (vybraný kandidát, zaostřená otázka).
        """
        pending = self.state.pending
        if pending is None:
            return None
        words = {deaccent(w.lower()) for w in re.findall(r"[\w.]+", text)}
        if not words:
            return None
        if words <= {"ano", "jo", "yes"}:
            chosen = pending.candidates[0]
        else:
            # vyhrává kandidát s NEJVĚTŠÍM průnikem slov („Josef Čapek"
            # sdílí příjmení s oběma bratry — rozhodne křestní jméno)
            overlaps = [(len(words & {deaccent(w.lower()) for w in c.split()}),
                         c) for c in pending.candidates]
            best = max(overlaps, key=lambda o: o[0])
            chosen = best[1] if best[0] > 0 else None
        if chosen is None:
            self.state.pending = None      # nové téma — dialog končí
            return None
        # zaostřená otázka: nejednoznačný termín → vybraný kandidát
        question = re.sub(re.escape(pending.term), chosen,
                          pending.question, count=1, flags=re.IGNORECASE)
        if chosen not in question:
            question = pending.question    # termín v otázce nebyl — nech být
        return chosen, question

    def _dialog(self, card, context, question, used_patterns,
                await_pick=None):
        """Vykoná pattern-kartu: text z šablony + akce (warm, čekání na volbu)."""
        candidates = context["candidates"]
        text = card.dialog.format(candidates=", ".join(candidates),
                                  term=context["term"])
        warmth = card.action.get("warm_candidates", 0.0)
        for node in candidates:
            self.answerer.context.warm(node, warmth)
        wants_pick = card.action.get("await") == "user-pick" \
            if await_pick is None else await_pick
        if wants_pick:
            self.state.pending = PendingFocus(question, context["term"],
                                              candidates, card.name)
        response = IrisResponse(
            text=text, kind="dialog", assurance=context["assurance"],
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used=self._used(used_patterns + [card.name]),
            clarify={"prompt": text, "candidates": candidates})
        self.state.remember(question, response)
        return response

    def _respond(self, answer, kind, assur, used_patterns):
        """Zabalí odpověď answereru do IrisResponse + metadata tahu."""
        response = IrisResponse(
            text=answer.text, kind=kind, assurance=round(assur, 3),
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used=self._used(used_patterns), trace=answer.trace,
            sources=answer.sources, alternatives=answer.alternatives)
        self.state.remember(answer.text, response)
        return response

    def _used(self, patterns):
        """Metadata použitých komponent (observabilita API — spec §5)."""
        pat = self.answerer.last_pattern
        components = []
        if pat is not None and pat.predicate is not None:
            components.append("pseudo-ql-parser")
        components.append("graph-answerer" if self.answerer.last_trace
                          else "fallback-extractive")
        return {"components": components, "patterns": patterns}
