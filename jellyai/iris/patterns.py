"""Pattern-karty — chování automatu Iris jako DATA, ne kód.

Jedna karta = jeden JSON soubor = jeden vzor chování: **trigger** říká, při
jaké události (a za jakých podmínek) se karta použije; **dialog** je šablona
věty k uživateli; **action** říká, co udělat s aktivací (zahřát kandidáty,
čekat na volbu); **teach** je výukové vysvětlení vzoru pro člověka.

Přidání karty do adresáře = rozšíření komunikačních schopností automatu bez
zásahu do kódu. Jiný jazyk = jiný adresář karet (`patterns/<jazyk>/`) — týž
princip jako `jellyai/lang/<jazyk>.json`.
"""

import json
import os

from dataclasses import dataclass, field

_DIR = os.path.dirname(__file__)


@dataclass(frozen=True)
class PatternCard:
    """Jeden vzor chování automatu (načtený z JSON).

    Atributy:
        name (str): Jméno vzoru (= jméno souboru bez přípony).
        trigger (dict): Kdy se vzor použije — klíče `event` (přesná shoda
            s událostí automatu), volitelně `assurance_below` (float; karta
            sedí jen pod touto jistotou), `min_candidates` (int; karta sedí
            až od tolika kandidátů), `priority` (int; vyšší vyhrává).
        dialog (str): Šablona věty k uživateli (`{candidates}`, `{term}`…).
        action (dict): Co s aktivací — např. `warm_candidates` (float, jas
            pro kandidáty), `await` ("user-pick" = čekáme na volbu uživatele).
        teach (str): Polopatické vysvětlení vzoru (výukový kontext).
    """
    name: str
    trigger: dict = field(default_factory=dict)
    dialog: str = ""
    action: dict = field(default_factory=dict)
    teach: str = ""


class PatternDeck:
    """Balíček pattern-karet jednoho jazyka + výběr karty k události.

    Deterministický výběr: karty se řadí podle `priority` (vyšší první)
    a pak podle jména; vyhrává PRVNÍ karta, jejíž trigger sedí.
    """

    def __init__(self, directory):
        """Vytvoří balíček nad adresářem s JSON kartami.

        Args:
            directory (str): Cesta k adresáři (každý `*.json` = jedna karta).
        """
        self.directory = directory
        self.cards = []

    @classmethod
    def for_language(cls, language):
        """Balíček vestavěných karet daného jazyka (`patterns/<jazyk>/`)."""
        return cls(os.path.join(_DIR, "patterns", language))

    def load(self):
        """Načte všechny karty z adresáře (seřazené deterministicky).

        Returns:
            int: Počet načtených karet.
        """
        cards = []
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            cards.append(PatternCard(
                name=data.get("name", name[:-5]),
                trigger=data.get("trigger", {}),
                dialog=data.get("dialog", ""),
                action=data.get("action", {}),
                teach=data.get("teach", "")))
        cards.sort(key=lambda c: (-c.trigger.get("priority", 0), c.name))
        self.cards = cards
        return len(cards)

    def _specificity(self, card, event, context):
        """Kolik podmínek triggeru na tahu sedí; None = trigger nesedí.

        Každá SPLNĚNÁ dodatečná podmínka (práh jistoty, počet kandidátů,
        požadovaný/zakázaný rys) zvyšuje těsnost — a těsnější karta je pro
        daný dotaz větším benefitem než obecná (matching dle otázky).
        """
        trig = card.trigger
        if trig.get("event") != event:
            return None
        score = 0
        below = trig.get("assurance_below")
        if below is not None:
            if context.get("assurance", 0.0) >= below:
                return None
            score += 1
        minimum = trig.get("min_candidates")
        if minimum is not None:
            if len(context.get("candidates", ())) < minimum:
                return None
            score += 1
        # rysové triggery: karta žádá rysy tahu (requires) a zakazuje jiné
        # (forbids) — klasifikaci dělají KARTY, kód jen počítá rysy
        features = frozenset(context.get("features", ()))
        requires = set(trig.get("requires", ()))
        if not requires <= features:
            return None
        forbids = set(trig.get("forbids", ()))
        if features & forbids:
            return None
        return score + len(requires) + len(forbids)

    def match(self, event, context):
        """Vybere PRVNÍ kartu (pořadí priorita→jméno), jejíž trigger sedí.

        Returns:
            PatternCard | None: Vítězná karta, nebo None (žádný vzor nesedí).
        """
        for card in self.cards:
            if self._specificity(card, event, context) is not None:
                return card
        return None

    def best(self, event, context):
        """Vybere kartu s NEJVĚTŠÍM BENEFITEM pro daný tah (spec §2.6b).

        Benefit = těsnost triggeru (kolik podmínek sedí) → priorita → jméno.
        Automat zkouší všechny kandidátky a bere tu, která dotazu nejlépe
        odpovídá; `match()` (first-match) zůstává pro jednoduché balíčky.

        Returns:
            PatternCard | None: Karta s největším benefitem, nebo None.
        """
        scored = []
        for card in self.cards:
            specificity = self._specificity(card, event, context)
            if specificity is not None:
                scored.append((-specificity, -card.trigger.get("priority", 0),
                               card.name, card))
        if not scored:
            return None
        return min(scored)[3]
