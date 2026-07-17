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

    def match(self, event, context):
        """Vybere první kartu, jejíž trigger sedí na událost a kontext.

        Args:
            event (str): Událost automatu (např. "resolve.ambiguous").
            context (dict): Stav tahu — `assurance` (float), `candidates`
                (list) a cokoli dalšího, na co se trigger ptá.

        Returns:
            PatternCard | None: Vítězná karta, nebo None (žádný vzor nesedí).
        """
        features = frozenset(context.get("features", ()))
        for card in self.cards:
            trig = card.trigger
            if trig.get("event") != event:
                continue
            below = trig.get("assurance_below")
            if below is not None and context.get("assurance", 0.0) >= below:
                continue
            minimum = trig.get("min_candidates")
            if minimum is not None \
                    and len(context.get("candidates", ())) < minimum:
                continue
            # rysové triggery: karta žádá rysy tahu (requires) a zakazuje
            # jiné (forbids) — klasifikaci dělají KARTY, kód jen počítá rysy
            if not set(trig.get("requires", ())) <= features:
                continue
            if features & set(trig.get("forbids", ())):
                continue
            return card
        return None
