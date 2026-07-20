"""Pattern-karty — chování automatu Iris jako DATA, ne kód.

Jedna karta = jeden JSON soubor = jeden vzor chování: **trigger** říká, při
jaké události (a za jakých podmínek) se karta použije; **dialog** je šablona
věty k uživateli; **action** říká, co udělat s aktivací (zahřát kandidáty,
čekat na volbu); **teach** je výukové vysvětlení vzoru pro člověka.

Přidání karty do adresáře = rozšíření komunikačních schopností automatu bez
zásahu do kódu. Jiný jazyk = jiný adresář karet (`patterns/<jazyk>/`) — týž
princip jako `jellyai/lang/<jazyk>.json`.
"""

import copy
import itertools
import json
import os

from dataclasses import dataclass, field

_DIR = os.path.dirname(__file__)


def _expand_family(data):
    """Rozvine RODINNOU kartu (kostra × dimenze) na konkrétní karty (#57 E1).

    Dimenze jsou ENUMERATIVNÍ osy, žádné podmínky (zákaz ATN): hodnota
    osy jen dosadí prvek do slotu kostry, přidá příponu jména a smí
    přepsat prioritu (definuje-li ji víc os, platí poslední). Prvek
    `null` = slot zmizí — smí být jen POSLEDNÍM prvkem vzoru, aby se
    neposouvaly odkazy $N (disciplína pasti 14); odkaz na zmizelý slot
    se z `known` vypustí a v `hole`/`predicate` spadne nahlas.
    """
    trigger = data["trigger"]
    skeleton = trigger["pattern"]
    dimensions = trigger["dimensions"]
    cards = []
    for combo in itertools.product(*(d["values"] for d in dimensions)):
        chosen = {d["slot"]: v for d, v in zip(dimensions, combo)}
        pattern, dropped = [], None
        for index, element in enumerate(skeleton, start=1):
            value = chosen.get(element)
            if value is None:                    # obyčejný prvek kostry
                pattern.append(element)
            elif value["element"] is None:
                if index != len(skeleton):
                    raise ValueError(
                        f"rodina {data['name']}: prázdný slot {element}"
                        f" musí být posledním prvkem vzoru")
                dropped = f"${index}"
            else:
                pattern.append(value["element"])
        priority = trigger.get("priority", 0)
        for value in combo:
            if "priority" in value:
                priority = value["priority"]
        action = copy.deepcopy(data.get("action", {}))
        query = action.get("query", {})
        for key in ("hole", "predicate"):
            if dropped is not None and query.get(key) == dropped:
                raise ValueError(f"rodina {data['name']}: {key}"
                                 f" míří na prázdný slot")
        if dropped is not None and "known" in query:
            kept = [k for k in query["known"]
                    if (k[-1] if isinstance(k, list) else k) != dropped]
            if kept:
                query["known"] = kept
            else:
                del query["known"]
        new_trigger = {k: v for k, v in trigger.items()
                       if k not in ("dimensions", "pattern")}
        new_trigger["pattern"] = pattern
        new_trigger["priority"] = priority
        cards.append({"name": data["name"]
                      + "".join(v["suffix"] for v in combo),
                      "trigger": new_trigger, "action": action,
                      "teach": data.get("teach", "")})
    return cards


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


_SHARED = {}


def shared_deck(language="cs"):
    """JEDEN deck za proces (postřeh 1.2) — obě cesty (automat, dotazy)
    čtou tytéž karty; test smí cache vyprázdnit (`_SHARED.clear()`)."""
    if language not in _SHARED:
        deck = PatternDeck.for_language(language)
        deck.load()
        _SHARED[language] = deck
    return _SHARED[language]


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
            variants = (_expand_family(data)
                        if "dimensions" in data.get("trigger", {})
                        else [data])
            for item in variants:
                cards.append(PatternCard(
                    name=item.get("name", name[:-5]),
                    trigger=item.get("trigger", {}),
                    dialog=item.get("dialog", ""),
                    action=item.get("action", {}),
                    teach=item.get("teach", "")))
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
        if trig.get("pattern"):
            # VZOROVÁ karta se vybírá matchem sekvence (matcher), ne rysy —
            # bez vazby by tu vyhrála NAPRÁZDNO (vakuová logika, past 2:
            # vsuvka s prioritou 9 přebila kratke-sloveso 8 a rozbila zápis)
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
