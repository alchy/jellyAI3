"""Otázkový graf (#57, experimentální větev) — kompilace a osvětlení.

Spec `2026-07-20-otazkovy-graf.md`. Graf je deterministický KOMPILÁT
dnešních zdrojů — nevzniká druhá pravda:

- uzly `otazka` = dotazové karty se vzorem (povrch = karta, worker graf);
- uzly `worker` = přímí experti (Metron výraz, Chronos hodiny, meta-fokus)
  s branami, které dnes drží ruční pořadí v turn();
- uzly `clarify` = ZPŘESŇOVACÍ karty (zadání user): je-li takový uzel
  aktivní, dialog pokračuje zpřesňováním a graf si sám ostří focus —
  hrany `zpresneni` (otázka → clarify) a `navrat` (clarify → přehrání
  otázky, dnešní PendingFocus).

Osvětlení tahu: větný graf (lexer hypotézy + nároky expertů) rozsvítí
uzly; pořadí = (tier, priorita, délka vzoru [, váha z telemetrie]).
DISPATCH SE NEPŘEPÍNÁ — shadow měření dělá benchmark/run_qgraph.py.
"""

from dataclasses import dataclass, field
from datetime import datetime

from jellyai.graph.canon import deaccent
from jellyai.lang import current
from jellyai.lang.lexer import classify
from jellyai.lang.matcher import expand_pattern, match_sequence

_CLARIFY_EVENTS = {"resolve.ambiguous", "data.overflow", "focus.low",
                   "statement.subject"}


@dataclass
class QEdge:
    kind: str        # "zpresneni" | "navrat"
    target: str      # jméno uzlu; "*" = přehraj rozpracovanou otázku


@dataclass
class QNode:
    name: str
    kind: str        # "otazka" | "worker" | "clarify"
    worker: str      # "graf" | "metron" | "chronos" | "iris" | "dialog"
    pattern: list = None
    card: str = None
    priority: int = 0              # priorita zdrojové karty (klíč decku)
    weight: float = 0.0            # zásahy z telemetrie (#38)
    edges: list = field(default_factory=list)


@dataclass
class QGraph:
    nodes: dict
    predicates: frozenset          # schéma datového grafu (líné instance)


def compile_qgraph(deck, predicates=frozenset(), telemetry_rows=()):
    """Zkompiluje otázkový graf z decku karet + schématu predikátů.

    Deterministické (týž deck + tatáž telemetrie → týž graf); karty
    zůstávají zdrojovým kódem, graf je jen jejich strukturovaný pohled.

    Returns:
        QGraph: Uzly (otázky, workeři, clarify) s hranami a vahami.
    """
    nodes = {}
    for card in deck.cards:
        trigger = card.trigger
        if trigger.get("event") == "utterance.query" and trigger.get("pattern"):
            nodes[card.name] = QNode(
                name=card.name, kind="otazka", worker="graf",
                pattern=trigger["pattern"], card=card.name,
                priority=trigger.get("priority", 0))
        elif trigger.get("event") in _CLARIFY_EVENTS:
            nodes[card.name] = QNode(
                name=card.name, kind="clarify", worker="dialog",
                card=card.name,
                edges=[QEdge("navrat", "*")])   # po volbě přehraj otázku
    # PŘÍMÍ EXPERTI — brány, které dnes drží ruční pořadí větví v turn()
    nodes["metron-vypocet"] = QNode("metron-vypocet", "worker", "metron")
    nodes["chronos-hodiny"] = QNode("chronos-hodiny", "worker", "chronos")
    nodes["meta-focus"] = QNode("meta-focus", "worker", "iris")
    clarify_names = [n.name for n in nodes.values() if n.kind == "clarify"]
    for node in nodes.values():
        if node.kind == "otazka":
            # digging: otázka smí pokračovat zpřesněním (homonymum,
            # přetečení, nízká jistota) — graf si ostří focus dialogem
            node.edges = [QEdge("zpresneni", name) for name in clarify_names]
    for row in telemetry_rows:                   # váhy uzlů z provozu
        for pattern_name in row.get("patterns", ()):
            if pattern_name in nodes:
                nodes[pattern_name].weight += 1.0
    return QGraph(nodes=nodes, predicates=frozenset(predicates))


def decorate(text, now=None):
    """DEKORUJÍCÍ nároky tahu (T3 spec — druhý druh rozsvěcení).

    Nároky expertů NEsoutěží s uzly otázek: věší se na vítěze jako
    OMEZENÍ. Model jen pojmenovává, co dnes v answereru už žije
    (time_filter, place_filter, user_subject, _theme_bound, novelty)
    — proto se dekorace čtou z TÝCHŽ zdrojů (jazykové tabulky).

    Returns:
        frozenset[str]: Jména dekorací („chronos:interval", „topos:oblast",
        „mnemos:prvni-osoba", „role:adresat", „novost").
    """
    from jellyai.iris.subsystems.chronos import resolve_temporal
    lang = current()
    found = set()
    if resolve_temporal(text, now or datetime.now()) is not None:
        found.add("chronos:interval")
    tagged = classify(text, is_node=None)
    classes = {cls for token in tagged for cls in token.classes}
    if "prvni_osoba" in classes:
        found.add("mnemos:prvni-osoba")
    if "dativ" in classes:
        found.add("role:adresat")
    norms = {token.norm for token in tagged}
    if "dalsi" in norms:
        found.add("novost")
    if norms & set(lang.get("relation_query_nouns", ())):
        found.add("vztah:operator")
    return frozenset(found)


class DialogPosition:
    """POZICE v otázkovém grafu = stav dialogu (T4 spec).

    Sjednocuje dnešní roztroušené reprezentace (PendingFocus, drill
    přes _prev_trace, pick_focus): stojíme v uzlu a ven vedou HRANY.
    Zpřesnění je krok po hraně `zpresneni`, volba kandidáta návrat
    hranou `navrat` (přehraj otázku) — graf si tak sám ostří focus.
    """

    def __init__(self, qgraph):
        self.qgraph = qgraph
        self.node = None
        self._origin = None       # uzel otázky, ke kterému se vracíme

    def enter(self, name):
        """Vstup do uzlu (tah směrovaný osvětlením)."""
        self.node = self.qgraph.nodes.get(name)
        self._origin = None
        return self.node

    def sharpen(self, name):
        """Krok po zpřesňovací hraně; False = hrana neexistuje."""
        if self.node is None:
            return False
        if not any(e.kind == "zpresneni" and e.target == name
                   for e in self.node.edges):
            return False
        self._origin, self.node = self.node, self.qgraph.nodes[name]
        return True

    def resume(self):
        """Návrat ze zpřesnění: cíl hrany `navrat` („*" = přehraj otázku)."""
        if self.node is None or self.node.kind != "clarify":
            return None
        target = next((e.target for e in self.node.edges
                       if e.kind == "navrat"), None)
        self.node, self._origin = self._origin, None
        return target


def illuminate(text, qgraph, now=None, is_node=None, use_weights=False):
    """Osvětlení tahu: kdo z uzlů svítí a jak silně.

    Tier 3 = přímý expert (nárok obsahu řádku: Metronův výraz, hodinová
    otázka, meta-fráze) — zrcadlí dnešní přednost bran. Tier 2 = vzorové
    uzly otázek (priorita, délka vzoru — týž klíč jako deck). Clarify
    uzly text nesvítí — aktivují se STAVEM (hranou z otázky), měří je
    shadow proti skutečným dialog tahům.

    Returns:
        list[QNode]: Rozsvícené uzly, nejsilnější první ([] = tma).
    """
    from jellyai.iris.subsystems.chronos import clock_answer
    from jellyai.iris.subsystems.metron import compute
    lang = current()
    lit = []
    if compute(text) is not None:
        lit.append(((3, 2, 0), qgraph.nodes["metron-vypocet"]))
    if clock_answer(text, now or datetime.now()) is not None:
        lit.append(((3, 1, 0), qgraph.nodes["chronos-hodiny"]))
    low = deaccent(text.lower())
    if any(p in low for p in lang.get("focus_query_phrases", ())):
        lit.append(((3, 0, 0), qgraph.nodes["meta-focus"]))
    tagged = classify(text, is_node=None)
    aliases = lang.get("pattern_aliases", {})
    for node in qgraph.nodes.values():
        if node.kind != "otazka":
            continue
        binding = match_sequence(expand_pattern(node.pattern, aliases),
                                 tagged, is_span=is_node)
        if binding is None:
            continue
        # týž klíč jako deck (priorita, délka vzoru); varianta s vahami
        # z telemetrie láme remízy provozem — kolik provozu je potřeba,
        # než váhy něco znamenají, určí testy (zadání user)
        key = (2, node.priority, len(node.pattern),
               node.weight if use_weights else 0)
        lit.append((key, node))
    lit.sort(key=lambda pair: pair[0], reverse=True)
    return [node for _, node in lit]
