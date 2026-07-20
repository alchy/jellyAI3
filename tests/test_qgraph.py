"""Otázkový graf (#57) — kompilace a osvětlení (experimentální větev).

Spec `2026-07-20-otazkovy-graf.md`: graf je deterministický KOMPILÁT
dnešních zdrojů (dotazové karty = uzly typů otázek, workery = přímí
experti, clarify karty = ZPŘESŇOVACÍ uzly s hranami; instance z
predikátů datového grafu). Osvětlení tahu (větný graf: lexer + nároky)
vybírá uzel — tady se měří SHADOW režimem, dispatch se nepřepíná.
"""

from datetime import datetime

from jellyai.iris.patterns import PatternDeck
from jellyai.iris.qgraph import compile_qgraph, illuminate

NOW = datetime(2026, 7, 17, 12, 0)


def _graph():
    deck = PatternDeck.for_language("cs")
    deck.load()
    return compile_qgraph(deck, predicates={"napsat", "narodit", "říci"})


def test_compile_builds_question_worker_and_clarify_nodes():
    qg = _graph()
    kinds = {node.kind for node in qg.nodes.values()}
    assert kinds == {"otazka", "worker", "clarify"}
    assert "q-otaz-minuly" in qg.nodes            # karta = uzel typu otázky
    assert qg.nodes["metron-vypocet"].worker == "metron"
    assert qg.nodes["chronos-hodiny"].worker == "chronos"
    assert qg.nodes["focus-offer-homonym"].kind == "clarify"


def test_clarify_nodes_carry_sharpening_edges():
    """Zpřesňovací uzly (zadání user): aktivní clarify uzel pokračuje
    dialogem a po volbě se vrací k otázce — hrany tam i zpět."""
    qg = _graph()
    otazka = qg.nodes["q-otaz-minuly"]
    assert any(edge.kind == "zpresneni" for edge in otazka.edges)
    clarify = qg.nodes["focus-offer-homonym"]
    assert any(edge.kind == "navrat" for edge in clarify.edges)


def test_illumination_routes_families_apart():
    """Dvě rodiny „kolik" se osvětlením rozdělí samy (T2 spec):
    výraz svítí Metronu, otázka grafu Metron nechává tmavý."""
    qg = _graph()
    lit = illuminate("Kolik je 1 plus 1?", qg)
    assert lit and lit[0].name == "metron-vypocet"
    lit = illuminate("Kolik měla dětí Božena Němcová?", qg,
                     is_node=lambda s: "Němcová" in s)
    assert not lit or lit[0].worker != "metron"
    lit = illuminate("Kdo napsal R.U.R.?", qg,
                     is_node=lambda s: s == "R.U.R.")
    assert lit and lit[0].name == "q-otaz-minuly"
    lit = illuminate("Kolik je hodin?", qg)
    assert lit and lit[0].name == "chronos-hodiny"


def test_decorations_attach_not_compete():
    """T3 spec (korekce modelu): nároky expertů NEsoutěží s uzly — věší
    se na vítěze jako omezení. „Kdy jsem měl v tomto roce knedlíky?"
    vyhraje uzel otázky a nese dekorace času i 1. osoby."""
    from jellyai.iris.qgraph import decorate

    qg = _graph()
    text = "Kdy jsem měl v tomto roce knedlíky?"
    lit = illuminate(text, qg, now=NOW, is_node=lambda s: s == "knedlíky")
    assert lit and lit[0].kind == "otazka"       # soutěž vyhrál uzel otázky
    deco = decorate(text, now=NOW)
    assert "chronos:interval" in deco            # „v tomto roce" = filtr
    assert "mnemos:prvni-osoba" in deco          # podmět je uživatel

    deco = decorate("Co řekl Ježíšovi?", now=NOW)
    assert "role:adresat" in deco                # dativ váže roli theme
    deco = decorate("Kdo další bydlí v Petrovicích?", now=NOW)
    assert "novost" in deco
    assert decorate("Kdo napsal R.U.R.?", now=NOW) == frozenset()


def test_dialog_position_walks_sharpening_edges():
    """Stav dialogu = POZICE v grafu (T4 spec): po otázce s clarify
    stojíme v clarify uzlu, volba se vrací hranou navrat k otázce."""
    from jellyai.iris.qgraph import DialogPosition

    qg = _graph()
    pos = DialogPosition(qg)
    assert pos.node is None
    pos.enter("q-otaz-minuly")
    assert pos.node.name == "q-otaz-minuly"
    assert pos.sharpen("focus-offer-homonym")    # legitimní hrana
    assert pos.node.name == "focus-offer-homonym"
    assert not pos.sharpen("q-cim-sloveso")      # není zpřesnění
    assert pos.resume() == "*"                   # návrat = přehraj otázku
    assert pos.node.name == "q-otaz-minuly"      # pozice zpět u otázky
