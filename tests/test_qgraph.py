"""Otázkový graf (#57) — kompilace a osvětlení (experimentální větev).

Spec `2026-07-20-otazkovy-graf.md`: graf je deterministický KOMPILÁT
dnešních zdrojů (dotazové karty = uzly typů otázek, workery = přímí
experti, clarify karty = ZPŘESŇOVACÍ uzly s hranami; instance z
predikátů datového grafu). Osvětlení tahu (větný graf: lexer + nároky)
vybírá uzel — tady se měří SHADOW režimem, dispatch se nepřepíná.
"""

from jellyai.iris.patterns import PatternDeck
from jellyai.iris.qgraph import compile_qgraph, illuminate


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
