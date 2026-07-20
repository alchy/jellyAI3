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


def test_default_claims_rozpoznavaji_prime_brany():
    """E2 (#26): nároky přímých expertů jako DATA registru — kompilace
    a osvětlení je čtou jednotně, ruční výčet v kódu grafu mizí."""
    from jellyai.iris.claims import default_claims
    claims = {claim.name: claim for claim in default_claims()}
    assert set(claims) == {"metron-vypocet", "chronos-hodiny",
                           "meta-focus"}
    assert claims["metron-vypocet"].recognize("Kolik je 1 plus 1?", NOW)
    assert not claims["metron-vypocet"].recognize(
        "Kolik měla dětí Božena Němcová?", NOW)
    assert claims["chronos-hodiny"].recognize("Kolik je hodin?", NOW)
    assert claims["meta-focus"].recognize("O kom mluvíme?", NOW)
    assert not claims["meta-focus"].recognize("Kdo napsal R.U.R.?", NOW)
    assert (claims["metron-vypocet"].priority
            > claims["chronos-hodiny"].priority
            > claims["meta-focus"].priority)


def test_novy_expert_claimem_bez_zasahu_do_kodu_grafu():
    """Kritérium E2: nový přímý expert = nový claim v registru —
    kompilace mu postaví worker uzel a osvětlení ho zvedne; do
    compile_qgraph/illuminate/turn se NEsahá."""
    from jellyai.iris.claims import ExpertClaim, default_claims
    deck = PatternDeck.for_language("cs")
    deck.load()
    fake = ExpertClaim("pokus-expert", "pokus", 9,
                       lambda text, now: "pokusný nárok" in text)
    qg = compile_qgraph(deck, claims=default_claims() + (fake,))
    assert qg.nodes["pokus-expert"].worker == "pokus"
    lit = illuminate("Tohle je pokusný nárok na tah.", qg, now=NOW)
    assert lit and lit[0].name == "pokus-expert"


def test_automat_dispatchuje_prime_experty_z_grafu():
    """Fáze D: pořadí bran přímých expertů je v DATECH (claims),
    ne v pořadí větví _turn(); neznámý worker bezpečně propadá."""
    from config import Config
    from jellyai.iris import IrisAutomaton
    from jellyai.iris.claims import ExpertClaim
    from jellyai.tasks import make_graph_answerer

    iris = IrisAutomaton(make_graph_answerer(Config()),
                         clock=lambda: NOW)
    assert iris.qgraph.nodes["metron-vypocet"].worker == "metron"
    r = iris.turn("Kolik je hodin?")
    assert r.used["components"] == ["chronos"]
    r = iris.turn("Kolik je 1 plus 1?")
    assert "2" in r.text
    # claim s neznámým workerem se hlásí o všechno — musí PROPADNOUT
    vetrelec = ExpertClaim("vetrelec", "neexistuje", 9,
                           lambda text, now: True)
    iris.qgraph.claims = (vetrelec,) + tuple(iris.qgraph.claims)
    r = iris.turn("Kolik je hodin?")
    assert r.used["components"] == ["chronos"]


def test_instance_ze_schematu_predikatu():
    """E3: instance svítí jen s rolí díry ve schématu; neznámý
    predikát nebo prázdné role = ŽÁDNÝ verdikt (vakuum, past 2)."""
    from config import Config
    from jellyai.graph.graph import instance_lit
    from jellyai.tasks import make_graph_answerer

    graph = make_graph_answerer(Config()).graph
    roles = graph.predicate_roles("napsat")
    assert "subj" in roles and "obj" in roles
    assert "loc" not in roles
    roles_of = graph.predicate_roles
    assert instance_lit("napsat", "obj", roles_of) is True
    assert instance_lit("napsat", "loc", roles_of) is False
    assert instance_lit("blafnout", "loc", roles_of) is None


def test_clarify_hrany_se_odvozuji():
    """E4: hrany otázka→clarify z dat karet (díra × event), ne
    kartézský součin — statement clarify z otázky nevede, overflow
    jen z výčtových děr."""
    qg = _graph()
    otaz = qg.nodes["q-otaz-minuly"]           # má díru
    cile = {e.target for e in otaz.edges if e.kind == "zpresneni"}
    assert "focus-offer-overflow" in cile
    assert "clarify-identity" not in cile       # statement-side
    zjist = qg.nodes["q-zjistovaci-prezens"]    # bez díry
    cile = {e.target for e in zjist.edges if e.kind == "zpresneni"}
    assert "focus-offer-overflow" not in cile   # existence nepřeteče
    assert "focus-offer-homonym" in cile


def test_turn_features_nese_otaznik_i_vyrokove_rysy():
    """#51 fáze 1: jedna funkce rysů tahu — povrch (otaznik) +
    výrokové rysy (tytéž, kterými vybírá parse_statement)."""
    from jellyai.iris.qgraph import turn_features
    assert "otaznik" in turn_features("Prší?")
    features = turn_features("Venku prší.")
    assert "otaznik" not in features
    assert "first_person" in turn_features("Dnes jsem měl knedlíky.")
