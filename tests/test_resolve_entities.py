"""Post-build resolver entit — sloučení pádových variant osob (cesta A blockeru).

Kanonické id = lexikograficky nejmenší člen clusteru (pádová koncovka nominativ
prodlužuje, takže minimum je nominativ; subj_count nefunguje — pro-drop rozdává
podměty i genitivním uzlům). Konzervativně: jen person↔person, jen týž kmenový klíč.
"""

from jellyai.graph.graph import FactGraph, resolve_entities
from jellyai.graph.extract import make_fact, Participant


def _bratr(subj, obj):
    return make_fact("bratr", [Participant("subj", subj, "person"),
                               Participant("obj", obj, "person")])


def _narodit(subj, loc):
    return make_fact("narodit", [Participant("subj", subj, "person"),
                                 Participant("loc", loc, "geo")])


def test_case_variants_merge_to_nominative():
    # přesně mechanismus blockeru: bratr-fakt visí na genitivu, narodit na nominativu
    g = FactGraph()
    g.add_fact(_narodit("Josef Čapek", "Hronově"))
    g.add_fact(_bratr("Karel Antonín Čapek", "Josefa Čapka"))
    resolve_entities(g)
    assert "Josefa Čapka" not in g.nodes
    bratr = g.facts_of("Josef Čapek", role="obj", predicate="bratr")
    assert bratr and g.participants(bratr[0], "subj") == ["Karel Antonín Čapek"]


def test_canonical_is_lex_min_not_weight():
    # genitiv frekventovanější (reálná data: 'Karla Čapka' w=15 vs 'Karel Čapek' w=3)
    g = FactGraph()
    for _ in range(5):
        g.add_fact(_narodit("Karla Čapka", "Praze"))
    g.add_fact(_narodit("Karel Čapek", "Praze"))
    resolve_entities(g)
    assert "Karel Čapek" in g.nodes and "Karla Čapka" not in g.nodes


def test_merged_fact_weights_aggregate():
    g = FactGraph()
    g.add_fact(_narodit("Josefa Čapka", "Hronově"))
    g.add_fact(_narodit("Josefa Čapka", "Hronově"))
    g.add_fact(_narodit("Josef Čapek", "Hronově"))
    resolve_entities(g)
    facts = g.facts_of("Josef Čapek", role="subj", predicate="narodit")
    assert len(facts) == 1 and facts[0].weight == 3
    assert g.nodes["Josef Čapek"].weight == 3


def test_conservative_father_son_and_bare_surname_stay_apart():
    # mantinel 4: jiný počet slov = jiný klíč → otec/syn/holé příjmení se nedotknou
    g = FactGraph()
    g.add_fact(_narodit("Karel Antonín Čapek", "Malých Svatoňovicích"))
    g.add_fact(_narodit("Antonína Čapka", "Žernově"))
    g.add_fact(_narodit("Čapek", "Praze"))
    resolve_entities(g)
    assert {"Karel Antonín Čapek", "Antonína Čapka", "Čapek"} <= set(g.nodes)


def test_non_person_nodes_untouched():
    # mantinel 3: geo/time/číselné uzly se (zatím) neshlukují
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Praha", "geo"),
                                 Participant("pred", "město", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Prahy", "geo"),
                                 Participant("pred", "město", "concept")]))
    resolve_entities(g)
    assert "Praha" in g.nodes and "Prahy" in g.nodes


def test_positional_merge_short_name_into_long():
    """„Karel Čapek" ⊂ „Karel Antonín Čapek": kmeny sedí na první (křestní)
    i poslední (příjmení) pozici → merge. Fakty obou splynou na delším uzlu."""
    g = FactGraph()
    g.add_fact(_narodit("Karel Antonín Čapek", "Malých Svatoňovicích"))
    g.add_fact(_bratr("Karel Čapek", "Josef Čapek"))
    resolve_entities(g)
    assert "Karel Čapek" not in g.nodes
    bratr = g.facts_of("Karel Antonín Čapek", role="subj", predicate="bratr")
    assert bratr and g.participants(bratr[0], "obj") == ["Josef Čapek"]


def test_positional_merge_skips_father():
    """„Antonín Čapek" (otec): křestní kmen sedí na PROSTŘEDNÍ pozici syna,
    ne na první → zůstává vlastním uzlem (žádné hladové subset-slučování)."""
    g = FactGraph()
    g.add_fact(_narodit("Karel Antonín Čapek", "Malých Svatoňovicích"))
    g.add_fact(_narodit("Antonína Čapka", "Žernově"))
    g.add_fact(_narodit("Čapek", "Praze"))          # holé příjmení taky ne
    resolve_entities(g)
    assert {"Karel Antonín Čapek", "Antonína Čapka", "Čapek"} <= set(g.nodes)


def test_positional_merge_requires_unique_target():
    """Dvojznačný cíl (dva delší kandidáti se stejným křestním i příjmením)
    → kratší jméno zůstává raději oddělené."""
    g = FactGraph()
    g.add_fact(_narodit("Karel Antonín Čapek", "Malých Svatoňovicích"))
    g.add_fact(_narodit("Karel Josef Čapek", "Praze"))
    g.add_fact(_narodit("Karel Čapek", "Brně"))
    resolve_entities(g)
    assert "Karel Čapek" in g.nodes


def test_resolve_is_idempotent_and_deterministic():
    def build():
        g = FactGraph()
        g.add_fact(_bratr("Karel Antonín Čapek", "Josefa Čapka"))
        g.add_fact(_narodit("Josef Čapek", "Hronově"))
        g.add_fact(_narodit("Josefu Čapkovi", "Hronově"))   # dativ (stemmer -ovi)
        return resolve_entities(g)
    g1, g2 = build(), build()
    assert list(g1.facts.keys()) == list(g2.facts.keys())
    assert list(resolve_entities(g1).facts.keys()) == list(g2.facts.keys())


def test_resolver_records_aliases_of_merged_forms():
    """Sloučené tvary se pamatují (graph.aliases) — pro detail uzlu ve viz
    a pro vysvětlitelnost kanonizace."""
    g = FactGraph()
    g.add_fact(_narodit("Josef Čapek", "Hronově"))
    g.add_fact(_bratr("Karel Antonín Čapek", "Josefa Čapka"))
    resolve_entities(g)
    assert "Josefa Čapka" in g.aliases.get("Josef Čapek", [])
    again = resolve_entities(g)      # idempotence drží i aliasy
    assert "Josefa Čapka" in again.aliases["Josef Čapek"]
