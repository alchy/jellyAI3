"""Hygiena grafu — korpusová evidence lemmat proti mis-tagům.

Jedna věta občas projde se špatnou značkou („hoď" jako NOUN → účastník
„hodit"; „Izaiáš" jako VERB → predikát). CELÝ korpus ale ví lépe: hlasování
upos přes všechny výskyty lemmatu odhalí, čím slovo převážně je. Účastník
s převahou slovesných hlasů není entita; predikát s převahou jmenných hlasů
není děj. Bez hlasů (paměť Mnemos, řídká slova) se nesoudí.
"""

from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.hygiene import lemma_upos_votes, scrub


def _annotations(tokens_by_sentence):
    """Anotace z [(lemma, upos), …] vět (minimum pro hlasování)."""
    return {("doc", i): {"sentences": [[{"lemma": l, "upos": u, "form": l,
                                         "head": 0, "deprel": "dep"}
                                        for l, u in sent]]}
            for i, sent in enumerate(tokens_by_sentence)}


def test_votes_count_upos_per_lemma():
    votes = lemma_upos_votes(_annotations([
        [("hodit", "VERB"), ("koleno", "NOUN")],
        [("hodit", "VERB"), ("hodit", "VERB"), ("koleno", "NOUN")],
    ]))
    assert votes["hodit"]["VERB"] == 3
    assert votes["koleno"]["NOUN"] == 2


def test_scrub_drops_verb_dominant_participant():
    """vzít(Mojžíš, obj=hodit) → „hodit" je v korpusu sloveso → účastník
    pryč; fakt zůstává (má ještě podmět a předmět)."""
    g = FactGraph()
    g.add_fact(make_fact("vzít", [Participant("subj", "Mojžíš", "person"),
                                  Participant("obj", "hůl", "concept"),
                                  Participant("obj", "hodit", "concept")]))
    votes = lemma_upos_votes(_annotations([[("hodit", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_participants == 1 and dropped_facts == 0
    nodes = {p.node for f in g.facts.values() for p in f.participants}
    assert "hodit" not in nodes and "hůl" in nodes


def test_scrub_keeps_noun_dominant_and_unvoted():
    """„koleno" (NOUN hlasy) i „knedlíky" (bez hlasů — paměť) zůstávají."""
    g = FactGraph()
    g.add_fact(make_fact("bolet", [Participant("subj", "koleno", "concept"),
                                   Participant("obj", "knedlíky", "concept")]))
    votes = lemma_upos_votes(_annotations([[("koleno", "NOUN")] * 3]))
    scrub(g, votes)
    nodes = {p.node for f in g.facts.values() for p in f.participants}
    assert "koleno" in nodes and "knedlíky" in nodes


def test_scrub_drops_fact_with_name_dominant_predicate():
    """Predikát „Izaiáš" (v korpusu PROPN) není děj → celý fakt pryč;
    legitimní slovesný predikát zůstává."""
    g = FactGraph()
    g.add_fact(make_fact("Izaiáš", [Participant("subj", "prorok", "concept"),
                                    Participant("obj", "svitek", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Izaiáš", "person"),
                                    Participant("obj", "svitek", "concept")]))
    votes = lemma_upos_votes(_annotations([
        [("Izaiáš", "PROPN")] * 3 + [("napsat", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 1
    predicates = {f.predicate for f in g.facts.values()}
    assert "Izaiáš" not in predicates and "napsat" in predicates


def test_structural_predicates_survive_noun_votes():
    """Reifikované vztahy (bratr), druh, kontext i dekompozice dat (rok)
    jsou JMENNÉ predikáty ZÁMĚRNĚ — hlasování je nesmí smazat."""
    g = FactGraph()
    g.add_fact(make_fact("bratr", [Participant("subj", "Josef", "person"),
                                   Participant("obj", "Karel", "person")]))
    g.add_fact(make_fact("rok", [Participant("subj", "2. května 1818", "time"),
                                 Participant("val", "1818", "number")]))
    votes = lemma_upos_votes(_annotations([
        [("bratr", "NOUN")] * 5 + [("rok", "NOUN")] * 5]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 0
    assert {f.predicate for f in g.facts.values()} == {"bratr", "rok"}


def test_scrub_drops_fact_left_without_partner():
    """Když po vyřazení účastníka zbude faktu jediný účastník, fakt padá
    (fakt bez protistrany nic nenese)."""
    g = FactGraph()
    g.add_fact(make_fact("nechat", [Participant("subj", "mluvit", "concept"),
                                    Participant("obj", "džbán", "concept")]))
    votes = lemma_upos_votes(_annotations([[("mluvit", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 1
    assert not g.facts
