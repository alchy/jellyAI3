"""Testy role ② — doplnění chybějícího titulu do grafu (recover_entities)."""

from jellyai.graph.graph import FactGraph
from jellyai.graph.recover import recover_entities


def _annotation():
    """Věta „Karel Čapek napsal hru R.U.R." — R.U.R. NER minul."""
    sent = [
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2,
         "deprel": "nmod", "start": 0, "end": 5},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 6, "end": 11},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 12, "end": 18},
        {"form": "hru", "lemma": "hra", "upos": "NOUN", "head": 3,
         "deprel": "obj", "start": 19, "end": 22},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 4,
         "deprel": "appos", "start": 23, "end": 29},
    ]
    entities = [{"text": "Karel Čapek", "type": "P", "start": 0, "end": 11}]
    return {("doc", 0): {"entities": entities, "sentences": [sent]}}


def test_recovers_missing_title_with_author():
    g = FactGraph()
    added = recover_entities(_annotation(), g)
    assert "R.U.R." in added
    # v grafu je teď fakt napsat(Čapek, R.U.R.) — dá se dojít z titulu na autora
    facts = g.facts_of("R.U.R.", role="obj", predicate="napsat")
    assert facts
    authors = g.participants(facts[0], "subj")
    assert "Karel Čapek" in authors
    assert g.nodes["R.U.R."].type == "dílo"     # nový uzel je typu dílo


def test_no_work_verb_no_recovery():
    """Bez „work" slovesa v okolí se nic nedoplní."""
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "viděl", "lemma": "vidět", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 6, "end": 11},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 2,
         "deprel": "obj", "start": 12, "end": 18},
    ]
    ann = {("doc", 0): {"entities": [{"text": "Čapek", "type": "P",
                                      "start": 0, "end": 5}], "sentences": [sent]}}
    g = FactGraph()
    assert recover_entities(ann, g) == []


def test_recover_adds_authorship_even_if_node_exists():
    """Uzel titulu může existovat z apoziční identity — autorský fakt se
    přesto doplní (filtr kouká na autorská fakta, ne na existenci uzlu)."""
    from jellyai.graph.extract import make_fact as mf, Participant as P
    g = FactGraph()
    g.add_fact(mf("být", [P("subj", "R.U.R.", "dílo"),
                          P("pred", "hra", "concept")]))
    added = recover_entities(_annotation(), g)
    assert "R.U.R." in added
    assert g.facts_of("R.U.R.", role="obj", predicate="napsat")
