"""Reifikace vztahů — relační podstatné jméno + genitivní osoba → osoba–vztah–osoba."""

from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import extract_facts


def _ann(sent, entities):
    return {"entities": entities, "sentences": [sent]}


def test_relational_noun_with_genitive_person():
    """„Josef byl bratr Karla Čapka" → bratr(Josef, Karel Čapek), ne jen být→bratr."""
    sent = [
        {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 6, "end": 9},
        {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 10, "end": 15},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "nmod", "start": 16, "end": 21},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4,
         "deprel": "flat", "start": 22, "end": 27},
    ]
    entities = [{"text": "Josef", "type": "P", "start": 0, "end": 5},
                {"text": "Karel Čapek", "type": "P", "start": 16, "end": 27}]
    facts = extract_facts(_ann(sent, entities))
    rel = next(f for f in facts if f.predicate == "bratr")
    g = FactGraph()
    for f in facts:
        g.add_fact(f)
    # z Karla se dá dojít na bratra Josefa (obj-role dotaz „Kdo byl bratr X?")
    br = g.facts_of("Karel Čapek", role="obj", predicate="bratr")
    assert br and "Josef" in g.participants(br[0], "subj")
    assert not any(p.role == "pred" and p.node == "bratr" for p in rel.participants)


def test_nonrelational_copula_stays_identity():
    """Nevztahové sponové podstatné jméno („spisovatel") zůstane identitou (pred)."""
    sent = [
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 6, "end": 9},
        {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 10, "end": 20},
    ]
    entities = [{"text": "Karel", "type": "P", "start": 0, "end": 5}]
    facts = extract_facts(_ann(sent, entities))
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "pred" and p.node == "spisovatel" for p in byt.participants)
