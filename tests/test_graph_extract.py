from jellyai.graph.extract import extract_facts, make_fact, Fact, Participant


def _ann(sent, entities=None):
    return {"entities": entities or [], "sentences": [sent]}


def test_svo_fact():
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]
    ents = [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")])
    assert expected in facts


def test_copula_fact():
    sent = [
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 7, "end": 9},
        {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 19},
    ]
    facts = extract_facts(_ann(sent))
    expected = make_fact("být", [Participant("subj", "Rossum", "concept"),
                                 Participant("pred", "vynálezce", "concept")])
    assert expected in facts


def test_nary_fact_place_and_time():
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 13},
        {"form": "Praze", "lemma": "Praha", "upos": "PROPN", "head": 2, "deprel": "obl", "start": 17, "end": 22},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 2, "deprel": "obl", "start": 23, "end": 27},
    ]
    ents = [{"text": "Čapek", "type": "P", "start": 0, "end": 5},
            {"text": "Praha", "type": "G", "start": 17, "end": 22}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                     Participant("loc", "Praha", "geo"),
                                     Participant("num", "1890", "number")])
    assert expected in facts
