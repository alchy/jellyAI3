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
