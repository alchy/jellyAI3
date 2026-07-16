from jellyai.ufal_client import FakeUfalClient
from jellyai.answerer.question import analyze_question


def test_copula_question_jaky():
    q = "Jaká byla Němcová?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jaká", "lemma": "jaký", "upos": "DET", "head": 3, "deprel": "det", "start": 0, "end": 4},
        {"form": "byla", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 5, "end": 9},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 0, "deprel": "root", "start": 10, "end": 17},
    ]]})
    qa = analyze_question(q, client)
    assert qa.qtype == "Jaký"        # rozpoznáno přes lemma „jaký"
    assert qa.is_copula is True
    assert "Němcová" in qa.topic_terms


def test_variant_form_maps_to_same_type():
    q = "Jaké je Rossum?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jaké", "lemma": "jaký", "upos": "DET", "head": 3, "deprel": "det", "start": 0, "end": 4},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 5, "end": 7},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 8, "end": 14},
    ]]})
    assert analyze_question(q, client).qtype == "Jaký"


def test_non_copula_verb_question():
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    qa = analyze_question(q, client)
    assert qa.qtype == "Kdo"
    assert qa.is_copula is False
    assert qa.verb_lemma == "napsat"
