from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _graph():
    g = FactGraph()
    born = make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("loc", "Praha", "geo"),
                                 Participant("num", "1890", "number")])
    for _ in range(3):
        g.add_fact(born)
    g.add_fact(make_fact("žít", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("num", "1915", "number")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    return g


def _client(q, tokens):
    return FakeUfalClient(parse={q: [tokens]})


def test_kdy_picks_most_repeated_fact():
    q = "kdy se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 14},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 15, "end": 20},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 21, "end": 26},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "1890"


def test_kde_uses_same_nary_fact():
    q = "kde se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "kde", "lemma": "kde", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 14},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 15, "end": 20},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 21, "end": 26},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Praha"


def test_kdo_traverses_object():
    q = "kdo napsal Babičku?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Božena Němcová"


def test_missing_topic_falls_back():
    q = "kdo je Rossum?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 4, "end": 6},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 7, "end": 13},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text != "Rossum"


def test_topic_prefers_proper_noun_over_common():
    # obecné 'babička' má vyšší váhu, ale téma 'Babička' (kniha) se nesmí splést
    g = FactGraph()
    for _ in range(9):   # obecné 'babička' hodně časté
        g.add_fact(make_fact("péct", [Participant("subj", "babička", "concept"),
                                      Participant("obj", "povídka", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ])
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Božena Němcová"


def test_kdy_requires_verb_match_no_wrong_event():
    # graf má jen svatbu (oženit), ne narození → 'kdy narodil' nesmí vrátit svatbu
    g = FactGraph()
    g.add_fact(make_fact("oženit", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("time", "26. srpna 1935", "time")]))
    q = "kdy se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 14},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 15, "end": 20},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 21, "end": 26},
    ])
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text != "26. srpna 1935"


def test_answer_exposes_trace():
    q = "kdo napsal Babičku?"
    client = _client(q, [
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ])
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    a.answer(q, [])
    assert a.last_trace["topic"] == "Babička"
    assert a.last_trace["answer"] == "Božena Němcová"
