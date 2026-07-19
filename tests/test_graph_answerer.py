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


def test_drill_v_kterem_roce():
    from jellyai.graph.graph import build_graph
    from tests.test_fact_graph import _capek_birth_annotations
    g = build_graph(_capek_birth_annotations())
    q = "v kterém roce se narodil Karel Čapek?"
    client = _client(q, [
        {"form": "v", "lemma": "v", "upos": "ADP", "head": 3, "deprel": "case", "start": 0, "end": 1},
        {"form": "kterém", "lemma": "který", "upos": "DET", "head": 3, "deprel": "det", "start": 2, "end": 8},
        {"form": "roce", "lemma": "rok", "upos": "NOUN", "head": 5, "deprel": "obl", "start": 9, "end": 13},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 5, "deprel": "expl", "start": 14, "end": 16},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 17, "end": 24},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 5, "deprel": "nsubj", "start": 25, "end": 30},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 6, "deprel": "flat", "start": 31, "end": 36},
    ])
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "1890"      # drill: událost → datum → rok


def test_place_answer_normalized_to_nominative():
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("loc", "Slezsku", "geo")]))
    q = "kde se narodila Božena Němcová?"
    client = FakeUfalClient(
        parse={q: [[
            {"form": "kde", "lemma": "kde", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
            {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 15},
            {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 16, "end": 22},
            {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 23, "end": 30},
        ]]},
        analyze={"Slezsku": [{"form": "Slezsku", "lemma": "Slezsko", "tag": "NNNS6-----A----"}]},
        generate={("Slezsko", "NNNS1-----A----"): ["Slezsko"]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Slezsko"     # skloněno do 1. pádu


def test_conversational_followup_resolves_topic():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "2. května 1818", "time")]))
    q1 = "kdo napsal Babičku?"
    q2 = "kdy se narodila?"
    client = FakeUfalClient(parse={
        q1: [[
            {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
            {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
            {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
        ]],
        q2: [[
            {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
            {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 15},
        ]],
    })
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q1, []).text == "Božena Němcová"
    assert a.answer(q2, []).text == "2. května 1818"     # navázalo na téma z minulého tahu


def test_conversation_history_records_trajectory():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "2. května 1818", "time")]))
    q1, q2 = "kdo napsal Babičku?", "kdy se narodila?"
    client = FakeUfalClient(parse={
        q1: [[
            {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
            {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
            {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
        ]],
        q2: [[
            {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod", "start": 0, "end": 3},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 4, "end": 6},
            {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 7, "end": 15},
        ]],
    })
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.answer(q1, [])
    a.answer(q2, [])
    assert len(a.history) == 2
    assert a.history[0]["answer"] == "Božena Němcová"
    assert a.history[1]["topic"] == "Božena Němcová"      # navázalo přes těžiště
    assert a.history[1]["answer"] == "2. května 1818"
    assert a.history[1]["gravity"] == "Božena Němcová"    # trajektorie těžiště
    a.reset()
    assert a.history == [] and a.context.hottest() is None


def test_run_pattern_executes_direct_ql():
    from jellyai.answerer.pattern import Pattern
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    a = GraphAnswerer(g, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))
    topic, values, fact = a.run_pattern(Pattern("napsat", [("obj", "Babička")],
                                                "subj", "person"))
    assert values == ["Božena Němcová"] and fact.predicate == "napsat"
    topic, values, fact = a.run_pattern(Pattern("napsat",
                                                [("subj", "Božena Němcová"),
                                                 ("obj", "Babička")], None, None))
    assert values == ["Ano"]
