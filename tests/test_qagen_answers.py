from config import QagenConfig
from qagen.tagger import Entity, Token, FakeTagger
from qagen.answers import candidates, Candidate


def test_candidate_from_person_entity():
    s = "Roboty vynalezl starý Rossum."
    ft = FakeTagger(entities={s: [Entity("starý Rossum", "P", 16, 28)]})
    cands = candidates(s, ft, QagenConfig())
    assert isinstance(cands[0], Candidate)
    assert cands[0].qtype == "Kdo"
    assert cands[0].answer == "starý Rossum"


def test_candidate_from_number_token():
    s = "Sluneční soustava má osm planet."
    ft = FakeTagger(tokens={s: [Token("osm", "osm", "C", 21, 24)]})
    cands = candidates(s, ft, QagenConfig())
    assert any(c.qtype == "Kolik" and c.answer == "osm" for c in cands)


def test_answer_equal_to_whole_sentence_skipped():
    s = "Rossum"
    ft = FakeTagger(entities={s: [Entity("Rossum", "P", 0, 6)]})
    assert candidates(s, ft, QagenConfig()) == []


def test_respects_max_answers():
    s = "A B C"
    ft = FakeTagger(entities={s: [
        Entity("A", "P", 0, 1), Entity("B", "P", 2, 3), Entity("C", "P", 4, 5),
    ]})
    cfg = QagenConfig(max_answers_per_sentence=2)
    assert len(candidates(s, ft, cfg)) == 2
