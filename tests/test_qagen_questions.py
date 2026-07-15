from qagen.answers import Candidate
from qagen.questions import build_question


def test_build_question_person():
    s = "Roboty vynalezl starý Rossum."
    c = Candidate(answer="starý Rossum", qtype="Kdo", start=16, end=28)
    assert build_question(s, c) == "Kdo roboty vynalezl?"


def test_build_question_number():
    s = "Sluneční soustava má osm planet."
    c = Candidate(answer="osm", qtype="Kolik", start=21, end=24)
    assert build_question(s, c) == "Kolik sluneční soustava má planet?"
