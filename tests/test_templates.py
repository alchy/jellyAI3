from jellyai.templates import fill, target_case


def test_fill_bare_answer():
    assert fill("Kdo", "Božena Němcová") == "Božena Němcová"
    assert fill("Co", "Babička") == "Babička"


def test_target_case():
    assert target_case("Kdo") == "1"     # 1. pád (nominativ)
    assert target_case("Co") == "1"
    assert target_case("Kdy") is None     # data se neskloňují
    assert target_case("Kolik") is None   # čísla se neskloňují
    assert target_case("Neznámý") is None


def test_new_question_types():
    assert target_case("Jaký") is None    # přísudek/adjektivum ponecháme
    assert target_case("Který") == "1"
    assert target_case("Čí") == "1"
    assert fill("Jaký", "významná spisovatelka") == "významná spisovatelka"
