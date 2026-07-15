from config import AnswererConfig
from jellyai.chunker import Passage
from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.extractive import ExtractiveAnswerer, explain


def _retrieved():
    p = Passage("rur", 0,
                "Roboty vyráběla firma Rossumovy univerzální roboty. "
                "Starý Rossum je vynalezl.", 0, 2)
    return [(p, 1.0)]


def test_extractive_picks_relevant_sentence():
    a = ExtractiveAnswerer(AnswererConfig(template=False))
    ans = a.answer("kdo vyráběl roboty", _retrieved())
    assert isinstance(ans, Answer)
    assert "roboty" in ans.text.lower()
    assert ans.sources == ["rur#0"]
    assert ans.score > 0


def test_extractive_template_prefix():
    a = ExtractiveAnswerer(AnswererConfig(template=True))
    ans = a.answer("kdo vyráběl roboty", _retrieved())
    assert ans.text.startswith("Podle textu:")


def test_extractive_no_results():
    a = ExtractiveAnswerer(AnswererConfig())
    ans = a.answer("cokoliv", [])
    assert ans.sources == []
    assert "nenašel" in ans.text.lower()


def test_base_is_abstract():
    try:
        Answerer().answer("q", [])
        assert False, "mělo vyhodit NotImplementedError"
    except NotImplementedError:
        pass


def test_explain_nonempty():
    assert explain().strip()
