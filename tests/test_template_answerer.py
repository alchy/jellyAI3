from config import AnswererConfig
from jellyai.chunker import Passage
from jellyai.answerer.base import Answer
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.template import TemplateAnswerer, explain
from jellyai.ufal_client import FakeUfalClient


def _passage_annotation():
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]
    return {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
            "sentences": [sent]}


def _question_client():
    q = "kdo napsal Babičku?"
    return q, FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})


def test_template_answerer_person():
    q, client = _question_client()
    passage = Passage("wiki_bn", 5, "Božena Němcová napsala Babičku.", 0, 1)
    answerer = TemplateAnswerer(client, {("wiki_bn", 0): _passage_annotation()},
                                ExtractiveAnswerer(AnswererConfig()))
    ans = answerer.answer(q, [(passage, 1.0)])
    assert isinstance(ans, Answer)
    assert ans.text == "Božena Němcová"     # podmět v nominativu ze šablony
    assert ans.sources == ["wiki_bn#5"]


def test_template_answerer_falls_back_without_annotation():
    q, client = _question_client()
    passage = Passage("wiki_bn", 5, "Božena Němcová napsala Babičku.", 0, 1)
    answerer = TemplateAnswerer(client, {},  # žádné anotace → fallback
                                ExtractiveAnswerer(AnswererConfig()))
    ans = answerer.answer(q, [(passage, 1.0)])
    assert isinstance(ans, Answer)
    assert "Podle textu" in ans.text          # extraktivní fallback


def test_to_nominative_fixes_multiword_agreement():
    from jellyai.answerer.template import _to_nominative
    client = FakeUfalClient(
        analyze={"Boženy Němcové": [
            {"form": "Boženy", "lemma": "Božena", "tag": "NNFS2-----A----"},
            {"form": "Němcové", "lemma": "Němcová", "tag": "NNFS2-----A----"},
        ]},
        generate={
            ("Božena", "NNFS1-----A----"): ["Božena"],
            ("Němcová", "NNFS1-----A----"): ["Němcová"],
        },
    )
    assert _to_nominative("Boženy Němcové", client) == "Božena Němcová"


def test_to_nominative_without_data_returns_phrase():
    from jellyai.answerer.template import _to_nominative
    assert _to_nominative("cokoliv", FakeUfalClient()) == "cokoliv"


def test_copula_definition_not_tautology():
    q = "kdo je Rossum?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 4, "end": 6},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 7, "end": 13},
    ]]})
    passage = Passage("wiki", 1, "Rossum je vynálezce.", 0, 1)
    annotations = {("wiki", 0): {"entities": [{"text": "Rossum", "type": "P", "start": 0, "end": 6}],
                                 "sentences": [[
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 7, "end": 9},
        {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 19},
    ]]}}
    answerer = TemplateAnswerer(client, annotations, ExtractiveAnswerer(AnswererConfig()))
    result = answerer.answer(q, [(passage, 1.0)])
    assert result.text == "vynálezce"    # definice, ne tautologie „Rossum"


def test_annotation_assembled_across_window():
    # okno pokrývá věty 0..1; spona je až ve větě 1 → přísudek se najde
    q = "kdo je Rossum?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 4, "end": 6},
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 0, "deprel": "root", "start": 7, "end": 13},
    ]]})
    passage = Passage("wiki", 0, "Něco jiného. Rossum je vynálezce.", 0, 2)
    annotations = {
        ("wiki", 0): {"entities": [], "sentences": [[
            {"form": "Něco", "lemma": "něco", "upos": "PRON", "head": 0, "deprel": "root", "start": 0, "end": 4},
        ]]},
        ("wiki", 1): {"entities": [], "sentences": [[
            {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 13, "end": 19},
            {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 20, "end": 22},
            {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 23, "end": 32},
        ]]},
    }
    answerer = TemplateAnswerer(client, annotations, ExtractiveAnswerer(AnswererConfig()))
    result = answerer.answer(q, [(passage, 1.0)])
    assert result.text == "vynálezce"


def test_explain_nonempty():
    assert explain().strip()
