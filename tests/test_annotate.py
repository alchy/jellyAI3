from jellyai.chunker import Passage
from jellyai.loader import Document
from jellyai.ufal_client import FakeUfalClient
from jellyai.annotate import (annotate_passages, annotate_documents,
                              save_annotations, load_annotations)


def test_annotate_and_roundtrip(tmp_path):
    passage = Passage("wiki", 2, "Text.", 0, 1)
    client = FakeUfalClient(
        entities={"Text.": [{"text": "Rossum", "type": "P", "start": 0, "end": 6}]},
        parse={"Text.": [[
            {"form": "Text", "lemma": "text", "upos": "NOUN",
             "head": 0, "deprel": "root", "start": 0, "end": 4}
        ]]},
    )
    ann = annotate_passages([passage], client)
    assert ("wiki", 2) in ann
    assert ann[("wiki", 2)]["entities"][0]["text"] == "Rossum"

    path = str(tmp_path / "ann.pkl")
    save_annotations(ann, path)
    loaded = load_annotations(path)
    assert loaded[("wiki", 2)]["sentences"][0][0]["deprel"] == "root"


def test_annotate_documents_keys_per_sentence_and_shifts_offsets():
    text = "Anna spí. Bere klobouk."
    docs = [Document("d", "d", text)]
    client = FakeUfalClient(
        parse={
            "Anna spí.": [[
                {"form": "Anna", "lemma": "Anna", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 4},
                {"form": "spí", "lemma": "spát", "upos": "VERB", "head": 0, "deprel": "root", "start": 5, "end": 8},
            ]],
            "Bere klobouk.": [[
                {"form": "Bere", "lemma": "brát", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 4},
                {"form": "klobouk", "lemma": "klobouk", "upos": "NOUN", "head": 1, "deprel": "obj", "start": 5, "end": 12},
            ]],
        },
        entities={"Anna spí.": [{"text": "Anna", "type": "P", "start": 0, "end": 4}]},
    )
    ann = annotate_documents(docs, client)
    assert set(ann.keys()) == {("d", 0), ("d", 1)}
    # věta 0 beze změny (base 0)
    assert ann[("d", 0)]["sentences"][0][0]["start"] == 0
    assert ann[("d", 0)]["entities"][0]["start"] == 0
    # věta 1 posunutá o base = len("Anna spí.") + 1 = 10 → disjunktní od věty 0
    assert ann[("d", 1)]["sentences"][0][0]["start"] == 10
    assert ann[("d", 1)]["sentences"][0][0]["form"] == "Bere"


def test_person_entity_splits_on_case_mismatch():
    """NER na volném slovosledu lepí („Ježíš Duchem" Nom+Ins) — entita osoby
    se usekne v místě pádové neshody tokenů."""
    from jellyai.annotate import _trim_case_mismatch
    entities = [{"text": "Ježíš Duchem", "type": "P", "start": 0, "end": 12}]
    sent = [
        {"form": "Ježíš", "upos": "PROPN", "start": 0, "end": 5,
         "feats": {"Case": "Nom"}},
        {"form": "Duchem", "upos": "NOUN", "start": 6, "end": 12,
         "feats": {"Case": "Ins"}},
    ]
    out = _trim_case_mismatch(entities, [sent])
    assert out[0]["text"] == "Ježíš" and out[0]["end"] == 5


def test_case_consistent_entity_untouched():
    """„Karla Čapka" (Gen+Gen) zůstává celé."""
    from jellyai.annotate import _trim_case_mismatch
    entities = [{"text": "Karla Čapka", "type": "P", "start": 0, "end": 11}]
    sent = [
        {"form": "Karla", "upos": "PROPN", "start": 0, "end": 5,
         "feats": {"Case": "Gen"}},
        {"form": "Čapka", "upos": "PROPN", "start": 6, "end": 11,
         "feats": {"Case": "Gen"}},
    ]
    out = _trim_case_mismatch(entities, [sent])
    assert out[0]["text"] == "Karla Čapka"
