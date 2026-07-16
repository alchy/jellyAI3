from jellyai.chunker import Passage
from jellyai.ufal_client import FakeUfalClient
from jellyai.annotate import annotate_passages, save_annotations, load_annotations


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
