from config import Config
from jellyai.pipeline import QAPipeline, explain


def _corpus(tmp_path):
    (tmp_path / "rur.txt").write_text(
        "Roboty vyráběla firma Rossumovy univerzální roboty. "
        "Starý Rossum je vynalezl. "
        "Helena přijela na ostrov. "
        "Domin jí ukázal továrnu.",
        encoding="utf-8",
    )
    return str(tmp_path)


def test_pipeline_end_to_end(tmp_path):
    pipe = QAPipeline.from_corpus(_corpus(tmp_path), Config())
    ans = pipe.ask("kdo vyráběl roboty")
    assert "roboty" in ans.text.lower()
    assert ans.sources and ans.sources[0].startswith("rur#")


def test_pipeline_no_answer(tmp_path):
    pipe = QAPipeline.from_corpus(_corpus(tmp_path), Config())
    ans = pipe.ask("kvantová chromodynamika")
    assert "nenašel" in ans.text.lower()


def test_pipeline_from_index(tmp_path):
    pipe = QAPipeline.from_corpus(_corpus(tmp_path), Config())
    index_path = str(tmp_path / "index.pkl")
    pipe.retriever.save(index_path)
    # Znovu postavená pipeline z uloženého indexu odpovídá stejně dobře.
    pipe2 = QAPipeline.from_index(index_path, Config())
    ans = pipe2.ask("kdo vyráběl roboty")
    assert "roboty" in ans.text.lower()
    assert ans.sources and ans.sources[0].startswith("rur#")


def test_explain_nonempty():
    assert explain().strip()
