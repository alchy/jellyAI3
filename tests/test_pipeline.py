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


def test_build_retriever_picks_sentence_for_sentence_granularity():
    from config import Config, RetrieverConfig
    from jellyai.loader import Document
    from jellyai.pipeline import _build_retriever
    from jellyai.sentence_retriever import SentenceRetriever
    from jellyai.retriever import Retriever
    docs = [Document("d", "d", "Alfa jedna. Klíč leží tady. Gama tři.")]

    cfg_sent = Config(); cfg_sent.retriever = RetrieverConfig(granularity="sentence")
    assert isinstance(_build_retriever(cfg_sent, docs), SentenceRetriever)

    cfg_pass = Config(); cfg_pass.retriever = RetrieverConfig(granularity="passage")
    assert isinstance(_build_retriever(cfg_pass, docs), Retriever)
