def test_corpus_tools_lifecycle_no_spawn():
    from config import ServicesConfig
    from jellyai.corpus import CorpusTools
    # bez volání parse/entities se žádná služba nespustí → stop je no-op
    tools = CorpusTools(ServicesConfig())
    tools.stop()                       # bez chyby
    with CorpusTools(ServicesConfig()) as t:
        assert t is not None           # context manager vrací self


def test_corpus_tools_is_corpus_port():
    from config import ServicesConfig
    from jellyai.corpus import CorpusTools
    from jellyai.ports import CorpusPort
    assert isinstance(CorpusTools(ServicesConfig()), CorpusPort)   # splní port
