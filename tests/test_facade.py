def test_jelly_injects_answerer_and_answers():
    from jellyai.facade import Jelly
    from jellyai.answerer.base import Answer

    class FakeAnswerer:
        def __init__(self):
            self.context = None
            self.history = []

        def answer(self, question, retrieved, **kw):
            return Answer(text="Božena Němcová", sources=["graf"], score=1.0)

    j = Jelly(answerer=FakeAnswerer())
    ans = j.ask("kdo napsal Babičku?")
    assert ans.text == "Božena Němcová"


def test_jelly_context_manager_closes_corpus():
    from jellyai.facade import Jelly
    closed = {"v": False}

    class FakeCorpus:
        def stop(self):
            closed["v"] = True

    with Jelly(corpus=FakeCorpus()):
        pass
    assert closed["v"] is True


def test_jelly_requires_answerer():
    from jellyai.facade import Jelly
    from jellyai.errors import JellyError
    try:
        Jelly().ask("kdo?")
        assert False, "mělo hodit JellyError"
    except JellyError as err:
        assert "Answerer" in str(err)
