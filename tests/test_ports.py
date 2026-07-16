def test_ports_are_structural():
    from jellyai.ports import QuestionAnalyzer, Composer

    class MyAnalyzer:
        def analyze(self, question):
            return None
    assert isinstance(MyAnalyzer(), QuestionAnalyzer)

    class MyComposer:
        def compose(self, question, facts):
            return "text"
    assert isinstance(MyComposer(), Composer)


def test_existing_blocks_satisfy_ports():
    from jellyai.ports import Composer
    from jellyai.answerer.composer import TemplateComposer
    # výchozí kompozitor (Task 6) splní port Composer — ověříme až po jeho vzniku
    try:
        assert isinstance(TemplateComposer(), Composer)
    except ImportError:
        pass
