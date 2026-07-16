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
