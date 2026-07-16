def test_template_composer_joins_facts():
    from jellyai.answerer.composer import TemplateComposer
    from jellyai.graph.extract import make_fact, Participant
    facts = [
        make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                              Participant("num", "1890", "number"),
                              Participant("loc", "Malých Svatoňovicích", "geo")]),
        make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                             Participant("obj", "R.U.R.", "concept")]),
    ]
    text = TemplateComposer().compose("kdo je Karel Čapek?", facts)
    assert "Karel Čapek" in text and "1890" in text and "R.U.R." in text
    assert text.strip().endswith(".")


def test_template_composer_empty():
    from jellyai.answerer.composer import TemplateComposer
    assert TemplateComposer().compose("?", []) == ""


def test_template_composer_is_composer_port():
    from jellyai.answerer.composer import TemplateComposer
    from jellyai.ports import Composer
    assert isinstance(TemplateComposer(), Composer)
