from config import Config, AnswererConfig
from jellyai.pipeline import _make_answerer
from jellyai.answerer.template import TemplateAnswerer
from jellyai.answerer.extractive import ExtractiveAnswerer


def test_make_answerer_template_mode():
    # vytvoření TemplateAnswereru je líné — ÚFAL klient službu nespustí, dokud
    # nepřijde první dotaz, takže tenhle test je hermetický.
    answerer = _make_answerer(Config(answerer=AnswererConfig(mode="template")))
    assert isinstance(answerer, TemplateAnswerer)


def test_make_answerer_extractive_default():
    answerer = _make_answerer(Config())
    assert isinstance(answerer, ExtractiveAnswerer)
