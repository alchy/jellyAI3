import logging


def test_jelly_error_actionable():
    from jellyai.errors import JellyError, ModelsMissingError
    err = ModelsMissingError("data/models")
    assert isinstance(err, JellyError)
    assert "qa-models" in str(err)          # hláška říká, jak opravit


def test_logger_silent_by_default_and_debug_toggles():
    from jellyai.logs import get_logger, set_debug
    log = get_logger()
    assert any(isinstance(h, logging.NullHandler) for h in log.handlers)
    set_debug(True)
    assert log.level == logging.DEBUG
    set_debug(False)
