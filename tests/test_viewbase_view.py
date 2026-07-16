import importlib.util
import sys


def test_module_import_does_not_import_viewbase():
    for m in list(sys.modules):
        if m.startswith("viewbase"):
            del sys.modules[m]
    import jellyai.viz.viewbase_view  # noqa: F401
    assert not any(m.startswith("viewbase") for m in sys.modules)


def test_missing_viewbase_raises_actionable():
    if importlib.util.find_spec("viewbase") is not None:
        return  # viewBase je nainstalovaný → tenhle hermetický test přeskoč
    from jellyai.viz.viewbase_view import ViewBaseView
    from jellyai.errors import JellyError
    try:
        ViewBaseView()
        assert False, "mělo hodit JellyError"
    except JellyError as err:
        assert "viewbase" in str(err).lower()
