import sys


def test_import_jellyai_does_not_import_viewbase():
    for mod in list(sys.modules):
        if mod.startswith("viewbase") or mod.startswith("networkx"):
            del sys.modules[mod]
    import jellyai  # noqa: F401
    assert not any(m.startswith("viewbase") for m in sys.modules)
    assert not any(m.startswith("networkx") for m in sys.modules)
