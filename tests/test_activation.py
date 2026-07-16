from jellyai.graph.activation import ActivationField


def test_hottest_recent_warm_wins():
    f = ActivationField(decay=0.5)
    f.warm("A", 1.0)
    f.warm("B", 0.5)
    assert f.hottest() == "A"
    f.step()                 # A=0.5, B=0.25
    f.warm("B", 0.4)         # B=0.65 > A=0.5
    assert f.hottest() == "B"


def test_empty_field_hottest_is_none():
    assert ActivationField().hottest() is None


def test_floor_forgets_cold_keys():
    f = ActivationField(decay=0.1, floor=0.05)
    f.warm("A", 0.1)
    f.step()                 # 0.01 < floor → zapomenuto
    assert f.hottest() is None
