def test_demo_runs_without_models():
    import jellyai
    result = jellyai.demo(verbose=False)
    assert result["kdo napsal Babičku?"] == "Božena Němcová"
    assert result["kdy se narodila?"] == "1818"        # navázalo přes těžiště
