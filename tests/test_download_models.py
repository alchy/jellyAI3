from qagen.download_models import MODELS


def test_models_cover_all_three():
    dests = [m["dest"] for m in MODELS]
    assert "data/models/czech-cnec.ner" in dests        # NameTag
    assert "data/models/czech-morfflex.tagger" in dests  # MorphoDiTa
    assert "data/models/udpipe-czech.model" in dests     # UDPipe
    # každá položka má buď přímý bitstream, nebo zip member
    for model in MODELS:
        assert ("bitstream" in model) or ("member" in model)
        assert model["handle"] and model["dest"]
