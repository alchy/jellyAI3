import os

import pytest

from config import QagenConfig
from qagen.answers import candidates

_cfg = QagenConfig()
_models_present = os.path.exists(_cfg.nametag_model)


@pytest.mark.skipif(not _models_present, reason="NameTag model není stažený")
def test_ufal_tagger_finds_person():
    from qagen.tagger import UfalTagger
    tagger = UfalTagger(_cfg.nametag_model)
    cands = candidates("Roboty vynalezl starý Rossum.", tagger, _cfg)
    assert any(c.qtype == "Kdo" for c in cands)
