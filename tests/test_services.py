import os
import socket

import pytest

from config import ServicesConfig
from jellyai.ufal_client import UfalClient

_defaults = ServicesConfig()
_has_nametag = os.path.exists(_defaults.nametag_model)
_has_morpho = os.path.exists(_defaults.morphodita_model)
_has_udpipe = os.path.exists(_defaults.udpipe_model)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cfg():
    # volné porty per test, ať se běhy nervou o obsazené porty
    return ServicesConfig(nametag_port=_free_port(), udpipe_port=_free_port(),
                          morpho_port=_free_port())


@pytest.mark.skipif(not _has_nametag, reason="NameTag model není stažený")
def test_nametag_service_finds_entity():
    client = UfalClient(_cfg())
    try:
        ents = client.entities("Karel Čapek napsal knihu.")
        assert any("Čapek" in e["text"] for e in ents)
    finally:
        client.close()


@pytest.mark.skipif(not _has_morpho, reason="MorphoDiTa model není stažený")
def test_morpho_service_inflects():
    client = UfalClient(_cfg())
    try:
        forms = client.generate("Praha", "NNFS2-----A----")  # genitiv → Prahy
        assert "Prahy" in forms
    finally:
        client.close()


@pytest.mark.skipif(not _has_udpipe, reason="UDPipe model není stažený")
def test_udpipe_service_parses_subject():
    client = UfalClient(_cfg())
    try:
        sentences = client.parse("Karel Čapek napsal knihu.")
        assert sentences and any(t["deprel"] == "nsubj" for t in sentences[0])
    finally:
        client.close()
