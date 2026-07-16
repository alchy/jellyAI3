"""Testy rozprostření teploty po tokenech (pseudo-attention na úrovni slov)."""

from jellyai.graph.spread import spread_field


def _tok(form, upos, head):
    return {"form": form, "upos": upos, "head": head}


def test_content_hotter_than_function():
    """Obsahová slova skončí teplejší než funkční (spojka)."""
    tokens = [_tok("Karel", "PROPN", 3), _tok("a", "CCONJ", 3),
              _tok("psal", "VERB", 0)]
    heat = spread_field(tokens)
    assert heat[1] < heat[0] and heat[1] < heat[2]     # „a" je nejchladnější


def test_missing_title_becomes_hot():
    """Titul R.U.R. (předmět slovesa, obklopený obsahem) je horký → kandidát."""
    tokens = [_tok("Čapek", "PROPN", 2), _tok("napsal", "VERB", 0),
              _tok("hru", "NOUN", 2), _tok("R.U.R.", "PROPN", 3)]
    heat = spread_field(tokens)
    assert heat[3] > 0.5                                # R.U.R. je horké


def test_spreading_warms_token_closer_to_hot():
    """Difúze šíří teplo: chladná spojka blíž horkému slovesu skončí tepleji."""
    # obě spojky jsou chladné; „a" je hned u horkého „psal", „i" až za „a"
    tokens = [_tok("psal", "VERB", 0), _tok("a", "CCONJ", 1), _tok("i", "CCONJ", 2)]
    heat = spread_field(tokens)
    assert heat[1] > heat[2]                              # bližší k horkému = tepleji


def test_empty_sentence():
    assert spread_field([]) == []
