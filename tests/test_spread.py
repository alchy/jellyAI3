"""Testy rozprostření teploty po tokenech (pseudo-attention na úrovni slov)."""

from jellyai.graph.spread import spread_field, entity_candidates, _neighbor_weights


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


def test_directional_weight_favours_chosen_side():
    """fwd > back → prostřední token dostane víc od pravého souseda než od levého."""
    tokens = [_tok("a", "NOUN", 2), _tok("b", "NOUN", 0), _tok("c", "NOUN", 2)]
    weights = _neighbor_weights(tokens, 1, 1.5, False, back=0.5, fwd=2.0)
    assert weights[1][2] > weights[1][0]                 # pravý soused silnější


def test_direction_changes_landscape():
    """Autoregresní (zpětné) vs dopředné vážení dá jinou teplotní krajinu."""
    tokens = [_tok("Čapek", "PROPN", 2), _tok("napsal", "VERB", 0),
              _tok("hru", "NOUN", 2), _tok("R.U.R.", "PROPN", 3)]
    backward = spread_field(tokens, back=2.0, fwd=0.5)
    forward = spread_field(tokens, back=0.5, fwd=2.0)
    assert backward != forward


def _wtok(form, upos, head, lemma=None):
    return {"form": form, "lemma": lemma or form, "upos": upos, "head": head}


def test_entity_candidate_recovers_missing_title():
    """Role ②: „R.U.R." (chybí, réma za napsal/hru) → kandidát na doplnění."""
    tokens = [_wtok("Karel", "PROPN", 2), _wtok("Čapek", "PROPN", 3),
              _wtok("napsal", "VERB", 0, "napsat"), _wtok("hru", "NOUN", 3, "hra"),
              _wtok("R.U.R.", "PROPN", 4)]
    cands = entity_candidates(tokens, known={"Čapek", "Karel"})
    assert "R.U.R." in cands


def test_entity_candidate_skips_known_and_subject():
    """Podmět (téma, chladnější, bez work-kontextu vlevo) a známé uzly se nepřidávají."""
    tokens = [_wtok("Karel", "PROPN", 2), _wtok("Čapek", "PROPN", 3),
              _wtok("napsal", "VERB", 0, "napsat"), _wtok("hru", "NOUN", 3, "hra"),
              _wtok("R.U.R.", "PROPN", 4)]
    cands = entity_candidates(tokens, known={"Čapek"})
    assert "Karel" not in cands            # podmět vlevo, žádný work-kontext před ním
    assert "Čapek" not in cands            # známý uzel


def test_empty_sentence():
    assert spread_field([]) == []
