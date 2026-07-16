"""Normalizace tokenizace — tečkované zkratky patternem, ne výčtem.

Tokenizace trhá „R.U.R." na R/./U/./R/. (lemma dokonce halucinuje U→United)
a NER vrací paskvil „R.U". Oprava na jediném chokepointu (UfalClient), takže
korpus i otázky projdou týmž kódem.
"""

from jellyai.normalize import merge_abbreviations, expand_abbreviation_entities


def test_merge_dotted_abbreviation_tokens():
    # reálný tvar z anotace wiki_r.u.r. (zjednodušený): R.U.R. je drama
    sent = [
        {"form": "R", "lemma": "R", "upos": "PROPN", "head": 8, "deprel": "nsubj", "start": 0, "end": 1},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 1, "end": 2},
        {"form": "U", "lemma": "United", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 2, "end": 3},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 3, "deprel": "punct", "start": 3, "end": 4},
        {"form": "R", "lemma": "R", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 4, "end": 5},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 5, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 8, "deprel": "cop", "start": 7, "end": 9},
        {"form": "drama", "lemma": "drama", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 15},
    ]
    merged = merge_abbreviations([sent])[0]
    assert [t["form"] for t in merged] == ["R.U.R.", "je", "drama"]
    rur = merged[0]
    assert rur["lemma"] == "R.U.R." and rur["upos"] == "PROPN"
    assert rur["deprel"] == "nsubj" and rur["head"] == 3      # přemapováno na drama
    assert (rur["start"], rur["end"]) == (0, 6)
    assert merged[1]["head"] == 3 and merged[2]["head"] == 0


def test_single_initial_stays_untouched():
    """„K. Čapek" — jediný pár ⟨písmeno⟩⟨.⟩ není zkratkový běh (iniciála)."""
    sent = [
        {"form": "K", "lemma": "K", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 1},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 1, "end": 2},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 3, "end": 8},
    ]
    merged = merge_abbreviations([sent])[0]
    assert [t["form"] for t in merged] == ["K", ".", "Čapek"]


def test_spaced_letters_do_not_merge():
    """Písmena s mezerami („a . b ." — výčtové odrážky) se neslučují — běh
    vyžaduje těsně navazující offsety."""
    sent = [
        {"form": "a", "lemma": "a", "upos": "PROPN", "head": 0, "deprel": "root", "start": 0, "end": 1},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 2, "end": 3},
        {"form": "b", "lemma": "b", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 4, "end": 5},
        {"form": ".", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct", "start": 6, "end": 7},
    ]
    merged = merge_abbreviations([sent])[0]
    assert len(merged) == 4


def test_expand_abbreviation_entities():
    text = "R.U.R. je drama Karla Čapka"
    entities = [{"text": "R.U", "type": "P", "start": 0, "end": 3},
                {"text": "R.", "type": "pf", "start": 0, "end": 2},
                {"text": "Karla Čapka", "type": "P", "start": 16, "end": 27}]
    out = expand_abbreviation_entities(text, entities)
    assert {"text": "R.U.R.", "type": "P", "start": 0, "end": 6} in out
    assert all(e["text"] != "R.U" for e in out)
    assert any(e["text"] == "Karla Čapka" for e in out)      # netknuté zůstávají
    # duplicitní expandované fragmenty se srazí (drží se první = kontejner)
    assert len([e for e in out if (e["start"], e["end"]) == (0, 6)]) == 1
