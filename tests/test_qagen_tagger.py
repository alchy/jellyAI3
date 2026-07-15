from qagen.tagger import Entity, Token, FakeTagger


def test_fake_tagger_returns_canned():
    s1 = "Roboty vynalezl starý Rossum."
    s2 = "Sluneční soustava má osm planet."
    ft = FakeTagger(
        entities={s1: [Entity("starý Rossum", "P", 16, 28)]},
        tokens={s2: [Token("osm", "osm", "C", 21, 24)]},
    )
    assert ft.entities(s1)[0].text == "starý Rossum"
    assert ft.entities(s1)[0].type == "P"
    assert ft.entities("neznámá věta") == []
    assert ft.tokens(s2)[0].pos == "C"
    assert ft.tokens("neznámá věta") == []
