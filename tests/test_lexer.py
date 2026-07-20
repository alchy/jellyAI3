"""Lexer — JEDEN určovač druhů slov (#46 fáze 1, spec vzorová gramatika).

Token nese MNOŽINU hypotéz tříd („byt" je spona i substantivum, „Můj"
přivlastňovací i kapitalizované jméno) — rozhodne až stavba věty, lexer
dvojznačnost jen poctivě přizná. Třídy se počítají z jazykových tabulek
cs.json; fáze 1 zrcadlí dnešní kontroly parse_statement (žádná změna
chování), sekvenční vzory přijdou ve fázi 2–3.
"""

from jellyai.lang.lexer import classify, tokenize


def _classes(text, form, is_node=None):
    return next(t.classes for t in classify(text, is_node=is_node)
                if t.form == form)


def test_token_carries_multiple_hypotheses():
    assert {"privlastnovaci", "jmeno"} <= _classes("Můj email…", "Můj")
    assert {"otaz", "jmeno"} <= _classes("Kdo je Roník.", "Kdo")
    # „byt" je po deakcentaci spona „být" I substantivum — lexer přizná obojí
    assert "spona" in _classes("V Plzni má pronajatý byt.", "byt")


def test_verb_classes_mirror_current_behavior():
    assert "sloveso_fin" in _classes("Venku prší.", "prší")
    assert "sloveso_fin" in _classes("Venku prsi.", "prsi")   # katalog
    # dvojznakové „jí" dnešní guard nepustí (gap #45 — fáze 1 nemění chování)
    assert "sloveso_fin" not in _classes("Roník jí stravu.", "jí")
    tagged = {t.form: t for t in classify("Dnes jsem měl knedlíky.")}
    assert "l_tvar" in tagged["měl"].classes
    assert tagged["měl"].l_stem == "měl"
    marcela = {t.form: t for t in classify("Marcela bydlela tady.")}
    assert marcela["Marcela"].l_stem == "marcel"    # l-lookalike hypotéza
    assert "jmeno" in marcela["Marcela"].classes


def test_entity_veto_blocks_finite_verb():
    assert "sloveso_fin" in _classes("U nádraží prší.", "prší",
                                     is_node=lambda t: t == "nádraží")
    assert "sloveso_fin" not in _classes("U nádraží prší.", "nádraží",
                                         is_node=lambda t: t == "nádraží")


def test_function_temporal_and_particle_classes():
    assert "funkcni" in _classes("Karel bydlí v Praze.", "v")
    assert "cas" in _classes("Včera jsem měl guláš.", "Včera")
    assert "castice" in _classes("Niki je však v Plzni.", "však")
    assert "prvni_osoba" in _classes("Dnes jsem měl knedlíky.", "jsem")
    assert "spona" in _classes("Roník je pes.", "je")
    assert "potvrzeni" in _classes("Ano, měl rád knedlíky.", "Ano")


def test_email_survives_tokenization_whole():
    tokens = tokenize("Můj email je jindra@example.com.")
    assert "jindra@example.com" in tokens
    assert "email" in _classes("Můj email je jindra@example.com.",
                               "jindra@example.com")


def test_trailing_dot_stripped_but_abbreviations_kept():
    assert tokenize("Venku prší.") == ["Venku", "prší"]
    assert "R.U.R." in tokenize("Napsal R.U.R. Čapek")


def test_dative_hypothesis_class():
    """#55: koncovky -ovi/-ům nesou hypotézu DATIVU (adresát) — lexer
    nerozhoduje, jen poctivě přizná; krátké tvary a jiné koncovky ne."""
    from jellyai.lang.lexer import classify

    tagged = {t.form: t.classes for t in
              classify("Co řekl Ježíšovi a učedníkům?")}
    assert "dativ" in tagged["Ježíšovi"]
    assert "dativ" in tagged["učedníkům"]
    assert "dativ" not in tagged["Co"] and "dativ" not in tagged["řekl"]


def test_participium_hypoteza():
    """Pasivní participium (krátký tvar) dostává třídu `participium`
    z tabulky koncovek — hypotéza, rozhodne až karta (P1b, dávka D)."""
    toks = classify("Kde byl Ježíš pokřtěn")
    krest = next(t for t in toks if t.form == "pokřtěn")
    assert "participium" in krest.classes
    toks = classify("Kdy byla vydána korespondence")
    vydana = next(t for t in toks if t.form == "vydána")
    assert "participium" in vydana.classes


def test_participium_kratka_slova_ne():
    """Krátká slova („den") participium nejsou — minimální délka."""
    toks = classify("Který den byl pátek")
    den = next(t for t in toks if t.form == "den")
    assert "participium" not in den.classes
