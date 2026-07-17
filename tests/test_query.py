"""Template query parser — otázka → vzor → pseudo-QL Query, bez UDPipe.

Slovník dotazu je sám graf (predikáty + uzly); diakritika ani mis-tagging
UDPipe nehrají roli. Korpus jen staví graf; dotaz jede šablonami.
"""

from jellyai.answerer.query import build_query
from jellyai.answerer.pattern import SubQuery


# slovník grafu, který parser vidí (predikáty, které fakty nesou)
PREDS = {"napsat", "narodit", "být", "bratr", "zemřít"}


def test_kdo_predicate_entity():
    """"Kdo napsal Babičku?" → predikát napsat, known obj=Babička, díra subj."""
    q = build_query("Kdo napsal Babičku?", PREDS)
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role == "subj"
    assert ("obj", "Babičku") in q.pattern.known
    assert q.qtype == "Kdo" and q.verb_lemma == "napsat"
    assert q.gender == "Masc"          # „napsal" — l-ové příčestí mužské


def test_kde_diacritic_free():
    """Bez diakritiky: „Kde se narodil Jezis?" → narodit, díra loc."""
    q = build_query("Kde se narodil Jezis?", PREDS)
    assert q.pattern.predicate == "narodit"
    assert q.pattern.hole_role == "loc"
    assert ("subj", "Jezis") in q.pattern.known


def test_kdo_je_identity():
    """"Kdo je Karel Capek?" → identita být, díra pred."""
    q = build_query("Kdo je Karel Capek?", PREDS)
    assert q.pattern.predicate == "být"
    assert q.pattern.hole_role == "pred"
    assert ("subj", "Karel Capek") in q.pattern.known


def test_relational_recursion():
    """"Kdo byl bratr Karla Capka?" → predikát bratr (vztahové jméno), díra
    subj, known obj = Karel Capek (rekurzivní předmět)."""
    q = build_query("Kdo byl bratr Karla Capka?", PREDS)
    assert q.pattern.predicate == "bratr"
    assert q.pattern.hole_role == "subj"
    assert ("obj", "Karla Capka") in q.pattern.known


def test_nested_subquery():
    """"Kdo byl bratr autora ktery napsal RUR?" → bratr(obj=SubQuery(napsat,
    obj=RUR)) — dotaz expanduje rekurzivně přes vztažné „který"."""
    q = build_query("Kdo byl bratr autora ktery napsal RUR?", PREDS)
    assert q.pattern.predicate == "bratr"
    sub = next(k for _, k in q.pattern.known if isinstance(k, SubQuery))
    assert sub.predicate == "napsat"
    assert ("obj", "RUR") in sub.known


def test_non_question_returns_none():
    """Věta bez otazníku není dotaz → None (parser řeší jen otázky)."""
    assert build_query("Karel napsal Babičku.", PREDS) is None


def _nodes(*spans):
    """Fake slovník grafu: is_node = přesná množina povolených rozpětí."""
    allowed = set(spans)
    return lambda span: span in allowed


def test_greedy_splits_glued_entities():
    """Dvě entity za sebou bez hranice se rozdělí na maximální uzlová rozpětí."""
    is_node = _nodes("Karel Čapek", "Válku s mloky")
    q = build_query("Co napsal Karel Čapek Válku s mloky?", {"napsat"}, is_node)
    spans = [t for _, t in q.pattern.known]
    assert spans == ["Karel Čapek", "Válku s mloky"]


def test_skip_word_bridges_title():
    """Předložka uvnitř titulu běh NErozbíjí: „Válku s mloky" je jedno rozpětí."""
    is_node = _nodes("Válku s mloky")
    q = build_query("Kdo napsal Válku s mloky?", {"napsat"}, is_node)
    assert ("obj", "Válku s mloky") in q.pattern.known


def test_trailing_orphan_after_match_dropped():
    """Graf zná jen „Válku" — „s mloky" je pokračování titulu, zahodí se."""
    is_node = _nodes("Válku")
    q = build_query("Kdo napsal Válku s mloky?", {"napsat"}, is_node)
    assert ("obj", "Válku") in q.pattern.known


def test_relation_question_builds_relation_pattern():
    """„Jaký měl vztah k Martě?" — „vztah" není entita, ale OPERÁTOR
    spojení (jazyková tabulka relation_query_nouns): pattern s dírou
    „relation" a Martou jako známým; druhého účastníka (elidovaný podmět)
    doplní answerer z těžiště. Marta se NESMÍ spolknout jako ocas titulu."""
    is_node = _nodes("vztah", "Martě")
    q = build_query("Jaký měl vztah k Martě?", {"měl"}, is_node)
    assert q.pattern.hole_role == "relation"
    spans = [k for _, k in q.pattern.known]
    assert spans == ["Martě"]


def test_leading_orphan_returns_none():
    """„Ludvík" graf nezná → vzor nelze bezpečně sestavit → None (žádné
    hádání přes „Němec", které by trefilo Němcovou)."""
    is_node = _nodes("Němec")
    assert build_query("Kdo je Ludvík Němec?", {"být"}, is_node) is None


def test_without_is_node_keeps_old_behavior():
    q = build_query("Kdo napsal Babičku?", PREDS, None)
    assert ("obj", "Babičku") in q.pattern.known


def test_yes_no_question_builds_existence_pattern():
    """„Napsal Karel Čapek Válku s mloky?" → predikát + všechny entity known,
    díra žádná (existenční test), qtype None."""
    is_node = _nodes("Karel Čapek", "Válku s mloky")
    q = build_query("Napsal Karel Čapek Válku s mloky?", {"napsat"}, is_node)
    assert q is not None and q.qtype is None
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role is None
    assert ("subj", "Karel Čapek") in q.pattern.known
    assert ("obj", "Válku s mloky") in q.pattern.known


def test_yes_no_needs_leading_verb():
    """Bez tázacího slova a bez počátečního slovesa vzor nevznikne (None)."""
    is_node = _nodes("Karel Čapek")
    assert build_query("Karel Čapek napsal?", {"napsat"}, is_node) is None


def test_type_filter_roles():
    """„Jakou hru napsal Karel Čapek?" → filtr obj=hru, téma subj=Karel Čapek."""
    is_node = _nodes("hru", "Karel Čapek")
    q = build_query("Jakou hru napsal Karel Čapek?", {"napsat"}, is_node)
    assert q.pattern.hole_role == "attr"
    assert ("obj", "hru") in q.pattern.known
    assert ("subj", "Karel Čapek") in q.pattern.known


def test_type_filter_prodrop_keeps_obj_role():
    """„Jakou hru napsal?" (pro-drop) → jediný known jako obj, aby
    _fill_subject směl doplnit podmět z konverzačního těžiště."""
    is_node = _nodes("hru")
    q = build_query("Jakou hru napsal?", {"napsat"}, is_node)
    assert q.pattern.known == [("obj", "hru")]
    assert q.pattern.hole_role == "attr" and q.gender == "Masc"


def test_relational_under_verb_becomes_subquery():
    """„Kde se narodil bratr Karla Čapka?" → narodit(subj=SubQuery(bratr,
    obj=Karla Čapka), díra loc) — vztah je vnořený dotaz, sloveso vládne."""
    q = build_query("Kde se narodil bratr Karla Čapka?", {"narodit", "bratr"})
    assert q.pattern.predicate == "narodit" and q.pattern.hole_role == "loc"
    role, sub = q.pattern.known[0]
    assert role == "subj" and isinstance(sub, SubQuery)
    assert sub.predicate == "bratr" and ("obj", "Karla Čapka") in sub.known


def test_bare_relational_without_verb_unchanged():
    q = build_query("Kdo byl bratr Karla Capka?", PREDS)
    assert q.pattern.predicate == "bratr"
    assert ("obj", "Karla Capka") in q.pattern.known


def test_date_part_drill():
    """„V kterém roce se narodila BN?" → date_part=rok (2-skokový drill),
    „roce" není účastník."""
    q = build_query("V kterém roce se narodila Božena Němcová?", {"narodit"})
    assert q.pattern.predicate == "narodit" and q.pattern.date_part == "rok"
    assert ("subj", "Božena Němcová") in q.pattern.known
    assert q.gender == "Fem"


def test_generic_event_verb_from_table():
    """„Co se stalo s rodinou?" → predikát stát (tvar z jazykové tabulky,
    prefix nestačí), rodina jako téma."""
    q = build_query("Co se stalo s rodinou?", set())
    assert q.pattern.predicate == "stát"
    assert ("subj", "rodinou") in q.pattern.known


def test_reverse_date_question_stays_none():
    """„Co se stalo v listopadu 1848?" bez uzlového rozpětí → None; reverzní
    lookup (datum→děj) zůstává na answereru (spec §7)."""
    assert build_query("Co se stalo v listopadu 1848?", set(), _nodes()) is None


def test_predicate_synonym_from_language_table():
    """„Kde žili…?" na graf s predikátem „bydlet": tvar se páruje i proti
    lemmatům z predicate_synonyms; vrátí se synonymum („žít") — expanzi na
    bydlet-fakty dělá answererův _synonym_ring (spec 4.2)."""
    q = build_query("Kde žili Čapkovi?", {"bydlet"}, _nodes("Čapkovi"))
    assert q.pattern.predicate == "žít"
    assert q.pattern.hole_role == "loc"


def test_diacritic_free_identity():
    q = build_query("Kdo je jezis?", set(), _nodes("jezis"))
    assert q.pattern.predicate == "být"
    assert ("subj", "jezis") in q.pattern.known


def test_diacritic_free_verb_and_entity():
    q = build_query("Kde se narodil Jezis?", {"narodit"}, _nodes("Jezis"))
    assert q.pattern.predicate == "narodit" and q.pattern.hole_role == "loc"
    assert ("subj", "Jezis") in q.pattern.known


def test_diacritic_free_nested_subquery():
    is_node = _nodes("autora", "R.U.R.")
    q = build_query("Kdo byl bratr autora, ktery napsal R.U.R.?",
                    {"bratr", "napsat"}, is_node)
    sub = next(k for _, k in q.pattern.known if isinstance(k, SubQuery))
    assert sub.predicate == "napsat" and ("obj", "R.U.R.") in sub.known


def test_query_gender_from_verb_form():
    q = build_query("Kdy se narodila Božena Němcová?", {"narodit"})
    assert q.gender == "Fem" and q.qtype == "Kdy"


def test_query_copula_flag():
    q = build_query("Kdo je Karel Capek?", PREDS)
    assert q.is_copula is True and q.verb_lemma is None
