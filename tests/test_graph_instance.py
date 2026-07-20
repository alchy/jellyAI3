"""Instanční vrstva — jméno není entita (backlog #8, fáze 1).

Srůst jmenných střepů smí založit jen TEXTOVÉ tvrzení („X řečený/zvaný
Y" → fakt `jmenovat`); kontextový otisk srůst pouze potvrzuje. Měření
korpusu doložilo, že otisk sám identitu nerozliší (Ježíš–Nazaretský
0.31 ≈ Jan–Křtitel 0.28, ač první je táž osoba a druzí dva lidé).
"""

from jellyai.graph.extract import make_fact, Participant, _alias_identities
from jellyai.graph.graph import FactGraph
from jellyai.graph.instance import resolve_instances


def _fact(pred, *pairs):
    return make_fact(pred, [Participant(r, n, t) for r, n, t in pairs])


def _shard_graph():
    """Ježíš (nositel) + střep „Ježíše Krista" + tvrzení Mt 1,16."""
    g = FactGraph()
    g.add_fact(_fact("jmenovat", ("subj", "Ježíš", "person"),
                     ("pred", "Kristus", "jméno")))
    for obj in ("Petr", "Jan", "Pilát"):
        g.add_fact(_fact("potkat", ("subj", "Ježíš", "person"),
                         ("obj", obj, "person")))
        g.add_fact(_fact("kontext", ("subj", "Ježíše Krista", "person"),
                         ("obj", obj, "person")))
    g.add_fact(_fact("být", ("subj", "Ježíše Krista", "person"),
                     ("pred", "Mesiáš", "concept")))
    return g


def test_asserted_alias_merges_shard():
    """„Ježíše Krista" = ježíš + krist ⊆ jména nositele ∪ tvrzený alias
    Kristus; otisk sdílí 3 sousedy → střep srůstá, kanon = nejkratší."""
    g = _shard_graph()
    assert resolve_instances(g) == 1
    assert "Ježíše Krista" not in g.nodes
    parts = {p.node for f in g.facts_of("Ježíš") for p in f.participants}
    assert "Mesiáš" in parts                 # fakty střepu přešly na kanon
    assert "Ježíše Krista" in g.aliases.get("Ježíš", [])


def test_unasserted_shard_stays_separate():
    """Bez textového tvrzení střep NEsrůstá, i když otisk sdílí svět —
    Jan Křtitel a Jan (evangelista) jsou dvě osoby jednoho příběhu."""
    g = FactGraph()
    for obj in ("Ježíš", "Jordán", "Herodes"):
        g.add_fact(_fact("potkat", ("subj", "Jan", "person"),
                         ("obj", obj, "person")))
        g.add_fact(_fact("potkat", ("subj", "Jan Křtitel", "person"),
                         ("obj", obj, "person")))
    assert resolve_instances(g) == 0
    assert "Jan" in g.nodes and "Jan Křtitel" in g.nodes
    assert "jan" in g.name_families          # rodina pro dialogovou nabídku
    assert set(g.name_families["jan"]) == {"Jan", "Jan Křtitel"}


def test_assertion_without_corroboration_does_not_merge():
    """Tvrzený alias bez opory v grafu (sdílení sousedů < práh) nesrůstá —
    ojedinělá věta nesmí přepsat strukturu."""
    g = FactGraph()
    g.add_fact(_fact("jmenovat", ("subj", "Ježíš", "person"),
                     ("pred", "Kristus", "jméno")))
    g.add_fact(_fact("být", ("subj", "Ježíše Krista", "person"),
                     ("pred", "Mesiáš", "concept")))
    g.add_fact(_fact("být", ("subj", "Ježíš", "person"),
                     ("pred", "tesař", "concept")))
    assert resolve_instances(g) == 0
    assert "Ježíše Krista" in g.nodes


def test_relation_answer_enumerates_shared_facts():
    """Vztahová otázka vyjmenuje fakty sdílené oběma uzly — slovesný děj
    (poslat) je silnější evidence než kontext; elidovaný druhý účastník
    se doplní z konverzačního těžiště (žhavý Ježíš)."""
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.ufal_client import FakeUfalClient
    g = FactGraph()
    g.add_fact(_fact("poslat", ("subj", "Jan Křtitel", "person"),
                     ("theme", "Ježíš", "person"),
                     ("theme", "otázka", "concept")))
    g.add_fact(_fact("kontext", ("subj", "Ježíš", "person"),
                     ("obj", "Jan Křtitel", "person")))
    g.add_fact(_fact("být", ("subj", "Ježíš", "person"),
                     ("pred", "Mesiáš", "concept")))
    a = GraphAnswerer(g, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))
    out = a.answer("Jaký měl Ježíš vztah k Janu Křtiteli?", [])
    assert "poslat" in out.text                  # děj, ne kontext
    a.reset()
    a.answer("Kdo je Ježíš?", [])                # rozsvítí Ježíše
    out = a.answer("Jaký měl vztah k Janu Křtiteli?", [])
    assert "poslat" in out.text                  # pro-drop z těžiště


def test_alias_extraction_reads_recene():
    """Mt 1,16: „narodil se Ježíš řečený Kristus" → jmenovat(Ježíš, Kristus)."""
    sent = [
        {"form": "narodil", "lemma": "narodit", "upos": "VERB"},
        {"form": "se", "lemma": "se", "upos": "PRON"},
        {"form": "Ježíš", "lemma": "Ježíš", "upos": "PROPN"},
        {"form": "řečený", "lemma": "řečený", "upos": "ADJ"},
        {"form": "Kristus", "lemma": "Kristus", "upos": "PROPN"},
    ]
    facts = _alias_identities(sent, [], {})
    assert len(facts) == 1
    parts = {(p.role, p.node) for p in facts[0].participants}
    assert facts[0].predicate == "jmenovat"
    assert ("subj", "Ježíš") in parts and ("pred", "Kristus") in parts


def _glue_graph(with_assertion=False):
    """Slepenec „Áron Mojžíš" sdílí svět (Hospodin, faraón, Izrael)
    s Áronem i Mojžíšem; Mojžíš je těžší komponenta. Kontrola: „Josef
    Čapek" sdílí svět jen s Čapkem — biblický Josef žije jinde."""
    g = FactGraph()
    for who in ("Áron Mojžíš", "Áron", "Mojžíš"):
        for other in ("Hospodin", "faraón", "Izrael"):
            g.add_fact(_fact("potkat", ("subj", who, "person"),
                             ("obj", other, "person")))
    for _ in range(2):   # komponenty spolu JEDNAJÍ (evidence d)
        g.add_fact(_fact("mluvit", ("subj", "Mojžíš", "person"),
                         ("obj", "Áron", "person")))
    for _ in range(3):   # Mojžíš = silnější komponenta
        g.add_fact(_fact("mluvit", ("subj", "Mojžíš", "person"),
                         ("obj", "Hospodin", "person")))
    for who in ("Josef Čapek", "Čapek"):
        for other in ("Praha", "Karel", "obraz"):
            g.add_fact(_fact("kontext", ("subj", who, "person"),
                             ("obj", other, "person")))
    g.add_fact(_fact("žít", ("subj", "Josef", "person"),
                     ("obj", "Egypt", "person")))
    if with_assertion:
        g.add_fact(_fact("jmenovat", ("subj", "Áron", "person"),
                         ("pred", "Mojžíš", "jméno")))
    return g


def test_slepenec_se_rozpusti_do_silnejsi_komponenty():
    """#8 fáze 2 (bod 2): „Áron Mojžíš" kryje dvě nezávislé osoby a
    otiskem sdílí svět s OBĚMA → rozpustí se do silnější (Mojžíš)."""
    from jellyai.graph.instance import dissolve_glued_persons
    g = _glue_graph()
    dissolved = dissolve_glued_persons(g)
    assert dissolved == 1
    assert "Áron Mojžíš" not in g.nodes
    assert any(p.node == "Mojžíš" for f in g.facts.values()
               for p in f.participants)


def test_cele_jmeno_prezije_rozpousteni():
    """„Josef Čapek" kryje uzly Josef i Čapek, ale svět biblického
    Josefa nesdílí (překryv jen k jedné komponentě) → zůstává."""
    from jellyai.graph.instance import dissolve_glued_persons
    g = _glue_graph()
    dissolve_glued_persons(g)
    assert "Josef Čapek" in g.nodes


def test_tvrzeny_alias_slepenec_nerozpousti():
    """Textové tvrzení `jmenovat` mezi komponentami („Šimon zvaný
    Petr") znamená JEDNU osobu — o srůstu rozhoduje resolve_instances,
    rozpouštění se nehlásí (nesmí dvojvládně přepisovat tvrzení)."""
    from jellyai.graph.instance import dissolve_glued_persons
    g = _glue_graph(with_assertion=True)
    assert dissolve_glued_persons(g) == 0
    assert "Áron Mojžíš" in g.nodes
