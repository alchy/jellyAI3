"""Instanční vrstva — JMÉNO NENÍ ENTITA (backlog #8, fáze 1).

Tentýž člověk vystupuje v korpusu pod více jmény („Ježíš", „Ježíš
Nazaretský", „Ježíše Krista"), a jedno jméno může nést víc lidí (Jan
evangelista × Jan Křtitel). MĚŘENÍ na korpusu ukázalo, že kontextový
otisk (sdílení sousedů grafu) sám o sobě identitu NEROZLIŠÍ: pokrytí
otisku Ježíš–Nazaretský (0.31, táž osoba) je k nerozeznání od
Jan–Křtitel (0.28, dvě osoby) — postavy jednoho příběhu sdílejí svět,
ať jsou totožné, nebo ne. Slepé statistické slučování by lhalo.

Fáze 1 proto srůstá VÝHRADNĚ na základě TEXTOVÉHO TVRZENÍ: extrakce
`jmenovat` čte „X řečený/zvaný Y" (Mt 1,16: „Ježíš řečený Kristus") a
teprve tvrzený alias smí přitáhnout jmenný střep — uzel, jehož všechna
slova jsou kmenově slučitelná se jmény nositele („Ježíše Krista" =
ježíš + krist ⊆ {ježíš} ∪ {kristus}); kontextový otisk pak srůst jen
POTVRZUJE (práh sdílených sousedů), nikdy nezakládá. Vše ostatní
zůstává oddělené a nejednoznačnost řeší DIALOG (jmenné rodiny →
nabídka zaostření; dialog > figly).

Fáze 2 (spec): instance per provenienční odstavec, rozpuštění
dvou-osobových slepenců, jmenovka jako plnohodnotný uzel.
"""

from jellyai.graph.canon import _stem
from jellyai.lang import current

_MIN_CORROBORATION = 3    # sdílených sousedů: tvrzený srůst musí i graf znát


def _stems(node_id):
    """Kmeny slov jména (malými, bez krátkých zbytků)."""
    return {_stem(w.lower()) for w in node_id.split() if len(w) > 2}


def _compatible(stem, pool):
    """Kmen je slučitelný s některým kmenem zásoby (prefixově, min 3)."""
    return any(len(s) >= 3 and len(stem) >= 3
               and (stem.startswith(s) or s.startswith(stem))
               for s in pool)


def _fingerprint(graph, node_id):
    """Kontextový otisk: sousedé uzlu napříč jeho fakty (bez časů/čísel)."""
    neighbors = set()
    for fact in graph.facts_of(node_id):
        for p in fact.participants:
            if p.node != node_id and p.type not in ("time", "number"):
                neighbors.add(p.node)
    return neighbors


def resolve_instances(graph):
    """Srůst jmenných střepů TVRZENÝCH textem + jmenné rodiny (in-place).

    1. Z faktů `jmenovat` (predikát z jazykových dat) se přečtou tvrzené
       aliasy nositelů.
    2. Osobní uzel, jehož všechna slova jsou kmenově slučitelná se jmény
       nositele ∪ jeho aliasy, a jehož otisk sdílí s nositelem dost
       sousedů, je STŘEP téže instance → přemapuje se; kanonické id =
       nejkratší jméno (nominativ nebývá skloněním prodloužený).
    3. `graph.name_families`: kmen křestního slova → uzly, které ho
       nesou — podklad dialogové nabídky („Kdo je Jan?" → rodina).

    Returns:
        int: Počet srostlých střepů.
    """
    name_predicate = current()["name_predicate"]
    persons = {n.id for n in graph.nodes.values() if n.type == "person"}
    aliases = {}                      # nositel → kmeny (vlastní + tvrzené)
    for fact in graph.facts.values():
        if fact.predicate != name_predicate:
            continue
        bearer = next((p.node for p in fact.participants
                       if p.role == "subj"), None)
        alias = next((p.node for p in fact.participants
                      if p.role == "pred"), None)
        if bearer in persons and alias:
            aliases.setdefault(bearer, _stems(bearer)) \
                .update(_stems(alias))
    node_map = {}
    for bearer, pool in sorted(aliases.items()):
        anchor = _fingerprint(graph, bearer)
        for shard in sorted(persons):
            if shard == bearer or shard in node_map:
                continue
            shard_stems = _stems(shard)
            if not shard_stems:
                continue      # bezkmenné id („Le") je slučitelné vakuově —
                #               nesmí pohltit svět (Ježíš→Le)
            if not all(_compatible(s, pool) for s in shard_stems):
                continue
            if len(anchor & _fingerprint(graph, shard)) \
                    < _MIN_CORROBORATION:
                continue          # tvrzení bez opory v grafu nesrůstá
            canonical = min((bearer, shard), key=lambda m: (len(m), m))
            other = shard if canonical == bearer else bearer
            node_map[other] = canonical
    if node_map:
        from jellyai.graph.graph import remap_nodes
        remap_nodes(graph, node_map)
    graph.name_families = _name_families(graph)
    return len(node_map)


def _name_families(graph):
    """Kmen slova jména → osobní uzly, které ho nesou (jen víceuzlové
    rodiny — jednoznačné jméno rodinu netvoří)."""
    families = {}
    for node in graph.nodes.values():
        if node.type != "person":
            continue
        for stem in _stems(node.id):
            families.setdefault(stem, set()).add(node.id)
    return {stem: sorted(members)
            for stem, members in sorted(families.items())
            if len(members) > 1}
