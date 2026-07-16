"""Export reifikovaného faktového grafu do viewBase.

viewBase je force-graph (Three.js + d3-force-3d) a umí `Canvas.from_networkx(G)`.
Do grafu jdou **entitní uzly i faktové uzly** (faktový uzel má typ `fact`, popisek =
predikát); hrany jsou role-hrany fakt→účastník s vahou faktu. `to_networkx` je
primární most (líný import), `to_json` je bezzávislostní alternativa.
"""


def _fact_id(fact_node):
    """Krátké čitelné id faktového uzlu (predikát + otisk klíče)."""
    return "fact:" + fact_node.predicate + ":" + str(abs(hash(fact_node.id)) % 100000)


def to_json(graph):
    """Serializuje graf do {nodes, edges} (entitní i faktové uzly, role-hrany).

    Args:
        graph (FactGraph): Graf k exportu.

    Returns:
        dict: {"nodes": [{id,type,weight,label}], "edges": [{src,dst,role,weight}]}.
    """
    nodes = [{"id": n.id, "type": n.type, "weight": n.weight, "label": n.id}
             for n in graph.nodes.values()]
    edges = []
    for fact in graph.facts.values():
        fid = _fact_id(fact)
        nodes.append({"id": fid, "type": "fact", "weight": fact.weight,
                      "label": fact.predicate})
        for p in fact.participants:
            edges.append({"src": fid, "dst": p.node, "role": p.role,
                          "weight": fact.weight})
    return {"nodes": nodes, "edges": edges}


def to_networkx(graph):
    """Převede graf na `networkx.DiGraph` (most do viewBase `from_networkx`).

    NetworkX se importuje líně (není závislost jádra). Faktové uzly mají
    `type="fact"` a `label=predikát`; hrany nesou `role` a `weight`.

    Args:
        graph (FactGraph): Graf k exportu.

    Returns:
        networkx.DiGraph: Uzly a role-hrany.
    """
    import networkx as nx
    g = nx.DiGraph()
    for n in graph.nodes.values():
        g.add_node(n.id, type=n.type, weight=n.weight, label=n.id)
    for fact in graph.facts.values():
        fid = _fact_id(fact)
        g.add_node(fid, type="fact", weight=fact.weight, label=fact.predicate)
        for p in fact.participants:
            g.add_edge(fid, p.node, role=p.role, weight=fact.weight)
    return g
