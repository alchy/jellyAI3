"""Detail uzlu pro vizualizaci — co o uzlu **držíme** v grafu, jako řádky.

Čistě z `FactGraph` (žádný viewBase) → hermeticky testovatelné. Řádky
`(popisek, hodnota)` naplní ve viewBase detailní okno (metadata uzlu), aby klik
na uzel ukázal jeho typ, váhu (opakování) a fakty — výukově cenné. Faktový uzel
má vlastní řádky (predikát + účastníci). Agregace faktů podle predikátu drží
popisky unikátní (klíč metadat) a čitelné.
"""

# typ uzlu → český popisek (výukově srozumitelnější než person/geo/…)
_CZ_TYPE = {
    "person": "osoba",
    "geo": "místo",
    "time": "čas",
    "number": "číslo",
    "concept": "pojem",
    "institution": "instituce",
    "fact": "fakt",
}


def node_detail_rows(graph, node_id):
    """Řádky detailu entitního/hodnotového uzlu (typ, váha, jeho fakty).

    Fakty se seskupí podle predikátu a směru (uzel jako podmět „→", jinak „←");
    hodnoty (partneři v ostatních rolích) se spojí do jednoho řádku. Popisky jsou
    tak unikátní a poslouží i jako klíče metadat ve viewBase.

    Args:
        graph (FactGraph): Zdrojový graf.
        node_id (str): Id uzlu.

    Returns:
        list[tuple[str, str]]: Dvojice (popisek, hodnota); pro neznámý uzel
        jediný řádek („uzel", node_id).
    """
    node = graph.nodes.get(node_id)
    if node is None:
        return [("uzel", node_id)]
    rows = [("typ", _CZ_TYPE.get(node.type, node.type)),
            ("váha", str(node.weight))]
    grouped = {}          # popisek → seznam partnerů (bez duplicit, v pořadí)
    order = []
    for fact in graph.facts_of(node_id):
        is_subject = any(p.role == "subj" and p.node == node_id
                         for p in fact.participants)
        label = f"{fact.predicate} {'→' if is_subject else '←'}"
        partners = [p.node for p in fact.participants if p.node != node_id]
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        for partner in partners:
            if partner not in grouped[label]:
                grouped[label].append(partner)
    for label in order:
        rows.append((label, ", ".join(grouped[label])))
    return rows


def fact_detail_rows(fact):
    """Řádky detailu faktového uzlu (predikát, váha opakování, účastníci po rolích).

    Args:
        fact (FactNode): Reifikovaný fakt.

    Returns:
        list[tuple[str, str]]: (popisek, hodnota) — typ/predikát/váha + role→partneři.
    """
    rows = [("typ", "fakt"),
            ("predikát", fact.predicate),
            ("váha", str(fact.weight))]
    by_role = {}          # role → partneři (víc předmětů apod. se spojí)
    order = []
    for participant in fact.participants:
        if participant.role not in by_role:
            by_role[participant.role] = []
            order.append(participant.role)
        by_role[participant.role].append(participant.node)
    for role in order:
        rows.append((role, ", ".join(by_role[role])))
    return rows
