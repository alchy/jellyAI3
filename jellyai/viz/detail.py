"""Detail uzlu pro vizualizaci — co o uzlu **držíme** v grafu, jako řádky.

Čistě z `FactGraph` (žádný viewBase) → hermeticky testovatelné. Řádky
`(popisek, hodnota)` naplní ve viewBase detailní okno (metadata uzlu), aby klik
na uzel ukázal jeho typ, váhu (opakování) a fakty — výukově cenné. Faktový uzel
má vlastní řádky (predikát + účastníci). Agregace faktů podle predikátu drží
popisky unikátní (klíč metadat) a čitelné.
"""

_MAX_GROUPS = 5      # kolik spojení (predikátů) v detailu ukázat — jinak přeteče
_MAX_PARTNERS = 4    # kolik partnerů na jeden řádek

# typ uzlu → český popisek (výukově srozumitelnější než person/geo/…)
_CZ_TYPE = {
    "person": "osoba",
    "geo": "místo",
    "time": "čas",
    "number": "číslo",
    "concept": "pojem",
    "institution": "instituce",
    "dílo": "dílo (doplněno spreadem)",
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
    groups = {}           # popisek → {partneři, váha (opakování), pořadí}
    for order, fact in enumerate(graph.facts_of(node_id)):
        is_subject = any(p.role == "subj" and p.node == node_id
                         for p in fact.participants)
        label = f"{fact.predicate} {'→' if is_subject else '←'}"
        group = groups.get(label)
        if group is None:
            group = {"partners": [], "weight": 0, "order": order}
            groups[label] = group
        group["weight"] += fact.weight        # četnost spojení = opakování faktů
        for participant in fact.participants:
            if participant.node != node_id and participant.node not in group["partners"]:
                group["partners"].append(participant.node)
    # jen nejfrekventovanější spojení (jinak se detail nevejde do okna)
    ranked = sorted(groups.items(),
                    key=lambda item: (-item[1]["weight"], item[1]["order"]))
    for label, group in ranked[:_MAX_GROUPS]:
        partners = group["partners"]
        value = ", ".join(partners[:_MAX_PARTNERS])
        if len(partners) > _MAX_PARTNERS:
            value += " …"
        rows.append((label, value))
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
