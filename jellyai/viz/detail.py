"""Detail uzlu pro vizualizaci — co o uzlu **držíme** v grafu, jako řádky.

Čistě z `FactGraph` (žádný viewBase) → hermeticky testovatelné. Řádky
`(popisek, hodnota)` naplní ve viewBase detailní okno (metadata uzlu), aby klik
na uzel ukázal jeho typ, váhu (opakování), **morfologii** (rod z tvaru jména,
kmenový klíč, sloučené pádové tvary z resolveru) a fakty — výukově cenné.
Faktový uzel má vlastní řádky (predikát + účastníci). Agregace faktů podle
predikátu drží popisky unikátní (klíč metadat) a čitelné.
"""

from jellyai.graph.canon import cluster_key, name_gender
from jellyai.lang import current

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
    "výrok": "výrok (obsah řeči)",
}


def _role_label(role):
    """Český popisek role (#9) — tabulka role_detail_labels (jazyk = data)."""
    return current().get("role_detail_labels", {}).get(role, role)


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
            ("váha (výskyty v textu)", str(node.weight)),
            ("aktivace (attention)", "0")]     # živě přepisuje web po dotazu
    if node.type == "person":
        # rod jen u osob (z tvaru jména)
        rows.append(("rod (tvar jména)",
                     "ženský" if name_gender(node_id) == "Fem" else "mužský"))
    if node.type in ("person", "geo"):
        # kmenový klíč clusteru — jména i místa (#9)
        rows.append(("kmen", " ".join(cluster_key(node_id))))
    merged = getattr(graph, "aliases", {}).get(node_id)
    if merged:
        # pádové tvary sloučené resolverem — u KAŽDÉHO typu uzlu (#9)
        rows.append(("sloučené tvary", ", ".join(merged)))
    groups = {}           # popisek → {partneři, váha (opakování), pořadí}
    for order, fact in enumerate(graph.facts_of(node_id)):
        # popisek nese ROLI uzlu ve faktu česky (#9) — místo šipek →/←
        # tak čtenář vidí, ve kterých rolích se uzel účastní
        role = next((p.role for p in fact.participants
                     if p.node == node_id), None)
        label = f"{fact.predicate} ({_role_label(role)})"
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
        czech = _role_label(role)
        label = f"{czech} ({role})" if czech != role else role
        rows.append((label, ", ".join(by_role[role])))
    return rows
