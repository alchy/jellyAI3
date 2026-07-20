"""Reifikovaný faktový graf — entitní uzly + faktové uzly + role-hrany.

Faktový uzel je reifikovaná událost (predikát + váha opakování); k němu vedou
role-hrany na účastníky (entitní/hodnotové uzly). Index `_by_node` umožní z uzlu
najít fakty, v nichž vystupuje (a v jaké roli) — to je základ 2-skokového průchodu.
"""

import os
import pickle
import re
from dataclasses import dataclass, field

from jellyai.graph.extract import (extract_facts, make_fact, Participant,
                                   _SUBJ, _entity_type)
from jellyai.graph.activation import ActivationField
from jellyai.graph.canon import cluster_key

_MONTHS = {
    "ledna": "leden", "února": "únor", "března": "březen", "dubna": "duben",
    "května": "květen", "června": "červen", "července": "červenec", "srpna": "srpen",
    "září": "září", "října": "říjen", "listopadu": "listopad", "prosince": "prosinec",
}


def parse_date(text):
    """Rozloží české datum na složky, které najde (rok/měsíc/den).

    Robustně regexem — datum bývá „13. ledna 1890", „roku 1890" i jen „1890".

    Args:
        text (str): Text časové entity.

    Returns:
        dict: Podmnožina {„rok": str, „měsíc": str (nominativ), „den": str}.
    """
    out = {}
    # numerické datum „21.1.1900" — den.měsíc.rok bez mezer i s mezerami
    numeric = re.search(r"\b([12]?\d|3[01])\.\s?(1[0-2]|0?[1-9])\.\s?(1\d{3}|20\d{2})\b",
                        text)
    if numeric:
        month_names = list(_MONTHS.values())
        return {"den": numeric.group(1),
                "měsíc": month_names[int(numeric.group(2)) - 1],
                "rok": numeric.group(3)}
    year = re.search(r"\b(1\d{3}|20\d{2})\b", text)
    if year:
        out["rok"] = year.group(1)
    day = re.search(r"\b([12]?\d|3[01])\.\s", text)
    if day:
        out["den"] = day.group(1)
    for genitive, nominative in _MONTHS.items():
        if genitive in text:
            out["měsíc"] = nominative
            break
    return out


@dataclass
class Node:
    """Entitní/hodnotový uzel.

    Atributy:
        id (str): Kanonické id (entita nebo nominativní lemma).
        type (str): Typ (person/geo/time/number/concept/institution).
        weight (int): V kolika faktech-výskytech uzel figuruje.
    """
    id: str
    type: str
    weight: int = 0


def instance_lit(predicate, hole_role, roles_of):
    """Verdikt líné instance (rodina × predikát) nad schématem (E3/#51).

    Patří ke grafu (postřeh 4.1) — čistá funkce nad schématem rolí.

    Returns:
        bool | None: True = role díry ve schématu je; False = fakty
        predikátu roli nikdy nenesou (hledání je marné — chytrá
        clarifikace); None = predikát neznám / bez rolí (vakuový
        guard, past 2 — nesoudit).
    """
    roles = roles_of(predicate) if predicate else frozenset()
    if not roles:
        return None
    return hole_role in roles if hole_role else None


@dataclass
class FactNode:
    """Reifikovaný fakt (uzel).

    Atributy:
        id (tuple): Klíč (predicate, participants) — identita faktu.
        predicate (str): Predikát.
        weight (int): Kolikrát se fakt v korpusu opakoval.
        participants (tuple): N-tice Participant.
    """
    id: tuple
    predicate: str
    weight: int
    participants: tuple
    source: set = field(default_factory=set)   # zdrojové dokumenty (provenience)


class FactGraph:
    """Graf entitních a faktových uzlů propojených role-hranami."""

    def __init__(self):
        """Prázdný graf (`nodes`, `facts`, index `_by_node`, `aliases`)."""
        self.nodes = {}
        self.facts = {}
        self._by_node = {}
        self.aliases = {}    # kanonické id → sloučené tvary (plní resolver)
        self.doc_links = {}  # graf DOKUMENTŮ: doc → {doc: síla} (sdílené entity)
        self.name_families = {}  # kmen jména → osobní uzly (instanční vrstva)

    def add_fact(self, fact, source=None):
        """Přidá fakt: sloučí podle identity (`váha++`) nebo založí; udrží indexy.

        Args:
            fact (Fact): Reifikovaný fakt z extrakce.
            source (str | None): Zdrojový dokument (provenience) — pro attention
                nad soubory (sloučený uzel drží fakty z různých korpusů odděleně).
        """
        key = (fact.predicate, fact.participants)
        node = self.facts.get(key)
        if node is None:
            node = FactNode(key, fact.predicate, 0, fact.participants)
            self.facts[key] = node
            for p in fact.participants:
                self._by_node.setdefault(p.node, []).append((key, p.role))
        node.weight += 1
        if source is not None:
            node.source.add(source)
        for p in fact.participants:
            self._touch(p.node, p.type)

    def _touch(self, node_id, node_type):
        """Zajistí uzel a připočte frekvenci."""
        n = self.nodes.get(node_id)
        if n is None:
            self.nodes[node_id] = Node(node_id, node_type, 1)
        else:
            n.weight += 1

    def predicate_roles(self, predicate):
        """Role, ve kterých fakty predikátu nesou účastníky (#57 E3).

        Schéma otázkového grafu se NEkurátoruje — čte se z faktů;
        prázdná množina = graf predikát nezná (žádný verdikt).
        """
        roles = set()
        for fact in self.facts.values():
            if fact.predicate == predicate:
                roles.update(p.role for p in fact.participants)
        return frozenset(roles)

    def facts_of(self, node_id, role=None, predicate=None):
        """Fakty, v nichž uzel vystupuje (volitelně filtr role a predikátu).

        Args:
            node_id (str): Id uzlu.
            role (str | None): Požadovaná role uzlu ve faktu.
            predicate (str | None): Požadovaný predikát faktu.

        Returns:
            list[FactNode]: Odpovídající faktové uzly.
        """
        out = []
        for (key, r) in self._by_node.get(node_id, []):
            if role is not None and r != role:
                continue
            fact = self.facts[key]
            if predicate is not None and fact.predicate != predicate:
                continue
            out.append(fact)
        return out

    @staticmethod
    def participants(fact_node, role):
        """Id účastníků faktu dané role."""
        return [p.node for p in fact_node.participants if p.role == role]

    def replace_facts(self, facts):
        """Vymění množinu faktů a přestaví uzly i index `_by_node`.

        Váha uzlu = Σ vah faktů za každou účast (sémantika `_touch`); typ
        uzlu = typ z prvního faktu v pořadí (deterministické).

        Args:
            facts (dict): (predicate, participants) → FactNode.
        """
        self.facts = facts
        self.nodes = {}
        self._by_node = {}
        for key, fact in facts.items():
            for p in fact.participants:
                node = self.nodes.get(p.node)
                if node is None:
                    self.nodes[p.node] = Node(p.node, p.type, fact.weight)
                else:
                    node.weight += fact.weight
                self._by_node.setdefault(p.node, []).append((key, p.role))

    def save(self, path):
        """Uloží graf (pickle). Vrátí cestu."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "facts": self.facts,
                         "by_node": self._by_node, "aliases": self.aliases,
                         "doc_links": self.doc_links,
                         "name_families": self.name_families}, f)
        return path

    @classmethod
    def load(cls, path):
        """Načte graf z disku."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        g = cls()
        g.nodes = state["nodes"]
        g.facts = state["facts"]
        g._by_node = state["by_node"]
        g.aliases = state.get("aliases", {})
        g.doc_links = state.get("doc_links", {})
        g.name_families = state.get("name_families", {})
        return g


def _canonical_persons(items):
    """Sestaví mapu osobní jméno → nejdelší tvar téhož jména v dokumentu.

    NameTag tvoří překrývající se jména („Karel", „Karel Čapek", „Karel Antonín
    Čapek"). Aby se fakta téže osoby nerozpadla, sjednotíme každý fragment na
    nejdelší jméno, které obsahuje všechna jeho slova.

    Args:
        items (list[tuple[int, dict]]): (index věty, anotace) dokumentu.

    Returns:
        dict: osobní jméno → kanonický (nejdelší) tvar.
    """
    persons = set()
    for _, annotation in items:
        for e in annotation.get("entities", []):
            if e.get("type", "")[:1].lower() == "p":
                persons.add(e["text"])
    # deterministicky (set má náhodné pořadí → jinak nereprodukovatelný graf)
    ordered = sorted(persons)
    canon = {}
    for p in ordered:
        words = set(p.split())
        best = p
        for q in ordered:
            if words <= set(q.split()) and len(q.split()) > len(best.split()):
                best = q
        canon[p] = best
    return canon


def _warm_persons(field, annotation, canon):
    """Rozsvítí osobní entity věty (kanonicky); podmětovou entitu silněji.

    Args:
        field (ActivationField): Aktivační pole dokumentu.
        annotation (dict): Anotace věty (entity + tokeny).
        canon (dict): Kanonizace osobních jmen.
    """
    persons = [e for e in annotation.get("entities", [])
               if e.get("type", "")[:1].lower() == "p"]
    subj_spans = [(t["start"], t["end"])
                  for sent in annotation.get("sentences", []) for t in sent
                  if t.get("deprel") in _SUBJ and t.get("start") is not None]
    for e in persons:
        is_subject = (e.get("start") is not None
                      and any(e["start"] <= s and en <= e["end"] for s, en in subj_spans))
        field.warm((canon.get(e["text"], e["text"]), "person"), 2.0 if is_subject else 1.0)


def _associate_context(graph, annotation, subject, canon, extra=(), source=None):
    """Role ③ aktivačního pole: KONTEXTOVÁ ASOCIACE entit věty se subjektem.

    Dokumentová struktura (bibliografický řádek, výčet) často nemá sloveso,
    ale vazbu nese: entita zmíněná, „když svítí" určitá osoba, k ní patří.
    Každá kontejnerová entita věty se přiváže slabým faktem `kontext` na
    aktuální subjekt dokumentu; opakovaná blízkost váhu akumuluje a resolver
    pádové tvary subjektu sloučí, takže asociace konverguje na jednu osobu.
    Predikát `kontext` je primitiv modelu — dotaz ho čte jako druhé patro,
    když přesný predikát fakt nemá (predikát je preference, ne filtr).

    Args:
        graph (FactGraph): Graf (mění se in-place).
        annotation (dict): Anotace věty (entities).
        subject (tuple | None): (id, typ) aktuálního subjektu dokumentu.
        canon (dict): Kanonizace osobních jmen dokumentu.
        extra (iterable): Další (id, typ) k asociaci — konceptové podměty
            faktů věty („rodina" musí mít vazby, i když ji NER nevidí).
    """
    if subject is None:
        return
    for node in sorted(extra):
        if node[0] != subject[0]:
            graph.add_fact(make_fact("kontext", [
                Participant("subj", subject[0], subject[1]),
                Participant("obj", node[0], node[1])]), source=source)
    entities = annotation.get("entities", [])
    for e in entities:
        typ = _entity_type(e)
        if typ in ("time", "number"):
            continue                     # data/čísla patří slovesným faktům
        es, ee = e.get("start"), e.get("end")
        if any(o is not e and o.get("start") is not None and es is not None
               and o["start"] <= es and ee <= o["end"] for o in entities):
            continue                     # fragment uvnitř delší entity
        text = canon.get(e["text"], e["text"]) if typ == "person" else e["text"]
        text = text.strip("„“”‚’'\"»«›‹")   # NER text s přilepenou uvozovkou
        if not text or text == subject[0]:
            continue
        graph.add_fact(make_fact("kontext", [
            Participant("subj", subject[0], subject[1]),
            Participant("obj", text, typ)]), source=source)


def build_graph(annotations):
    """Postaví faktový graf ze všech větných anotací (s aktivační koreferencí).

    Anotace se zpracují **po dokumentech v pořadí vět**; aktivační pole drží
    „aktuální subjekt" (naposledy zmíněná osoba, s pohasínáním). Věty s elidovaným
    podmětem (pro-drop) se přiřadí nejteplejší osobě — tím se zachytí biografická
    fakta a správně se ošetří i přesun tématu (odstavec o jiné osobě).

    Args:
        annotations (dict): (doc_id, index věty) → anotace (viz `annotate_documents`).

    Returns:
        FactGraph: Naplněný graf.
    """
    by_doc = {}
    for key, annotation in annotations.items():
        doc_id, idx = key if isinstance(key, tuple) else (key, 0)
        by_doc.setdefault(doc_id, []).append((idx, annotation))
    graph = FactGraph()
    for doc_id, items in by_doc.items():
        items.sort(key=lambda t: t[0])
        canon = _canonical_persons(items)
        field = ActivationField()
        for _, annotation in items:
            subject = field.hottest()          # (id, typ) nejteplejší osoby, nebo None
            sentence_facts = extract_facts(annotation, default_subject=subject,
                                           canon=canon, context=field.ranked())
            for fact in sentence_facts:
                graph.add_fact(fact, source=doc_id)
            concept_subjects = {(p.node, p.type) for f in sentence_facts
                                for p in f.participants
                                if p.role == "subj" and p.type == "concept"}
            _associate_context(graph, annotation, subject, canon,
                               extra=concept_subjects, source=doc_id)
            _warm_persons(field, annotation, canon)
            field.step()
    _decompose_dates(graph)
    resolve_entities(graph)
    _build_doc_graph(graph)
    return graph


def _build_doc_graph(graph):
    """Graf dokumentů: hrana mezi dvěma dokumenty = kolik ENTIT sdílejí
    (spolu-zmínění). Pro každou entitu posbírá dokumenty přes všechny její
    fakty a pospojuje jejich dvojice. Týž primitiv jako graf termínů —
    aktivace pak vyzařuje i po dokumentech (attention nad soubory se škáluje
    s jejich počtem)."""
    from itertools import combinations
    links = {}
    for node_id, entries in graph._by_node.items():   # pylint: disable=protected-access
        node = graph.nodes.get(node_id)
        if node is None or node.type not in ("person", "geo", "dílo"):
            continue
        docs = set()
        for key, _ in entries:
            docs |= graph.facts[key].source
        for a, b in combinations(sorted(docs), 2):
            links.setdefault(a, {})[b] = links.setdefault(a, {}).get(b, 0) + 1
            links.setdefault(b, {})[a] = links.setdefault(b, {}).get(a, 0) + 1
    graph.doc_links = {d: dict(sorted(n.items())) for d, n in sorted(links.items())}


def _decompose_dates(graph):
    """Zanoří časové uzly: datum se stane uzlem s vlastními pod-fakty rok/měsíc/den.

    „13. ledna 1890" pak není jen řetězcová hodnota, ale uzel grafu, z něhož se dá
    dojít na rok (1890) — umožní dotaz „v kterém roce". Reifikace o patro níž.

    Args:
        graph (FactGraph): Graf s časovými uzly (upraví se in-place).
    """
    for node in list(graph.nodes.values()):
        if node.type != "time":
            continue
        for part, value in parse_date(node.id).items():
            vtype = "number" if part in ("rok", "den") else "concept"
            graph.add_fact(make_fact(part, [Participant("subj", node.id, "time"),
                                            Participant("val", value, vtype)]))


def _positional_merge(canon_by_key):
    """Druhý pass resolveru: kratší víceslovné jméno → jednoznačné delší.

    Sloučí se jen při shodě kmenů na **první (křestní) a poslední (příjmení)**
    pozici: „Karel Čapek" → „Karel Antonín Čapek". Otec „Antonín Čapek" má
    křestní kmen na prostřední pozici syna → zůstává. Dvojznačný cíl (víc
    delších kandidátů) nebo jednoslovné jméno → žádné slučování.

    Args:
        canon_by_key (dict): Kmenový klíč clusteru → kanonické id.

    Returns:
        dict: Kanonické id → kanonické id delšího jména.
    """
    mapping = {}
    keys = sorted(canon_by_key)
    for key in keys:
        if len(key) < 2:
            continue
        targets = [k for k in keys
                   if len(k) > len(key) and k[0] == key[0] and k[-1] == key[-1]]
        if len(targets) == 1:
            mapping[canon_by_key[key]] = canon_by_key[targets[0]]
    for src in list(mapping):
        # zřetězení (2-slovné → 3-slovné → 4-slovné); cíl má vždy víc slov,
        # cyklus nehrozí
        dst = mapping[src]
        while dst in mapping:
            dst = mapping[dst]
        mapping[src] = dst
    return mapping


def resolve_entities(graph):
    """Post-build entity resolution: sloučí varianty osoby do jednoho uzlu.

    Dva konzervativní passy: (1) **pádové varianty** — shluk kmenovým klíčem
    (`cluster_key`), kanonické id = lexikograficky nejmenší člen (pádové
    koncovky nominativ prodlužují, takže minimum bývá nominativ; subj-count
    jako proxy nefunguje — pro-drop koreference rozdává podměty i genitivům);
    (2) **poziční fragmenty** — kratší víceslovné jméno do jednoznačného
    delšího se shodou křestního a příjmení (`_positional_merge`). Fakty se
    přepíšou s přemapovanými účastníky (kolize identity → součet vah), uzly a
    index `_by_node` se přestaví. Jen person↔person; holá jednoslovná jména se
    neslučují. Idempotentní a deterministické — smí běžet i opakovaně (po
    `recover_entities`).

    Args:
        graph (FactGraph): Graf k přepsání (in-place).

    Returns:
        FactGraph: Týž graf (pro řetězení).
    """
    persons = sorted(n.id for n in graph.nodes.values() if n.type == "person")
    clusters = {}
    for name in persons:
        clusters.setdefault(cluster_key(name), []).append(name)
    # NER nekonzistence: KAPITALIZOVANÝ koncept se shodným klíčem („Ježíš"
    # koncept vedle „Ježíše" person — tagger jednou osobu minul) patří do
    # person clusteru; malé písmeno („bůh" vedle „Bůh") zůstává pojmem
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        if node.type != "concept" or not node.id[:1].isupper():
            continue
        key = cluster_key(node.id)
        if key in clusters:
            clusters[key].append(node.id)
    # kanon = NEJKRATŠÍ člen (nominativ nebývá skloněním prodloužený), pak
    # lex — samotné lex-min přes diakritiku klame („Nazaretského" < „ý")
    canon_by_key = {key: min(members, key=lambda m: (len(m), m))
                    for key, members in clusters.items()}
    positional = _positional_merge(canon_by_key)
    node_map = {}
    for key, members in clusters.items():
        canonical = canon_by_key[key]
        final = positional.get(canonical, canonical)
        for name in members:
            if name != final:
                node_map[name] = final
    if not node_map:
        return graph
    remap_nodes(graph, node_map)
    return graph


def remap_nodes(graph, node_map, force_person=True):
    """Přemapuje uzly faktů podle `node_map` (in-place, sdílená mechanika
    kanonizace, instanční vrstvy i nominativizace). S `force_person` se
    členům mapy VYNUTÍ typ person (koncept „Ježíš" nesmí po sloučení
    vnutit typ concept); nominativizace typy NEmění (geo zůstává geo).
    Kolize identity faktů → součet vah; aliasy se zaznamenají.
    """
    person_ids = (set(node_map) | set(node_map.values())) if force_person \
        else set()
    remapped = {}
    for fact in graph.facts.values():
        moved = make_fact(fact.predicate,
                          [Participant(p.role, node_map.get(p.node, p.node),
                                       "person" if node_map.get(p.node, p.node)
                                       in person_ids and p.type != "jméno"
                                       else p.type)
                           for p in fact.participants])
        if len({p.node for p in moved.participants}) < 2:
            continue    # účastníci splynuli v jeden uzel — fakt nic nenese
        key = (moved.predicate, moved.participants)
        existing = remapped.get(key)
        if existing is None:
            remapped[key] = FactNode(key, moved.predicate, fact.weight,
                                     moved.participants, set(fact.source))
        else:
            existing.weight += fact.weight
            existing.source |= fact.source
    _record_aliases(graph, node_map)
    graph.replace_facts(remapped)


def _record_aliases(graph, node_map):
    """Zapamatuje sloučené tvary (kanonické id → aliasy) pro detail uzlu ve
    vizualizaci a vysvětlitelnost kanonizace; opakovaný běh resolveru dřívější
    aliasy přemapuje (idempotence)."""
    aliases = {}
    for old, new in node_map.items():
        aliases.setdefault(new, set()).add(old)
    for canonical, olds in graph.aliases.items():
        target = node_map.get(canonical, canonical)
        aliases.setdefault(target, set()).update(olds)
    graph.aliases = {key: sorted(values)
                     for key, values in sorted(aliases.items())}
