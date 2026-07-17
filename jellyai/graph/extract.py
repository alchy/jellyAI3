"""Extrakce reifikovaných faktů z větné anotace.

Každá slovesná událost se stane jedním **faktem** (Fact) s predikátem a účastníky
v rolích (podmět/předmět/čas/místo). Fakt je pozdější uzel grafu — reifikace vztahu.
Uzel účastníka je pojmenovaná entita (kanonicky) nebo nominativní lemma tokenu.
"""

from dataclasses import dataclass

from jellyai.answerer.selection import _clean_lemma
from jellyai.graph.canon import name_gender
from jellyai.lang import current

_SUBJ = {"nsubj", "nsubj:pass"}
_OBJ = {"obj"}   # iobj (adresát „řekl UČEDNÍKŮM") je theme, ne předmět
_ATTR = {"obl", "nmod"}
_ENTITY_TYPE = {"p": "person", "g": "geo", "t": "time", "i": "institution"}
_ATTR_ROLE = {"time": "time", "geo": "loc", "number": "num"}   # typ cíle → role
_SKIP_UPOS = {"PRON", "DET"}   # zájmena/určovatele = balast (a záminka pro pro-drop)


@dataclass(frozen=True)
class Participant:
    """Účastník faktu v konkrétní roli.

    Atributy:
        role (str): Role (subj/obj/time/loc/num/pred).
        node (str): Id uzlu účastníka.
        type (str): Typ uzlu (person/geo/time/number/concept/institution).
    """
    role: str
    node: str
    type: str


@dataclass(frozen=True)
class Fact:
    """Reifikovaná událost — pozdější faktový uzel.

    Atributy:
        predicate (str): Predikát (lemma slovesa, nebo „být" u spony).
        participants (tuple): Seřazená n-tice Participant (určuje identitu faktu).
    """
    predicate: str
    participants: tuple


def make_fact(predicate, participants):
    """Sestaví `Fact` s deterministicky seřazenými účastníky (kvůli identitě).

    Args:
        predicate (str): Predikát faktu.
        participants (list[Participant]): Účastníci (libovolné pořadí).

    Returns:
        Fact: Fakt s n-ticí účastníků seřazenou podle (role, node).
    """
    return Fact(predicate, tuple(sorted(participants, key=lambda p: (p.role, p.node))))


def _entity_type(entity):
    """CNEC typ entity (první písmeno) → typ uzlu."""
    return _ENTITY_TYPE.get(entity.get("type", "")[:1].lower(), "concept")


def _node_for(token, entities, canon=None):
    """Vrátí (id, typ) uzlu pro token: entita (kanonicky) nebo nominativní lemma.

    U osobních entit se id sjednotí přes `canon` (fragment „Karel" → „Karel Čapek"),
    aby se fakta téže osoby nerozpadla na víc uzlů.

    Args:
        token (dict): Token s start/end/lemma/upos.
        entities (list[dict]): Entity věty s offsety.
        canon (dict | None): Mapa osobní jméno → kanonický (nejdelší) tvar.

    Returns:
        tuple[str, str] | None: (id, typ), nebo None když token nemá lemma.
    """
    start, end = token.get("start"), token.get("end")
    if start is not None and end is not None:
        best = None
        for e in entities:
            es, ee = e.get("start"), e.get("end")
            if es is not None and es <= start and end <= ee:
                # pylint: disable=unsubscriptable-object
                if best is None or (ee - es) > (best["end"] - best["start"]):
                    best = e            # nejdelší obklopující entita (celé „13. ledna 1890")
        if best is not None:
            text, typ = best["text"], _entity_type(best)
            if typ == "person" and canon:
                text = canon.get(text, text)
            return text, typ
    lemma = _clean_lemma(token.get("lemma", ""))
    if not lemma:
        return None
    return lemma, ("number" if token.get("upos") == "NUM" else "concept")


def _children(sent, head_id):
    """Tokeny věty s `head` == head_id (1-based)."""
    return [t for t in sent if t.get("head") == head_id]


def _default_if_agrees(default_subject, tok, strict=False):
    """Pro-drop dosazení jen při shodě rodu slovesného tvaru se jménem osoby.

    „**Byla** válka" (Fem) při nejteplejším Čapkovi (Masc) je existenciál,
    ne elize — dosazení by vyrobilo šum být(Čapek, válka). Tvary bez rodu
    (prézens „je") dosazení nechávají — až na `strict` režim SPONY: prézentní
    bezpodmětá spona („je vyjadřována obava") je existenciál, rod je nutný.
    """
    if default_subject is None:
        return None
    gender = tok.get("feats", {}).get("Gender")
    if gender is None:
        return None if strict else default_subject
    if gender != name_gender(default_subject[0]):
        return None
    return default_subject


def _conj_group(tok, sent):
    """Souřadná skupina tokenu: [tok] + jeho `conj` děti (UD věší všechny
    konjunkty na první člen). Pro distribuci faktu přes koordinaci."""
    tok_id = sent.index(tok) + 1
    return [tok] + [t for t in sent
                    if t.get("head") == tok_id
                    and str(t.get("deprel", "")).startswith("conj")]


def _pronoun_person(tok, context):
    """Anafora: osobní zájmeno 3. osoby → nejteplejší rodově shodná osoba.

    Zobecněný pro-drop: rod zájmena (feats `Gender`) se páruje s rodem jména
    (`name_gender` — tvar příjmení, jazyková data) přes kandidáty aktivačního
    pole (nejteplejší první). Demonstrativa (`PronType=Dem`, „to"), reflexiva
    (`Reflex=Yes`, „se") a 1./2. osoba osobu nevážou nikdy.

    Args:
        tok (dict | None): Zájmenný token (s feats).
        context (list[tuple] | None): Kandidáti [(id, typ), …] dle jasu.

    Returns:
        tuple | None: (id, typ) navázané osoby, nebo None.
    """
    if tok is None:
        return None
    feats = tok.get("feats", {})
    if feats.get("PronType") != "Prs" or feats.get("Reflex") == "Yes" \
            or feats.get("Person") not in (None, "3"):
        return None
    gender = feats.get("Gender")
    if gender not in ("Masc", "Fem"):
        return None
    for candidate in context or ():
        if candidate[1] == "person" and name_gender(candidate[0]) == gender:
            return candidate
    return None


def _relation_person(children, sent, entities, canon):
    """Osoba v HOLÉM genitivním přívlastku („bratr **Karla Čapka**").

    Osoba s předložkou („drama **o Karlu Čapkovi**") vztah/autorství nenese —
    je to „o kom", ne „čí". Holý genitiv poznáme podle chybějícího `case`
    (ADP) dítěte u nmod tokenu.
    """
    for tok in children:
        if tok.get("deprel", "").startswith("nmod") and tok.get("upos") not in _SKIP_UPOS:
            tok_id = sent.index(tok) + 1
            if any(c.get("head") == tok_id and c.get("upos") == "ADP" for c in sent):
                continue
            node = _node_for(tok, entities, canon)
            if node is not None and node[1] == "person":
                return node
    return None


def _first(tokens, deprels):
    """První token s `deprel` z množiny, nebo None."""
    return next((t for t in tokens if t.get("deprel") in deprels), None)


def _verb_head(index, sent):
    """Vrátí 1-based id nejbližšího slovesa-předka tokenu (i sebe), nebo None.

    Zanořený atribut (datum/číslo pod přívlastkem) se tak přiřadí slovesu, které ho
    skutečně řídí — a v souvětí zůstane u svého slovesa, ne u sousedního.

    Args:
        index (int): 0-based index tokenu ve větě.
        sent (list[dict]): Věta (tokeny s `head`, 1-based; 0 = kořen).

    Returns:
        int | None: 1-based id slovesa-předka, nebo None.
    """
    seen = set()
    cur = index
    while 0 <= cur < len(sent) and cur not in seen:
        seen.add(cur)
        if sent[cur].get("upos") == "VERB":
            return cur + 1
        head = sent[cur].get("head", 0)
        if not head:
            return None
        cur = head - 1
    return None


def _copular_facts(sent, head_id, subj_tok, subj, entities, canon):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Fakty sponové věty „X je Y" — univerzální reifikace genitivu + identita.

    Y s osobním genitivem → `Y(X, osoba)` pro **libovolné** Y (bratr, drama,
    román…): vztah je struktura věty, ne položka slovníku. Žánrové Y
    (`work_nouns` z jazykových dat) přidá autorství `napsat(osoba, X)`. Vždy
    vznikne i identita/vlastnost `být(X, Y)` („Kdo/Jaký je X?"). Kompenzuje
    parser-quirk, kdy kořenem spony je adjektivum („je satirický sci-fi román
    Karla Čapka") a jmenný přísudek visí jako druhý nsubj za sponou — slovosled
    dělí podmět|přísudek.

    Args:
        sent (list[dict]): Věta (tokeny).
        head_id (int): 1-based id hlavy sponové věty.
        subj_tok (dict | None): Token skutečného podmětu (kvůli odlišení).
        subj (tuple | None): (id, typ) podmětu (i pro-drop náhrada).
        entities (list[dict]): Entity věty.
        canon (dict | None): Kanonizace osobních jmen.

    Returns:
        list[Fact]: Fakty věty (reifikovaný vztah, autorství, identita).
    """
    head = sent[head_id - 1]
    children = _children(sent, head_id)
    rel_head, rel_children = head, children
    if head.get("upos") == "ADJ":
        cop_tok = _first(children, {"cop"})
        nominal = next(
            (t for t in children
             if t.get("deprel") in _SUBJ and t.get("upos") == "NOUN"
             and t is not subj_tok and sent.index(t) > sent.index(cop_tok)),
            None)
        if nominal is not None:
            rel_head = nominal
            rel_children = _children(sent, sent.index(nominal) + 1)
    facts = []
    rel = _clean_lemma(rel_head.get("lemma", ""))
    other = _relation_person(rel_children, sent, entities, canon)
    if subj and rel and other is not None and rel_head.get("upos") == "NOUN":
        facts.append(make_fact(rel, [
            Participant("subj", subj[0], subj[1]),
            Participant("obj", other[0], other[1]),
        ]))
        if rel in current()["work_nouns"]:
            facts.append(make_fact("napsat", [
                Participant("subj", other[0], other[1]),
                Participant("obj", subj[0], subj[1]),
            ]))
    if rel_head.get("feats", {}).get("PronType") in ("Int", "Rel") \
            or _clean_lemma(rel_head.get("lemma", "")).lower() \
            in current()["interrogative_pronouns"]:
        return facts                       # tázací/vztažný kořen („jaký") ≠ identita
    rel_pos = sent.index(rel_head) + 1
    if any(c.get("head") == rel_pos
           and str(c.get("deprel", "")).startswith(("acl", "csubj"))
           for c in sent):
        # přísudek s vedlejší větou („je OBAVA, že…") je postoj/existenciál,
        # ne identita podmětu
        return facts
    rel_id = sent.index(rel_head) + 1
    if any(c.get("head") == rel_id and c.get("upos") == "DET"
           and c.get("feats", {}).get("PronType") == "Dem" for c in sent):
        # „byl TOUTO válkou (ovlivněn)" — demonstrativum u přísudkového jména
        # = adjunkt s přivěšenou sponou (parser-quirk), ne identita
        return facts
    pred = _node_for(rel_head, entities, canon)
    if pred and subj:
        role = "attr" if rel_head.get("upos") == "ADJ" else "pred"
        facts.append(make_fact("být", [
            Participant("subj", subj[0], subj[1]),
            Participant(role, pred[0], pred[1]),
        ]))
    return facts


def _subject_group(subj, subj_tok, sent, entities, canon):
    """Souřadní podměty jako uzly („Karel a Josef psali") — bez duplicit;
    pro-drop/anaforický podmět (bez tokenu) zůstává jediný."""
    if subj_tok is None:
        return [subj]
    deduped = []
    for tok in _conj_group(subj_tok, sent):
        node = _node_for(tok, entities, canon)
        if node and node not in deduped:
            deduped.append(node)
    return deduped or [subj]


_INSTANCE_DEPRELS = ("appos", "nmod", "amod", "flat")


def _surface_node(tok, entities, canon):
    """Uzel s preferencí POVRCHU u vlastních jmen: PROPN bez NER entity si
    nechá form („Vějíř"), ne lemma („vějíř") — titul nesmí kolidovat s obecným
    pojmem (lowercase uzel by ukradl rozřešení tématu)."""
    node = _node_for(tok, entities, canon)
    if node and tok.get("upos") == "PROPN" and node[1] in ("concept", "number"):
        return (tok.get("form", node[0]), "dílo")
    return node


def _apposition_identities(sent, entities, canon):
    """Vlastní jméno přivěšené k druhovému substantivu = identita:
    „(s) hrou R.U.R." → být(R.U.R., pred=hra).

    Parser titul věší jako appos, nmod i amod — rozhoduje tvar: PROPN bez
    předložky a mimo genitiv (Case=Gen je přivlastnění — „bratr **Karla
    Čapka**" identitu nezakládá); hlava musí být OBECNÉ substantivum
    (koncept) — person↔person apozice ve výčtech identity nejsou.
    """
    facts = []
    for i, tok in enumerate(sent):
        if tok.get("upos") != "PROPN" \
                or not str(tok.get("deprel", "")).startswith(_INSTANCE_DEPRELS):
            continue
        if any(c.get("head") == i + 1 and c.get("upos") == "ADP" for c in sent):
            continue
        head_id = tok.get("head", 0)
        if not head_id or sent[head_id - 1].get("upos") != "NOUN":
            continue
        case = tok.get("feats", {}).get("Case")
        head_case = sent[head_id - 1].get("feats", {}).get("Case")
        if case == "Gen" and head_case not in (None, "Gen"):
            continue          # přivlastňovací genitiv („bratr Karla"), ne instance
        if case and head_case and case != head_case:
            continue          # pádová neshoda = mis-tagged vztah, ne titul+jméno
        if case == "Gen" and \
                _clean_lemma(sent[head_id - 1].get("lemma", "")) \
                not in current()["relational_nouns"]:
            continue          # Gen-Gen: „bratra Josefa" je instance, „díla
            #                    Josefa" přivlastnění — rozhodne vztahovost
        kind = _node_for(sent[head_id - 1], entities, canon)
        instance = _surface_node(tok, entities, canon)
        if instance and kind and kind[1] == "concept" \
                and kind[0] not in current()["metalanguage_nouns"] \
                and instance[0] != kind[0]:
            # apozice = DRUHOVÉ zařazení (slabší evidence než spona „být")
            facts.append(make_fact("druh", [
                Participant("subj", instance[0], instance[1]),
                Participant("pred", kind[0], kind[1]),
            ]))
    return facts


_DASHES = ("–", "—", "-")


def _dash_identities(sent, entities, canon):
    """Bezslovesná pomlčková definice „(1926) Adam stvořitel – Divadelní hra…"
    → druh(titul, hra). Encyklopedická struktura: vlevo pojmenovaná položka,
    za pomlčkou druhové substantivum. Jen věty bez slovesa i spony."""
    if any(t.get("upos") == "VERB" or t.get("deprel") == "cop" for t in sent):
        return []
    forms = [t.get("form", "") for t in sent]
    dash = next((i for i, f in enumerate(forms) if f in _DASHES), None)
    if dash is None:
        return []
    instance = next((_surface_node(t, entities, canon)
                     for t in sent[:dash] if t.get("upos") == "PROPN"), None)
    kind = next((_node_for(t, entities, canon)
                 for t in sent[dash + 1:] if t.get("upos") == "NOUN"), None)
    if not instance or not kind or kind[1] != "concept" \
            or kind[0] in current()["metalanguage_nouns"] \
            or instance[0] == kind[0]:
        return []
    return [make_fact("druh", [Participant("subj", instance[0], instance[1]),
                               Participant("pred", kind[0], kind[1])])]


def _negated(index, sent):
    """Slovesný tvar je záporný: vlastní Polarity=Neg, záporné aux dítě, nebo
    xcomp pod záporným řídicím slovesem („NEmusel bojovat")."""
    tok = sent[index]
    if tok.get("feats", {}).get("Polarity") == "Neg":
        return True
    tok_id = index + 1
    if any(c.get("head") == tok_id and c.get("upos") == "AUX"
           and c.get("feats", {}).get("Polarity") == "Neg" for c in sent):
        return True
    if str(tok.get("deprel", "")).startswith("xcomp"):
        head_id = tok.get("head", 0)
        if head_id and _negated(head_id - 1, sent):
            return True
    return False


def _clause_content(head_id, sent, limit=8):
    """Obsah řeči/postoje: ccomp/parataxis klauzule jako povrchový text
    (od hlavy klauzule dál, ohraničeno). „Co řekl X?" pak odpovídá obsahem."""
    for i, tok in enumerate(sent):
        if tok.get("head") == head_id \
                and str(tok.get("deprel", "")).startswith(("ccomp", "parataxis")):
            words = [t.get("form", "") for t in sent[i:i + limit]
                     if t.get("upos") != "PUNCT"]
            if words:
                return " ".join(words)
    return None


def _object_groups(obj_tok, sent, entities, canon, context):
    """Skupiny předmětu: za každý souřadný člen `[obj, jeho appos tituly]`.

    Koordinace („romány a dramata") fakty NÁSOBÍ (skupina na člen); apozice
    („hru **R.U.R.**") zůstává v TÉMŽE faktu jako další obj — titul je pak
    dostupnou dírou pro „Jakou hru napsal X?".
    """
    groups = []
    for tok in _conj_group(obj_tok, sent):
        node = (_node_for(tok, entities, canon)
                if tok.get("upos") not in _SKIP_UPOS
                else _pronoun_person(tok, context))    # anafora „ji/mu"
        if not node:
            continue
        group = [node]
        tok_id = sent.index(tok) + 1
        for child in sent:
            if child.get("head") == tok_id \
                    and str(child.get("deprel", "")).startswith("appos"):
                # apozice patří k předmětu jen PŘILEHLÁ („hru R.U.R.");
                # „apozice" přes interpunkci je mis-tag z jiné klauze
                # („…obilí, jedna přijata, druhá ZANECHÁNA") a do faktu nejde
                lo, hi = sorted((sent.index(tok), sent.index(child)))
                if any(t.get("upos") == "PUNCT" for t in sent[lo + 1:hi]):
                    continue
                appos = _surface_node(child, entities, canon)
                if appos and appos not in group:
                    group.append(appos)
        if group not in groups:
            groups.append(group)
    return groups


def _distribute(verb, subj_nodes, obj_groups, attrs):
    """Fakty pro kartézský součin souřadných podmětů × předmětových skupin."""
    out = []
    for snode in subj_nodes:
        for group in (obj_groups or [()]):
            parts = [Participant("subj", snode[0], snode[1])]
            parts += [Participant("obj", n[0], n[1]) for n in group]
            out.append(make_fact(verb, parts + attrs))
    return out


def extract_facts(annotation, default_subject=None, canon=None, context=None):
    # pylint: disable=too-many-locals,too-many-branches
    """Vytáhne z anotace věty seznam reifikovaných faktů.

    Pro každý sloveso-token vznikne jeden n-ární fakt (podmět + předmět +
    atributy). **Elidovaný podmět** (pro-drop: „Narodil se 1890") dostane
    `default_subject` (hlavní entita dokumentu). **Osobní zájmeno** v roli
    podmětu/předmětu se rozváže anaforou (`_pronoun_person`) na rodově shodnou
    osobu z `context`; demonstrativa a nerozvázaná zájmena osobu NEdědí —
    overtní podmět není elize. Fakt bez dalšího účastníka než podmět se zahazuje.

    Args:
        annotation (dict): {"entities": [...], "sentences": [[token,...],...]}.
        default_subject (tuple | None): (id, typ) náhradního podmětu pro pro-drop.
        canon (dict | None): Kanonizace osobních jmen (sjednocení fragmentů).
        context (list[tuple] | None): Kandidáti anafory [(id, typ), …] dle jasu.

    Returns:
        list[Fact]: Nalezené fakty (mohou se opakovat — agreguje graf).
    """
    facts = []
    entities = annotation.get("entities", [])
    for sent in annotation.get("sentences", []):
        facts.extend(_apposition_identities(sent, entities, canon))
        facts.extend(_dash_identities(sent, entities, canon))
        # atributy (čas/místo/číslo/téma) — i zanořené — přiřaď nejbližšímu
        # slovesu-předku
        attrs_by_verb = {}
        for i, tok in enumerate(sent):
            node = _node_for(tok, entities, canon)
            if node is None:
                continue
            role = _ATTR_ROLE.get(node[1])
            if role is None:
                # konceptové příslovečné určení („uvažovat o literatuře")
                # se nezahazuje — role „theme"; jen obl substantiva. Funkční
                # substantiva („v souvislosti s…") jsou spojovací vata, ne
                # obsah — theme nedostanou (šumové uzly v grafu)
                if node[1] == "concept" and tok.get("upos") in ("NOUN", "PROPN") \
                        and str(tok.get("deprel", "")).startswith("obl") \
                        and _clean_lemma(tok.get("lemma", "")).lower() \
                            not in current()["function_nouns"]:
                    role = "theme"
                else:
                    continue
            vh = _verb_head(i, sent)
            if vh is not None:
                attrs_by_verb.setdefault(vh, set()).add(Participant(role, node[0], node[1]))

        for head_id in range(1, len(sent) + 1):
            head = sent[head_id - 1]
            children = _children(sent, head_id)
            subj_tok = _first(children, _SUBJ)
            pronoun_subj = (subj_tok is not None
                            and subj_tok.get("upos") in _SKIP_UPOS)
            subj_node = None
            if subj_tok is not None and not pronoun_subj:
                subj_node = _node_for(subj_tok, entities, canon)
            elif pronoun_subj:
                subj_node = _pronoun_person(subj_tok, context)   # anafora on/ona

            # sponová věta: (podmět)–být–(přísudek). Rozlišíme **identitu**
            # (podstatné jméno → role „pred": „je spisovatelka") od **vlastnosti/
            # stavu** (přídavné jméno → role „attr": „je nemocná") — „Kdo je"
            # čerpá z identity, „Jaký je" z vlastnosti, takže se nepletou.
            cop_tok = _first(children, {"cop"})
            if cop_tok is not None:
                # overtní zájmenný podmět NENÍ pro-drop elize: buď se rozváže
                # anaforou (on/ona), nebo („Je TO lepra") osobu nedědí vůbec;
                # pro-drop navíc vyžaduje rodovou shodu spony („Byla válka")
                subj = subj_node or (None if pronoun_subj else
                                     _default_if_agrees(default_subject,
                                                        cop_tok, strict=True))
                facts.extend(_copular_facts(sent, head_id, subj_tok, subj,
                                            entities, canon))
                continue

            if head.get("upos") != "VERB":
                continue
            verb = _clean_lemma(head.get("lemma", ""))
            if _negated(head_id - 1, sent):
                verb = "ne" + verb        # polarita patří do predikátu
            obj_tok = _first(children, _OBJ)
            obj_groups = (_object_groups(obj_tok, sent, entities, canon, context)
                          if obj_tok is not None else [])
            iobj_tok = _first(children, {"iobj"})
            if iobj_tok is not None and iobj_tok.get("upos") not in _SKIP_UPOS:
                addressee = _node_for(iobj_tok, entities, canon)
                if addressee:
                    attrs_by_verb.setdefault(head_id, set()).add(
                        Participant("theme", addressee[0], addressee[1]))
            if not obj_groups:
                content = _clause_content(head_id, sent)
                if content:
                    # obsah řeči/postoje (ccomp/parataxis) jako předmět —
                    # „Co řekl X?" odpovídá obsahem, ne adresátem
                    obj_groups = [[(content, "výrok")]]
            attrs = sorted(attrs_by_verb.get(head_id, ()),
                           key=lambda p: (p.role, p.node))
            if not obj_groups and not attrs:
                continue
            # nerozvázaný overtní zájmenný podmět („To vedlo…") osobu nedědí;
            # pro-drop jen při rodové shodě slovesa („Narodila" ≠ Čapek)
            subj = subj_node or (None if pronoun_subj else
                                 _default_if_agrees(default_subject, head))
            if subj is None:
                continue
            subj_nodes = _subject_group(
                subj, None if pronoun_subj else subj_tok, sent, entities, canon)
            facts.extend(_distribute(verb, subj_nodes, obj_groups, attrs))
    return facts
