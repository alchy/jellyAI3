"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

import re

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.question import analyze_question
from jellyai.answerer.pattern import question_pattern, SubQuery
from jellyai.answerer.template import _to_nominative
from jellyai.graph.activation import ActivationField
from jellyai.graph.canon import _stem

_DATE_PARTS = {"rok", "měsíc", "den"}   # drill: „v kterém roce/měsíci…"


def _name_gender(name):
    """Heuristický rod českého jména podle posledního slova: „-á/-a"→Fem
    (Němcová, Božena), jinak Masc (Čapek, Karel, Josef). Jen orientační."""
    last = name.split()[-1] if name else ""
    return "Fem" if last.endswith(("á", "a")) else "Masc"


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback.

    Drží **konverzační těžiště** (`ActivationField` nad id uzlů): každý tah rozsvítí
    téma a odpověď, každý tah pohasíná. Když navazující otázka nemá vlastní téma
    („Kdy se narodila?"), vezme se nejteplejší uzel z rozhovoru — dialog tak plyne.
    """

    def __init__(self, graph, client, fallback, *, context_decay=0.55):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (rozbor otázky).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
            context_decay (float): Pohasínání konverzačního těžiště na dotaz
                (viz `ActivationField.decay`; nižší = kratší paměť kontextu).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback
        self.context_decay = context_decay
        self.last_trace = None   # trasa poslední odpovědi (téma → fakt → hodnota)
        self.visited = []        # uzly protnuté (rekurzivním) matchem → rozsvícení
        self.context = ActivationField(decay=context_decay)   # těžiště (id uzlu → jas)
        self.history = []        # trajektorie konverzace (tahy s trasou a těžištěm)

    def reset(self):
        """Začne nový rozhovor — vymaže těžiště i historii."""
        self.context = ActivationField(decay=self.context_decay)
        self.history = []
        self.last_trace = None

    def _resolve_topic(self, topic_terms):
        """Najde uzel tématu otázky — nejlepší shodu s obsahovými lemmaty.

        **Přesná shoda velikosti má přednost**, case-insensitive je fallback:
        UDPipe někdy velikost zachová (PROPN „Babička" → lemma „Babička" — odliší
        knihu od obecné „babička"), jindy lemmatizuje na malá („Vějíř"→„vějíř" —
        pak by kapitalizovaný uzel byl jinak jménem nedostupný). Třetí patro je
        **kmenová shoda** (`canon._stem` — týž mechanismus jako build-side
        `resolve_entities`): skloněný termín („Galéna") tak dosáhne na kanonický
        uzel („Galén"), aniž by kdy přebil přesnější shodu. Dál preferuje uzel
        pokrývající **víc témat** (aby „Božena Němcová" přebila „Němcová"), delší
        (víceslovnou) entitu a nakonec vyšší frekvenci.

        Args:
            topic_terms (list[str]): Obsahová lemmata otázky.

        Returns:
            str | None: Id uzlu tématu, nebo None když nic nesedí.
        """
        terms = [t for t in topic_terms if t]
        low_terms = [t.lower() for t in terms]
        stems = [_stem(t) for t in terms]
        best_id, best_score = None, None
        for node in self.graph.nodes.values():
            low_id = node.id.lower()
            low_words = low_id.split()
            ins_hits = sum(1 for t in low_terms if t == low_id or t in low_words)
            node_stems = {_stem(w) for w in low_words}
            stem_hits = sum(1 for s in stems if s in node_stems)
            if ins_hits == 0 and stem_hits == 0:
                continue
            exact_hits = sum(1 for t in terms
                             if t == node.id or t in node.id.split())
            # přesná > case-insensitive > kmenová; pak témata, délka, váha
            score = (exact_hits, ins_hits, stem_hits, len(low_words), node.weight)
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
        return best_id

    def _context_candidates(self):
        """Uzly kontextu seřazené podle jasu (nejteplejší první)."""
        return sorted(self.context.scores, key=self.context.scores.get, reverse=True)

    def _gender_ok(self, qa, node_id):
        """Shoda rodu otázky s (heuristickým) rodem osoby. U ne-osob vždy True."""
        node = self.graph.nodes.get(node_id)
        if qa.gender is None or node is None or node.type != "person":
            return True
        return _name_gender(node_id) == qa.gender

    def _pattern_answer(self, question, qa):
        """Univerzální match: otázka → neúplný fakt → najdi shodný v grafu → díra.

        Nahrazuje ruční qtype/relační pravidla i attention: predikát + známé role
        musí sedět, vrátí se účastník v roli díry (ranking váhou + aktivací). Bez
        explicitní entity (**navazující dotaz**) se téma vezme z **kontextu** —
        projdou se kandidáti od nejteplejšího (gender-filtr) a vezme první, co odpoví.

        Args:
            question (str): Dotaz uživatele.
            qa (QuestionAnalysis): Rozbor (rod pro shodu kontextového tématu).

        Returns:
            tuple: (téma | None, hodnota | None, fakt | None).
        """
        self.visited = []                    # uzly protnuté (i)rekurzí → rozsvítit
        pat = question_pattern(question, self.client)
        if pat.known:
            known_set = set()
            for _, known in pat.known:
                node = self._solve(known)    # rekurzivně (i vnořené pod-dotazy)
                if node is None:
                    return None, None, None  # pojmenované, ale neznámé → nehádat
                known_set.add(node)
            return self._answer_from(pat, known_set)
        if pat.predicate is None or qa.topic_terms:
            # bez predikátu nelze; pojmenoval-li něco, co pattern nezachytil (RUR),
            # NEhádat z kontextu — kontext je jen pro skutečně navazující dotaz
            return None, None, None
        for candidate in self._context_candidates():   # navazující dotaz → z těžiště
            if not self._gender_ok(qa, candidate):
                continue                     # „narodil?" ≠ ženská entita (a naopak)
            topic, value, fact = self._answer_from(pat, {candidate})
            if value is not None:
                return topic, value, fact
        return None, None, None

    def _answer_from(self, pat, known_set):
        """Z množiny známých uzlů dořeší díru patternu (vč. 2-skokového drillu)."""
        node0 = next(iter(known_set))
        if pat.date_part:
            # 2-SKOK (rekurze): událost → datum (uzel) → jeho pod-fakt rok/měsíc/den
            date_node, _ = self._match(pat.predicate, known_set, "time", "time")
            if date_node is None:
                return None, None, None
            value, fact = self._match(pat.date_part, {date_node}, "val", "number")
            return (date_node, value, fact) if value is not None else (None, None, None)
        value, fact = self._match(pat.predicate, known_set, pat.hole_role, pat.hole_type)
        return (node0, value, fact) if value is not None else (None, None, None)

    def _solve(self, known):
        """Rekurzivně vyřeší `known` na uzel: list = přímá entita; `SubQuery` =
        vnořený pod-dotaz (vyřeš jeho známé rekurzivně, pak `_match`). **Samo-
        rozbalování/zabalování**: hloubka = zanoření otázky, auto-trigger struktura.

        Args:
            known (str | SubQuery): Termín entity nebo vnořený pod-dotaz.

        Returns:
            str | None: Id uzlu, nebo None když nejde vyřešit.
        """
        if not isinstance(known, SubQuery):
            return self._resolve_topic(known.split())
        sub = set()
        for _, inner in known.known:
            node = self._solve(inner)        # rekurze do hloubky
            if node is None:
                return None
            sub.add(node)
        value, _ = self._match(known.predicate, sub, known.hole_role, None)
        return value

    def _match(self, predicate, known_set, hole_role, hole_type):
        """Jeden skok matche: najdi fakt s `predicate` obsahující všechny `known_set`
        a vrať nejlepší **díru** (jiný účastník) — role/typ díry je preference (ne
        filtr), ranking dělá váha faktu + aktivace. Znovupoužitelné pro rekurzi.

        Returns:
            tuple: (hodnota | None, fakt | None).
        """
        if not known_set:
            return None, None
        node0 = next(iter(known_set))
        best = None
        for fact in self.graph.facts_of(node0, predicate=predicate):
            if not known_set <= {p.node for p in fact.participants}:
                continue
            for part in fact.participants:
                if part.node in known_set:
                    continue                 # díra = jiný než známé (i symetrie vztahů)
                score = fact.weight + self.context.scores.get(part.node, 0.0)
                # „kdy" bere čas i rok-jako-číslo; jinak přesná role díry
                if hole_role and (part.role == hole_role
                                  or (hole_type == "time" and part.role in ("time", "num"))):
                    score += 1000
                if hole_type and part.type == hole_type:
                    score += 100
                if best is None or score > best[0]:   # pylint: disable=unsubscriptable-object
                    best = (score, part.node, fact)
        if best is not None:
            # zapamatuj všechny účastníky použitého faktu → rozsvítí se celá cesta
            self.visited.extend(p.node for p in best[2].participants)
            return best[1], best[2]
        return None, None

    def _reverse_lookup(self, question):
        """Reverzní dotaz: z roku v otázce najde událost (datum → podmět faktu).

        Poslední záchrana pro „Co se stalo <datum>?" — `facts_of` indexuje po uzlu,
        takže z časového uzlu s daným rokem vezmeme podmět jeho nejsilnějšího faktu.

        Args:
            question (str): Původní dotaz (hledá se v něm rok).

        Returns:
            tuple: (uzel data | None, podmět | None, fakt | None).
        """
        match = re.search(r"\b(1\d{3}|20\d{2})\b", question)
        if not match:
            return None, None, None
        year = match.group(1)
        for node in self.graph.nodes.values():
            if node.type == "time" and year in node.id:
                value, fact = self._pick(self.graph.facts_of(node.id, role="time"), "subj")
                if value is not None:
                    return node.id, value, fact
        return None, None, None

    def _pick(self, facts, role):
        """Z faktů vrátí (hodnotu cílové role, fakt) z faktu s nejvyšší vahou."""
        best_weight, best_value, best_fact = -1, None, None
        for fact in facts:
            values = self.graph.participants(fact, role)
            if values and fact.weight > best_weight:
                best_weight, best_value, best_fact = fact.weight, values[0], fact
        return best_value, best_fact

    def _candidate_facts_roles(self, qa, topic):
        """Vrátí (fakty, cílové role) pro daný typ otázky (zdroj pro alternativy)."""
        g, verb = self.graph, qa.verb_lemma
        if qa.qtype in ("Jaký", "Který"):
            return g.facts_of(topic, role="subj", predicate="být"), ["attr"]
        if qa.is_copula:
            return g.facts_of(topic, role="subj", predicate="být"), ["pred"]
        if qa.qtype in ("Kdy", "Kde", "Kolik"):
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            roles = {"Kdy": ["time", "num"], "Kde": ["loc"], "Kolik": ["num"]}[qa.qtype]
            return facts, roles
        if qa.qtype in ("Kdo", "Co"):
            facts = g.facts_of(topic, role="obj", predicate=verb)
            return (facts, ["subj"]) if facts else \
                (g.facts_of(topic, role="subj", predicate=verb), ["obj"])
        return [], []

    def _neighbor_context(self, node_id, exclude, limit=10):
        """Širší kontext kolem uzlu: hodnoty faktů, kde uzel vystupuje jako podmět
        (co ho graf spojuje — pro „Co se stalo <datum>?" tak z Boženy vyplyne i
        „Babička" přes napsat). Bere **top-K podle váhy** (ne úzký práh), aby se
        do kontextu dostaly i řidší, ale věcné vazby."""
        weight_by_value = {}
        for fact in self.graph.facts_of(node_id, role="subj"):
            for participant in fact.participants:
                if participant.node != node_id and participant.node not in exclude:
                    if fact.weight > weight_by_value.get(participant.node, 0):
                        weight_by_value[participant.node] = fact.weight
        ranked = sorted(weight_by_value.items(), key=lambda kv: -kv[1])
        return [v for v, _ in ranked[:limit]]

    def _rank_values(self, facts, roles, temperature):
        """Seřadí hodnoty cílových rolí dle váhy faktu; nechá ty nad prahem teploty."""
        weight_by_value = {}
        for fact in facts:
            for role in roles:
                for value in self.graph.participants(fact, role):
                    if fact.weight > weight_by_value.get(value, 0):
                        weight_by_value[value] = fact.weight
        ranked = sorted(weight_by_value.items(), key=lambda kv: -kv[1])
        if not ranked:
            return []
        top = ranked[0][1]
        return [v for v, w in ranked if w >= top * (1.0 - temperature)]

    def answer(self, question, retrieved, *, temperature=0.0):
        """Odpoví 2-skokem grafu; při neúspěchu deleguje na fallback.

        Uloží i `last_trace` (téma → fakt → hodnota). **Teplota** `> 0` navíc vrátí
        v `Answer.alternatives` další kandidáty (nad prahem `top × (1 − temperature)`)
        — vhodné, když z nich kompozitor skládá delší text.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list): Pasáže (jen pro fallback).
            temperature (float): Teplota shody (0 = jen primární, 1 = široce).

        Returns:
            Answer: Odpověď z grafu (zdroj „graf"), nebo výsledek fallbacku.
        """
        self.last_trace = None
        qa = analyze_question(question, self.client)
        # UNIVERZÁLNÍ princip: otázka→neúplný fakt→match→díra (nahrazuje qtype pravidla
        # i attention — kontextově navazující dotaz řeší tatáž cesta)
        topic, value, fact = self._pattern_answer(question, qa)
        reverse = False
        if value is None:
            # poslední záchrana: „Co se stalo <datum>?" — datum → událost
            topic, value, fact = self._reverse_lookup(question)
            reverse = value is not None
        if value is not None:
            if qa.qtype == "Kde":
                value = _to_nominative(value, self.client) or value   # „Slezsku"→„Slezsko"
            alternatives = []
            if temperature:
                # fuzzy: další kontext s menší vahou (krmivo pro kompozitor/NN)
                if reverse:
                    facts, roles = self.graph.facts_of(topic, role="time"), ["subj"]
                else:
                    facts, roles = self._candidate_facts_roles(qa, topic)
                alternatives = [v for v in self._rank_values(facts, roles, temperature)
                                if v != value]
                # + širší kontext: okolí odpovědi v grafu (co ji spojuje)
                for near in self._neighbor_context(value, {value, topic}):
                    if near not in alternatives:
                        alternatives.append(near)
            self.last_trace = {"topic": topic, "predicate": fact.predicate,
                               "fact": fact.id, "answer": value}
            for node in self.visited:        # rozsvítí celou (rekurzivní) cestu, ne jen konce
                self.context.warm(node, 0.7)
            self._remember(qa, topic, value)
            self._log_turn(question, topic, fact.predicate, value)
            return Answer(text=value, sources=["graf"], score=1.0,
                          alternatives=alternatives, trace=self.last_trace)
        # neúspěch NErozmělňuje kontext: attention nesmí vyhasnout jen proto, že
        # jsme odpověď nenašli (jinak by po pár marných dotazech spadla k nule).
        # Pohasíná se jen při úspěchu (v _remember spolu s rozsvícením nového tématu).
        self._log_turn(question, topic, None, None)
        return self.fallback.answer(question, retrieved)

    def _log_turn(self, question, topic, predicate, answer):
        """Zapíše tah do historie (trajektorie těžiště přes graf).

        Args:
            question (str): Otázka tahu.
            topic (str | None): Rozřešené téma (i z těžiště).
            predicate (str | None): Predikát použitého faktu (None u fallbacku).
            answer (str | None): Odpověď z grafu (None u fallbacku).
        """
        self.history.append({"question": question, "topic": topic,
                             "predicate": predicate, "answer": answer,
                             "gravity": self.context.hottest()})

    def _remember(self, qa, topic, value):
        """Zapíše tah do konverzačního těžiště a pohasí ho.

        Odpověď-entita (Kdo/Co) se rozsvítí nejsilněji — bývá tématem navazující
        otázky („…kdo napsal X? Y. …kdy se narodila?"). U hodnotových odpovědí
        (datum/místo) drží těžiště téma. Pak se vše pohasí (× decay).

        Args:
            qa (QuestionAnalysis): Rozbor otázky.
            topic (str): Id uzlu tématu.
            value (str): Vrácená odpověď (id uzlu).
        """
        if qa.qtype in ("Kdo", "Co"):
            self.context.warm(value, 2.0)
            self.context.warm(topic, 1.0)
        else:
            self.context.warm(topic, 2.0)
        self.context.step()
