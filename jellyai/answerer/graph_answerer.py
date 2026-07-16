"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

import re

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.question import analyze_question
from jellyai.answerer.template import _to_nominative
from jellyai.graph.activation import ActivationField
from jellyai.graph.extract import _REL_NOUNS

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
        pak by kapitalizovaný uzel byl jinak jménem nedostupný). Dál preferuje uzel
        pokrývající **víc témat** (aby „Božena Němcová" přebila „Němcová"), delší
        (víceslovnou) entitu a nakonec vyšší frekvenci.

        Args:
            topic_terms (list[str]): Obsahová lemmata otázky.

        Returns:
            str | None: Id uzlu tématu, nebo None když nic nesedí.
        """
        terms = [t for t in topic_terms if t]
        low_terms = [t.lower() for t in terms]
        best_id, best_score = None, None
        for node in self.graph.nodes.values():
            low_id = node.id.lower()
            low_words = low_id.split()
            ins_hits = sum(1 for t in low_terms if t == low_id or t in low_words)
            if ins_hits == 0:
                continue
            words = node.id.split()
            exact_hits = sum(1 for t in terms if t == node.id or t in words)
            # přesná shoda velikosti > jen case-insensitive; pak témata, délka, váha
            score = (exact_hits, ins_hits, len(low_words), node.weight)
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
        return best_id

    def _attend(self, qa):
        """Vybere téma a projde graf — 'attention' nad aktivačním polem.

        Explicitní téma z otázky má přednost (dotaz je konkrétně o něm). Bez něj
        (navazující dotaz) se **nezvolí slepě nejteplejší uzel**, ale projdou se
        kandidáti kontextu **od nejteplejšího** a vezme se první, který na otázku
        *umí odpovědět* (má odpovídající fakt) — relevance vážená aktivací.

        Args:
            qa (QuestionAnalysis): Rozbor otázky.

        Returns:
            tuple: (téma | None, hodnota | None, fakt | None).
        """
        explicit = self._resolve_topic(qa.topic_terms)
        if explicit is not None:
            value, fact = self._traverse(qa, explicit)
            return explicit, value, fact
        if qa.topic_terms:
            # uživatel něco pojmenoval, ale v grafu to neznáme → nehádat z kontextu
            return None, None, None
        for candidate in self._context_candidates():
            if not self._gender_ok(qa, candidate):
                continue          # „narodil?" ≠ ženská entita (a naopak)
            value, fact = self._traverse(qa, candidate)
            if value is not None:
                return candidate, value, fact
        return self.context.hottest(), None, None   # nic neodpovědělo (pro log/step)

    def _context_candidates(self):
        """Uzly kontextu seřazené podle jasu (nejteplejší první)."""
        return sorted(self.context.scores, key=self.context.scores.get, reverse=True)

    def _gender_ok(self, qa, node_id):
        """Shoda rodu otázky s (heuristickým) rodem osoby. U ne-osob vždy True."""
        node = self.graph.nodes.get(node_id)
        if qa.gender is None or node is None or node.type != "person":
            return True
        return _name_gender(node_id) == qa.gender

    def _relation_answer(self, qa):
        """„Kdo je/byl bratr/sestra/… X?" → z relačního faktu druhý účastník.

        V otázce je relační jméno (bratr…) + osoba; vztah je symetrický, takže
        vrátí toho druhého z faktu, kde osoba vystupuje (v libovolné roli).

        Args:
            qa (QuestionAnalysis): Rozbor otázky.

        Returns:
            tuple: (osoba | None, druhý účastník | None, fakt | None).
        """
        relations = [t for t in qa.topic_terms if t.lower() in _REL_NOUNS]
        if not relations:
            return None, None, None
        relation = relations[0].lower()
        person_terms = [t for t in qa.topic_terms if t.lower() not in _REL_NOUNS]
        person = self._resolve_topic(person_terms)
        if person is None:
            return None, None, None
        for fact in self.graph.facts_of(person, predicate=relation):
            others = [p.node for p in fact.participants if p.node != person]
            if others:
                return person, others[0], fact
        return None, None, None

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

    def _traverse(self, qa, topic):  # pylint: disable=too-many-return-statements
        """Projde graf podle typu otázky; vrátí (hodnota, fakt) nebo (None, None).

        U „kdy/kde/kolik" **trvá na shodě slovesa** (žádná náhrada nesouvisející
        událostí — na „kdy se narodil" nesmí odpovědět datem svatby). N-arita: „kdy"
        i „kde" čerpají z téhož faktu.

        Args:
            qa (QuestionAnalysis): Rozbor otázky.
            topic (str): Id uzlu tématu.

        Returns:
            tuple: (hodnota | None, fakt | None).
        """
        g, verb = self.graph, qa.verb_lemma
        # drill „v kterém roce se narodil X": událost → datum (uzel) → pod-fakt rok
        date_part = next((t for t in qa.topic_terms if t in _DATE_PARTS), None)
        if date_part:
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            time_value, _ = self._pick(facts, "time")
            if time_value is None:
                return None, None
            return self._pick(g.facts_of(time_value, role="subj", predicate=date_part), "val")
        if qa.qtype in ("Jaký", "Který"):     # vlastnost/stav (přídavné jméno)
            return self._pick(g.facts_of(topic, role="subj", predicate="být"), "attr")
        if qa.is_copula:                        # identita (podstatné jméno)
            return self._pick(g.facts_of(topic, role="subj", predicate="být"), "pred")
        if qa.qtype in ("Kdy", "Kde", "Kolik"):
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            if qa.qtype == "Kdy":
                value, fact = self._pick(facts, "time")
                return (value, fact) if value is not None else self._pick(facts, "num")
            if qa.qtype == "Kde":
                return self._pick(facts, "loc")
            return self._pick(facts, "num")
        if qa.qtype in ("Kdo", "Co"):
            value, fact = self._pick(g.facts_of(topic, role="obj", predicate=verb), "subj")
            if value is not None:
                return value, fact
            return self._pick(g.facts_of(topic, role="subj", predicate=verb), "obj")
        return None, None

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
        # vztahový dotaz („Kdo byl bratr X?") má přednost — jinak by ho přebila spona
        topic, value, fact = self._relation_answer(qa)
        if value is None:
            topic, value, fact = self._attend(qa)
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
