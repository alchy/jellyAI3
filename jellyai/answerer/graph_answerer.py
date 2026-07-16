"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.question import analyze_question


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback."""

    def __init__(self, graph, client, fallback):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (rozbor otázky).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback

    def _resolve_topic(self, topic_terms):
        """Najde uzel tématu (shoda lemmatu s id uzlu), s nejvyšší vahou."""
        terms = [t.lower() for t in topic_terms if t]
        best = None
        for node in self.graph.nodes.values():
            nid = node.id.lower()
            if any(term == nid or term in nid.split() for term in terms):
                if best is None or node.weight > best.weight:
                    best = node
        return best.id if best else None

    def _pick(self, facts, role):
        """Z faktů vrátí účastníka cílové role z faktu s nejvyšší vahou."""
        best = None
        for fact in facts:
            values = self.graph.participants(fact, role)
            if not values:
                continue
            if best is None or fact.weight > best[0]:
                best = (fact.weight, values[0])
        return best[1] if best else None

    def answer(self, question, retrieved):
        """Odpoví 2-skokem grafu; při neúspěchu deleguje na fallback.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list): Pasáže (jen pro fallback).

        Returns:
            Answer: Odpověď z grafu (zdroj „graf"), nebo výsledek fallbacku.
        """
        qa = analyze_question(question, self.client)
        topic = self._resolve_topic(qa.topic_terms)
        result = None
        if topic is not None:
            g, verb = self.graph, qa.verb_lemma
            if qa.is_copula or qa.qtype in ("Jaký", "Který"):
                result = self._pick(g.facts_of(topic, role="subj", predicate="být"), "pred")
            elif qa.qtype in ("Kdy", "Kde", "Kolik"):
                facts = (g.facts_of(topic, role="subj", predicate=verb)
                         or g.facts_of(topic, role="subj"))
                if qa.qtype == "Kdy":
                    result = self._pick(facts, "time") or self._pick(facts, "num")
                elif qa.qtype == "Kde":
                    result = self._pick(facts, "loc")
                else:
                    result = self._pick(facts, "num")
            elif qa.qtype in ("Kdo", "Co"):
                result = (self._pick(g.facts_of(topic, role="obj", predicate=verb), "subj")
                          or self._pick(g.facts_of(topic, role="subj", predicate=verb), "obj"))
        if result is None:
            return self.fallback.answer(question, retrieved)
        return Answer(text=result, sources=["graf"], score=1.0)
