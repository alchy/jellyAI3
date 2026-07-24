#!/usr/bin/env python3
"""Regrese pro lazy entity routing bez závislosti na UDPipe či produkčních datech."""
import json
import os
import tempfile
import unittest

from fact_store import FactStore
from answering import Answering


class EntityRoutingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        facts = os.path.join(self.tmp.name, "facts")
        os.makedirs(facts)
        with open(os.path.join(self.tmp.name, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"fact_store": {"dir": facts}}, f)
        self.config = f.name
        self._fact("capek", {"predicate": "narodit", "roles": {
            "who": ["Karel", "Čapek"], "where": ["Svatoňovice"]}, "doc": "capek", "sent": 1})
        self._fact("macha", {"predicate": "narodit", "roles": {
            "who": ["Karel", "Hynek", "Mácha"], "where": ["Praha"]}, "doc": "macha", "sent": 2})
        self._fact("other", {"predicate": "zemřít", "roles": {
            "who": ["Čapek"], "when": ["1938"]}, "doc": "other", "sent": 3})

    def tearDown(self):
        self.tmp.cleanup()

    def _fact(self, doc, row):
        with open(os.path.join(self.tmp.name, "facts", f"{doc}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def test_prefers_document_matching_more_entity_terms(self):
        store = FactStore(self.config)
        docs = store.route_docs("narodit", "where", {"Karel", "Čapek"}, max_docs=1)
        self.assertEqual(["capek"], docs)

    def test_respects_predicate_role_and_document_budget(self):
        store = FactStore(self.config)
        self.assertEqual([], store.route_docs("narodit", "when", {"Čapek"}, max_docs=4))
        self.assertEqual(["capek"], store.route_docs(
            "narodit", "where", {"Karel", "Čapek"}, max_docs=1, max_fact_refs=1))


class ComposeMountTest(unittest.TestCase):
    """Kompozice routed × lexikálního mountu (answering._compose_mount) — přes stub self,
    bez těžké inicializace Answering (offline, bez UDPipe/dat)."""

    class _Stub:
        pass

    def _compose(self, mode, routed, lexical):
        stub = self._Stub()
        stub.entity_retrieval = {"mode": mode}
        return Answering._compose_mount(stub, routed, lexical)

    def test_replace_keeps_only_routed(self):
        self.assertEqual(["A", "B"], self._compose("replace", ["A", "B"], ["B", "C", "D"]))

    def test_union_adds_lexical_without_dupes(self):
        self.assertEqual(["A", "B", "C", "D"], self._compose("union", ["A", "B"], ["B", "C", "D"]))

    def test_union_cap_limits_to_max_side(self):
        self.assertEqual(["A", "B", "C"], self._compose("union_cap", ["A", "B"], ["B", "C", "D"]))

    def test_no_routed_falls_back_to_lexical(self):
        self.assertEqual(["B", "C", "D"], self._compose("union", [], ["B", "C", "D"]))


if __name__ == "__main__":
    unittest.main()
