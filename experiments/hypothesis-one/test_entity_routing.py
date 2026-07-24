#!/usr/bin/env python3
"""Regrese pro lazy entity routing bez závislosti na UDPipe či produkčních datech."""
import json
import os
import tempfile
import unittest

from fact_store import FactStore


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


if __name__ == "__main__":
    unittest.main()
