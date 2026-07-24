#!/usr/bin/env python3
"""Build-time corpus přirozených otázek pro podpůrný neural ranker.

Nevytváří fakta ani nemění corpus: každý řádek jen propojí otázku vytvořenou
lokální Ollamou s existujícím registry factem a jeho proveniencí. Split je po
dokumentech, takže model nemůže potkat tentýž dokument v tréninku i testu.

Spuštění (po startu Ollamy):
    python3 build_neural_pairs.py
Kontrola bez volání modelu:
    python3 build_neural_pairs.py --dry-run
"""
import argparse
import hashlib
import json
import os

import ollama_iface as ollama

HERE = os.path.dirname(os.path.abspath(__file__))


def stable_int(*parts):
    return int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def split_for(doc, train_percent, dev_percent):
    bucket = stable_int(doc) % 100
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + dev_percent:
        return "dev"
    return "test"


def load_candidates(registry, max_per_doc):
    """Vybere reprezentativní facts bez sémantických whitelistů.

    Jediné obecné filtry jsou neprázdný text, odpověď a role. Limit per
    dokument chrání proti tomu, aby dlouhé dokumenty určovaly celý korpus;
    stabilní hash zajišťuje opakovatelnost i při resumování.
    """
    by_doc = {}
    with open(registry, encoding="utf-8") as source:
        for line in source:
            fact = json.loads(line)
            text = (fact.get("text") or "").strip()
            if len(text) < 12:
                continue
            for answer in fact.get("answers", []):
                lemma, role = (answer.get("lemma") or "").strip(), answer.get("role")
                if not lemma or not role:
                    continue
                row = {
                    "doc": fact["doc"], "sent": fact["sent"], "predicate": fact.get("predicate"),
                    "text": text, "answer": lemma, "role": role,
                    "fact_vzor": answer.get("fact_vzor"),
                }
                key = stable_int(row["doc"], str(row["sent"]), role, lemma)
                by_doc.setdefault(row["doc"], []).append((key, row))
    selected = []
    for doc, rows in by_doc.items():
        selected.extend(row for _key, row in sorted(rows, key=lambda item: item[0])[:max_per_doc])
    return sorted(selected, key=lambda row: (row["doc"], row["sent"], row["role"], row["answer"]))


def existing_fact_ids(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, encoding="utf-8") as saved:
        for line in saved:
            if line.strip():
                row = json.loads(line)
                keys.add(row.get("fact_id", row["id"]))
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="omezí počet factů; vhodné pro první ověření modelu")
    args = parser.parse_args()

    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as config_file:
        cfg = json.load(config_file)
    ncfg = cfg["neural_training"]
    registry = os.path.join(HERE, cfg["synthesis"]["registry"])
    out = os.path.join(HERE, ncfg["pairs"])
    rows = load_candidates(registry, ncfg["max_examples_per_doc"])
    if args.limit is not None:
        # Malý pilot musí reprezentovat stejné dokumentové splitty jako celý běh.
        rows = sorted(rows, key=lambda row: stable_int(
            row["doc"], str(row["sent"]), row["role"], row["answer"]
        ))[:args.limit]
    splits = {name: sum(split_for(row["doc"], ncfg["train_percent"], ncfg["dev_percent"]) == name
                         for row in rows) for name in ("train", "dev", "test")}
    print(f"facts: {len(rows)} | split: {splits} | výstup: {out}")
    if args.dry_run:
        return

    done = existing_fact_ids(out)
    written = 0
    with open(out, "a", encoding="utf-8") as target:
        for i, row in enumerate(rows, 1):
            example_id = hashlib.sha1(
                f"{row['doc']}:{row['sent']}:{row['role']}:{row['answer']}".encode("utf-8")
            ).hexdigest()[:16]
            if example_id in done:
                continue
            questions = ollama.gen_questions(row["text"], row["answer"], ncfg["questions_per_fact"])
            for question in questions:
                pair_id = hashlib.sha1(f"{example_id}:{question}".encode("utf-8")).hexdigest()[:16]
                target.write(json.dumps({
                    "id": pair_id, "fact_id": example_id,
                    "split": split_for(row["doc"], ncfg["train_percent"], ncfg["dev_percent"]),
                    "question": question, "answer": row["answer"], "role": row["role"],
                    "predicate": row["predicate"], "fact_vzor": row["fact_vzor"],
                    "fact_ref": [row["doc"], row["sent"]], "source_text": row["text"],
                }, ensure_ascii=False) + "\n")
                written += 1
            done.add(example_id)
            if i % 25 == 0:
                target.flush()
                print(f"  {i}/{len(rows)} facts, {written} otázek", flush=True)
    print(f"hotovo: přidáno {written} otázek")


if __name__ == "__main__":
    main()
