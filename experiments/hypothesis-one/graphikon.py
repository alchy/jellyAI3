#!/usr/bin/env python3
"""hypothesis-two — JEDEN reálný záznam grafikonu end-to-end (ilustrace).

Vezme SOURCE FACT a složí uzel grafikonu:
  roles.decompose  → DÍRY (OTÁZKY, skelet)
  Ollama           → DOTAZY (povrchy, offline)
  frame_sig (r=X)  → VZOR každého dotazu = QUERY PATTERN (match-klíč)
  projekce díra→výplň → ANSWER PATTERN (fragment + VZOR + větná šablona)
Nakonec runtime demo: ŽIVÁ otázka → VZOR → trefí QUERY PATTERN → ODPOVĚĎ.

Vyžaduje UDPipe :8092 + Ollama :11434.
"""
import sys, json
import roles as R
run = R._run
import ollama_iface as OL

R_SET = (1, 2)                     # VZOR počítáme při r=1 i r=2 (ukázat slití DOTAZŮ)
HOLE_ROLES = ("who", "what_subject", "whom_what", "to_whom", "with_whom_what",
              "where", "when", "state", "whose_of_what")
JUNK = {"on", "ten", "tento", "který", "jenž", "svůj", "sám", "žádný", "se", "co"}
ASK_LEMMAS = {"kdo", "co", "kde", "kam", "kdy", "kým", "komu", "koho", "čí",
              "jaký", "který", "jak", "čeho", "čemu"}

def q_frames(dotaz):
    """VZOR(y) dotazu na pozici tázacího slova (nositel díry), při r∈R_SET."""
    toks = R.udpipe(dotaz)[0]
    mod = run.sentence_modality(toks)
    for i, t in enumerate(toks):
        pt = t.get("feats", {}).get("PronType", "")
        if "Int" in pt or t["lemma"].lower() in ASK_LEMMAS:
            return {f"r{r}": run.frame_sig(toks, i, mod, r) for r in R_SET}
    return {}

def build(fact, nvar=2):
    toks = R.udpipe(fact)[0]
    mod = run.sentence_modality(toks)
    clauses = R.decompose(toks)
    holes = {}
    for c in clauses:
        for role, vals in c["roles"].items():
            if role in HOLE_ROLES:
                vv = [v for v in vals if len(v) > 1 and v.lower() not in JUNK]
                if vv:
                    holes.setdefault(role, vv)
    rec = {"source_fact": fact, "clauses": [c["predicate"] for c in clauses], "queries": []}
    for role, answer in list(holes.items())[:3]:            # 3 díry stačí na ilustraci
        try:
            dotazy = OL.gen_questions(fact, ", ".join(answer), nvar)
            sentence = OL.gen_answer(dotazy[0], fact) if dotazy else ""
        except Exception as e:                              # noqa
            dotazy, sentence = [], f"(ollama chyba: {e})"
        qps = [{"dotaz": d, "vzor": q_frames(d)} for d in dotazy]
        ans_vzor = {}
        for i, t in enumerate(toks):
            if run.canon_lemma(t) in answer:
                ans_vzor = {f"r{r}": run.frame_sig(toks, i, mod, r) for r in R_SET}
                break
        rec["queries"].append({
            "otazka_hole": role, "tazaci": run.LANG["role_ask"].get(role, "?"),
            "query_patterns": qps,
            "answer_pattern": {"fill_role": role, "answer_fragment": answer,
                               "answer_vzor": ans_vzor, "answer_sentence": sentence}})
    return rec

def match(live_q, rec, r="r1"):
    """Runtime: živá otázka → VZOR → trefí uložený QUERY PATTERN → ODPOVĚĎ."""
    vz = q_frames(live_q).get(r)
    for q in rec["queries"]:
        for qp in q["query_patterns"]:
            if qp["vzor"].get(r) == vz:
                return q["otazka_hole"], q["answer_pattern"]["answer_fragment"], vz
    return None, None, vz

def show(rec):
    print("┌─ SOURCE FACT:", rec["source_fact"])
    print("│  klauzule:", rec["clauses"])
    for q in rec["queries"]:
        ap = q["answer_pattern"]
        print(f"├─ OTÁZKA (díra={q['otazka_hole']}, tázací={q['tazaci']})")
        for qp in q["query_patterns"]:
            print(f"│    DOTAZ:  {qp['dotaz']}")
            print(f"│      VZOR (QUERY PATTERN):  {qp['vzor']}")
        print(f"│    → ANSWER PATTERN:  výplň={ap['answer_fragment']}")
        print(f"│        VZOR odpovědi: {ap['answer_vzor']}")
        print(f"│        šablona (věta): {ap['answer_sentence']}")
    print("└─")

if __name__ == "__main__":
    fact = sys.argv[1] if len(sys.argv) > 1 else \
        "Ježíš svolal svých Dvanáct a dal jim sílu a moc vyhánět démony."
    rec = build(fact)
    json.dump(rec, open("graphikon_sample.json", "w"), ensure_ascii=False, indent=2)
    show(rec)
    print("\n=== RUNTIME: živá otázka → VZOR → trefí QUERY PATTERN → ODPOVĚĎ ===")
    for live in ("Kdo svolal svých dvanáct?", "Komu dal Ježíš sílu?", "Co dal Ježíš dvanácti?"):
        hole, ans, vz = match(live, rec, "r1")
        verdict = f"→ díra {hole}, odpověď {ans}" if hole else "→ netrefeno (jiný VZOR)"
        print(f"  „{live}\"  [VZOR r1 {vz}]  {verdict}")
    print("\n→ graphikon_sample.json")
