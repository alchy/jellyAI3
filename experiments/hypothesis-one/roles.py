#!/usr/bin/env python3
"""hypothesis-two — ROZKLAD FAKTU DO KATALOGU VĚTNÝCH ČLENŮ (schéma odpovědi).

Odpověď není holý fragment — je to fakt rozložený do univerzálního katalogu rolí
(větných členů). Klíče jsou jazykově NEZÁVISLÉ (who/where/action/…), české popisy
i mapování předložek/spojek jsou DATA v lang/cs.json (zákon 3).

Rozklad je PER KLAUZULE (každý přísudek = jedna klauzule). Vedlejší (relativní)
klauzule dědí místo/čas z řídící věty — meziklauzulová inference (#59):
„…do Jeruzaléma, kde se narodil Ježíš" → where druhé klauzule = Jeruzalém.

Vyžaduje UDPipe :8092.
"""
import sys, json, importlib.util, urllib.request

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb
vb.serve = lambda *a, **k: None
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)

LANG = _run.LANG                            # JEDINÝ zdroj jazykových dat i klíčů
CAT = LANG["role_catalog"]
REL_PLACE = set(LANG["relative_place_adverbs"])
REL_TIME = set(LANG["relative_time_adverbs"])
PASS_SUF = tuple(LANG["participle_passive_suffixes"])
CLAUSE_MARK = LANG["clause_markers"]
PRED_ACL = ("acl", "acl:relcl", "advcl")   # vedlejší klauzule závislé na řídící
ARG_DEPREL = ("nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:arg", "obl:agent")

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["sentences"]

# ── pomůcky nad jednou větou (1-based id = index+1, head=0 → kořen) ──────────
def _byid(sent):
    return {i + 1: t for i, t in enumerate(sent)}

def _cop_heads(sent):
    return {t["head"] for t in sent if t["deprel"] == "cop"}

def _is_pred(t, cop_h, tid):
    return t["upos"] == "VERB" or (t["upos"] in ("NOUN", "ADJ", "PROPN") and tid in cop_h)

def _participle_key(t):
    if t.get("feats", {}).get("VerbForm") != "Part":
        return None
    if t["feats"].get("Voice") == "Pass" or t["form"].lower().endswith(PASS_SUF):
        return "passive_participle"
    return "past_participle"        # činné příčestí (-l)

# ── role jednoho tokenu → klíč katalogu (jádro sdílí run.role_key) ───────────
def role_of(t, sent, byid):
    dep, up, f = t["deprel"], t["upos"], t.get("feats", {})
    if dep == "conj" and t["head"] in byid:      # souřadné spojení → role řídícího
        return role_of(byid[t["head"]], sent, byid)
    if dep in ARG_DEPREL:                         # jádro: TÝŽ mapovač jako registr
        prep = _run.prep_of(sent, sent.index(t) + 1)
        return _run.role_key("obl" if dep == "obl:agent" else dep, up, f, prep)
    if dep == "advmod":
        if t["lemma"] in REL_PLACE: return "where"
        if t["lemma"] in REL_TIME:  return "when"
        return "how"
    if dep in ("amod", "nmod", "nmod:poss", "det", "det:poss"):
        return "which_attribute"
    if dep == "xcomp":
        return "as_what_state"
    return None

# ── STANDARDIZACE: každý sektor → NÁŠ čitelný klíč (deprel se ven nedostane) ──
def standard_role(t, sent, byid, cop_heads=None):
    """Standardizovaný klíč sektoru. Nečitelné deprel zkratky (nsubj/obl/amod)
    NAHRAZUJEME katalogovými klíči (who/where/which_attribute) — i pro předložku,
    spojku, interpunkci. Výstup 1. fáze je vždy v našem slovníku, ne v deprel."""
    if cop_heads is None:
        cop_heads = {x["head"] for x in sent if x["deprel"] == "cop"}
    up, dep = t["upos"], t["deprel"]
    tid = sent.index(t) + 1
    if up == "VERB":
        return "action"                        # přísudek slovesný (i konjunkt)
    if tid in cop_heads:
        return "state"                         # jmenný přísudek se sponou
    struct = LANG["deprel_structural"].get(dep)
    if struct:
        return struct                          # preposition/conjunction/punctuation…
    return role_of(t, sent, byid)              # obsahové role (who/where/…)

def standardize(sent):
    """Výstup 1. fáze: [(token, náš klíč)] — plně v katalogu, bez deprel."""
    byid = _byid(sent)
    cop_heads = {x["head"] for x in sent if x["deprel"] == "cop"}
    return [(t, standard_role(t, sent, byid, cop_heads)) for t in sent]

# ── rozklad věty na klauzule + role ─────────────────────────────────────────
def decompose(sent):
    byid, cop_h = _byid(sent), _cop_heads(sent)
    def clause_of(tid):                    # vyšplhej k nejbližšímu přísudku
        seen = set()
        while tid and tid not in seen:
            seen.add(tid)
            if _is_pred(byid[tid], cop_h, tid): return tid
            tid = byid[tid]["head"]
        return None
    clauses = {}
    for i, t in enumerate(sent):
        tid = i + 1
        if _is_pred(t, cop_h, tid):
            c = clauses.setdefault(tid, {"predicate": None, "form": t["form"],
                                         "deprel": t["deprel"], "head": t["head"], "roles": {}})
            if t["upos"] == "VERB":
                c["predicate"] = t["lemma"]; c["roles"]["action"] = [t["lemma"]]
                pk = _participle_key(t)
                if pk: c["roles"].setdefault(pk, []).append(t["form"])
            else:                          # jmenný přísudek se sponou
                c["predicate"] = LANG["copula_lemma"]; c["roles"]["state"] = [t["lemma"]]
    for i, t in enumerate(sent):
        tid = i + 1
        if _is_pred(t, cop_h, tid) or t["upos"] in ("PUNCT", "CCONJ", "SCONJ", "ADP", "AUX", "PART"):
            continue
        key = role_of(t, sent, byid)
        c = clause_of(byid[tid]["head"] if t["deprel"] == "root" else tid)
        if key and c in clauses:
            clauses[c]["roles"].setdefault(key, []).append(t["lemma"])
    # meziklauzulová inference (#59): relativní klauzule dědí místo z antecedentu
    for cid, c in clauses.items():
        if c["deprel"] in PRED_ACL and c["head"] in byid:
            ant = byid[c["head"]]
            where = c["roles"].get("where", [])
            if any(w in REL_PLACE for w in where) and ant["upos"] in ("NOUN", "PROPN"):
                c["roles"]["where"] = [ant["lemma"] if w in REL_PLACE else w for w in where]
                c["roles"]["_where_inherited"] = [ant["lemma"]]
    return list(clauses.values())

def answer(question, clauses):
    """Vyber klauzuli podle přísudku otázky a vrať její strukturní odpověď."""
    qpreds = {t["lemma"] for t in udpipe(question)[0] if t["upos"] == "VERB"}
    for c in clauses:
        if c["predicate"] in qpreds:
            return {k: v for k, v in c["roles"].items() if not k.startswith("_")}
    return clauses[0]["roles"] if clauses else {}

# ── demo na uživatelově ukázce ──────────────────────────────────────────────
def _show(fact, questions):
    print(f"FAKT: {fact}")
    clauses = decompose(udpipe(fact)[0])
    for c in clauses:
        tag = f"  [{c['deprel']}]" if c["deprel"] in PRED_ACL else ""
        print(f"  klauzule «{c['predicate']}»{tag}: "
              f"{json.dumps({k: v for k, v in c['roles'].items() if not k.startswith('_')}, ensure_ascii=False)}")
        if "_where_inherited" in c["roles"]:
            print(f"      ↳ #59 zděděné místo: {c['roles']['_where_inherited'][0]}")
    for q in questions:
        print(f"  OTÁZKA: {q}")
        print(f"     → {json.dumps(answer(q, clauses), ensure_ascii=False)}")
    print()

if __name__ == "__main__":
    _show("Josef a Maria odešli do Jeruzaléma, kde se jim narodil Ježíš.",
          ["Kdo odešel do Jeruzaléma?", "Kdo se narodil v Jeruzalémě?"])
    _show("Karel Čapek byl českým spisovatelem.", ["Kdo byl Karel Čapek?"])
    _show("Bílá nemoc byla napsána Karlem Čapkem v roce 1937.",
          ["Kým byla napsána Bílá nemoc?"])
