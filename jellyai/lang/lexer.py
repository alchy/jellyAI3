"""Lexer — JEDEN určovač druhů slov (spec 2026-07-19-vzorova-gramatika, fáze 1).

Sjednocuje kontroly dosud roztroušené po zápisové (Mnemos) a dotazové cestě:
token nese MNOŽINU hypotéz tříd — „byt" je spona i substantivum, „Můj"
přivlastňovací i kapitalizované jméno, „Marcela" jméno i l-lookalike.
Lexer dvojznačnost NEROZHODUJE, jen ji poctivě přizná; rozhodne až stavba
věty (sekvenční vzory, fáze 2–3).

Třídy se počítají VÝHRADNĚ z jazykových tabulek `cs.json` (mechanismus zde,
rozhodnutí v datech):

    otaz            question_words (tázací slovo)
    spona           copula_forms
    prvni_osoba     first_person
    potvrzeni       confirmation_words
    funkcni         query_skip_words (předložky, ukazovací…)
    castice         particle_words (už, však, občas…)
    privlastnovaci  possessive_words
    cas             temporal: day_words ∪ units ∪ now_words ∪ last/next/current
    email           hodnota mimo přirozený jazyk (regexp, obchází tagger)
    jmeno           kapitalizovaný alfanumerický tvar
    l_tvar          l-ové příčestí po ořezu koncovky (kmen v `l_stem`)
    sloveso_fin     kandidát prézentního slovesa: délka ≥ 3, není
                    prvni_osoba/spona/funkcni, koncovka z tabulky NEBO tvar
                    v katalogu finite_verb_forms, a není uzel grafu (veto
                    `is_node` — entity-first)
"""

import re

from dataclasses import dataclass

from jellyai.graph.canon import deaccent
from jellyai.lang import current

# Hodnoty MIMO přirozený jazyk (e-mail) tokenizér rozbije (@, tečky) a tagger
# neklasifikuje. Regexp je proto rozpozná JAKO CELEK.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\w")


@dataclass(frozen=True)
class TaggedToken:
    """Slovo s množinou hypotéz tříd (viz docstring modulu).

    Atributy:
        form (str): Povrchový tvar.
        norm (str): Deakcentovaný lowercase (klíč do tabulek).
        classes (frozenset[str]): Hypotézy druhů slova.
        l_stem (str | None): Kmen l-ového příčestí („měla" → „měl");
            None, když tvar l-příčestím být nemůže.
    """
    form: str
    norm: str
    classes: frozenset
    l_stem: str = None


def l_stem(token):
    """Kmen l-ového příčestí („měl"/„měla"/„měli" → „měl"); jinak None."""
    stripped = token.lower().rstrip("aioy")
    return stripped if stripped.endswith("l") and len(stripped) >= 3 else None


def tokenize(text):
    """Rozdělí text na tokeny: e-mail jako CELEK, koncová větná tečka pryč,
    tečkované zkratky (R.U.R.) zůstávají."""
    raw = re.findall(r"[\w.+-]+@[\w.-]+\.\w+|[\w.]+", text)
    tokens = [t if EMAIL_RE.fullmatch(t)
              else (t.rstrip(".") if "." not in t[:-1] else t) for t in raw]
    return [t for t in tokens if t]


def _temporal_words(lang):
    temporal = lang.get("temporal", {})
    return (set(temporal.get("day_words", ()))
            | set(temporal.get("units", ()))
            | set(temporal.get("now_words", ()))
            | set(temporal.get("last_words", ()))
            | set(temporal.get("next_words", ()))
            | set(temporal.get("current_words", ())))


def classify(text, is_node=None):
    """Text → typované tokeny s hypotézami tříd.

    Args:
        text (str): Věta/výrok/otázka.
        is_node (callable | None): Veto entity-first pro sloveso_fin —
            doslovné slovo uzlu grafu slovesem není („nádraží").

    Returns:
        list[TaggedToken]: Tokeny v pořadí textu.
    """
    lang = current()
    endings = tuple(lang.get("present_verb_endings", ()))
    finite_catalog = set(lang.get("finite_verb_forms", ()))
    temporal = _temporal_words(lang)
    tables = (("otaz", set(lang.get("question_words", ()))),
              ("spona", set(lang.get("copula_forms", ()))),
              ("prvni_osoba", set(lang.get("first_person", ()))),
              ("potvrzeni", set(lang.get("confirmation_words", ()))),
              ("funkcni", set(lang.get("query_skip_words", ()))),
              ("castice", set(lang.get("particle_words", ()))),
              ("privlastnovaci", set(lang.get("possessive_words", ()))),
              ("cast_data", set(lang.get("date_part_forms", ()))),
              ("vztah_dotazu", set(lang.get("relation_query_nouns", ()))),
              ("vztazne_jmeno", set(lang.get("relational_nouns", ()))),
              ("trida_deju", set(lang.get("predicate_class_forms", ()))),
              ("cas", temporal))
    tagged = []
    for form in tokenize(text):
        norm = deaccent(form.lower())
        classes = {name for name, table in tables if norm in table}
        if EMAIL_RE.fullmatch(form):
            classes.add("email")
        if form[:1].isupper() and form.replace("-", "").isalpha():
            classes.add("jmeno")
        stem = l_stem(form)
        if stem is not None:
            classes.add("l_tvar")
        if len(form) >= 5 and form.lower().endswith(
                tuple(lang.get("dative_endings", ()))):
            # HYPOTÉZA dativu (#55: adresát — „Ježíšovi", „učedníkům");
            # lexer nerozhoduje, roli přiřadí až vzor na kartě
            classes.add("dativ")
        if len(form) >= 4 and form.lower().endswith(
                tuple(lang.get("participle_endings", ()))):
            # HYPOTÉZA pasivního participia („pokřtěn", „vydána" — P1b);
            # kolize s podstatnými jmény („žena") nevadí: karta se nehlásí,
            # dokud tvar nerozřeší na predikát grafu (past 11)
            classes.add("participium")
        if (len(form) >= 3
                and not ({"prvni_osoba", "spona", "funkcni"} & classes)
                and (norm in finite_catalog
                     or form.lower().endswith(endings))
                and not (is_node is not None and is_node(form))):
            classes.add("sloveso_fin")
        tagged.append(TaggedToken(form=form, norm=norm,
                                  classes=frozenset(classes), l_stem=stem))
    return tagged
