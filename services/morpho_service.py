"""MorphoDiTa služba — morfologická analýza a generování tvarů přes HTTP.

Dvě věci, které V3 potřebuje: `POST /analyze` (`{text}` → tokeny s lemma+tag) a
hlavně `POST /generate` (`{lemma, tag}` → seznam tvarů). Právě generování skloňuje
odpovědní entitu do správného pádu — tím se řeší česká shoda deterministicky.
Model `.tagger` v sobě nese i morfologický slovník (`getMorpho`).
"""

from _common import serve, parse_args


def make_analyze(tagger):
    """Vytvoří handler `/analyze` (tokenizace + tagování).

    Args:
        tagger: Načtený `ufal.morphodita.Tagger`.

    Returns:
        callable: funkce(payload) → {"tokens": [{form, lemma, tag}]}.
    """
    from ufal.morphodita import Forms, TaggedLemmas, TokenRanges

    def analyze(payload):
        text = payload["text"]
        forms, lemmas, ranges = Forms(), TaggedLemmas(), TokenRanges()
        tokenizer = tagger.newTokenizer()
        tokenizer.setText(text)
        out = []
        while tokenizer.nextSentence(forms, ranges):
            tagger.tag(forms, lemmas)
            for i in range(len(lemmas)):
                start = ranges[i].start
                length = ranges[i].length
                out.append({"form": text[start:start + length],
                            "lemma": lemmas[i].lemma, "tag": lemmas[i].tag,
                            "start": start, "end": start + length})
        return {"tokens": out}

    return analyze


def make_generate(morpho):
    """Vytvoří handler `/generate` (skloňování/časování lemmatu dle tagu).

    Args:
        morpho: `ufal.morphodita.Morpho` (z `tagger.getMorpho()`).

    Returns:
        callable: funkce(payload) → {"forms": [str]}.
    """
    from ufal.morphodita import TaggedLemmasForms

    def generate(payload):
        lemma = payload["lemma"]
        tag = payload["tag"]  # může být wildcard (např. „NNFS2?????????")
        result = TaggedLemmasForms()
        morpho.generate(lemma, tag, morpho.GUESSER, result)
        forms = []
        for lemma_forms in result:
            for tagged_form in lemma_forms.forms:
                forms.append(tagged_form.form)
        return {"forms": forms}

    return generate


def main():
    args = parse_args()
    from ufal.morphodita import Tagger
    tagger = Tagger.load(args.model)
    if tagger is None:
        raise SystemExit(f"Nelze načíst MorphoDiTa model: {args.model}")
    morpho = tagger.getMorpho()
    serve(args.host, args.port, {
        "/analyze": make_analyze(tagger),
        "/generate": make_generate(morpho),
    })


if __name__ == "__main__":
    main()
