"""NameTag služba — rozpoznávání pojmenovaných entit přes HTTP.

Běží jako samostatný proces (izolace SWIG modulu). Endpoint `POST /entities`
přijme `{text}` a vrátí `{entities: [{text, type, start, end}]}`. Import ÚFAL je
až tady, uvnitř procesu služby.
"""

from _common import serve, parse_args


def make_entities(ner):
    """Vytvoří handler `/entities` navázaný na načtený NameTag model.

    Args:
        ner: Načtený `ufal.nametag.Ner`.

    Returns:
        callable: funkce(payload) → {"entities": [...]}.
    """
    from ufal.nametag import Forms, TokenRanges, NamedEntities

    def entities(payload):
        text = payload["text"]
        forms, ranges, ents = Forms(), TokenRanges(), NamedEntities()
        tokenizer = ner.newTokenizer()
        tokenizer.setText(text)
        out = []
        while tokenizer.nextSentence(forms, ranges):
            ner.recognize(forms, ents)
            for ent in ents:
                first = ent.start
                last = ent.start + ent.length - 1
                start = ranges[first].start
                end = ranges[last].start + ranges[last].length
                out.append({"text": text[start:end], "type": ent.type,
                            "start": start, "end": end})
        return {"entities": out}

    return entities


def main():
    args = parse_args()
    from ufal.nametag import Ner
    ner = Ner.load(args.model)
    if ner is None:
        raise SystemExit(f"Nelze načíst NameTag model: {args.model}")
    serve(args.host, args.port, {"/entities": make_entities(ner)})


if __name__ == "__main__":
    main()
