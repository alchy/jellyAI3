"""UDPipe služba — tokenizace, tagování a závislostní rozbor přes HTTP.

Endpoint `POST /parse` přijme `{text}` a vrátí `{sentences: [[token, …], …]}`, kde
token je `{form, lemma, upos, head, deprel, start, end}`. Právě `deprel`/`head`
(podmět `nsubj`, předmět `obj`) umožní vybrat správnou odpovědní entitu — kdo co
dělá, ne jen „nějaká osoba poblíž". UDPipe vydá CoNLL-U, který tu rozparsujeme.
"""

from _common import serve, parse_args


def _parse_conllu(conllu):
    """Rozparsuje CoNLL-U výstup na věty tokenů.

    Args:
        conllu (str): Výstup UDPipe ve formátu CoNLL-U.

    Returns:
        list[list[dict]]: Věty; token = {form, lemma, upos, head, deprel, start, end}.
    """
    sentences = []
    current = []
    for line in conllu.split("\n"):
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10 or "-" in cols[0] or "." in cols[0]:
            continue  # přeskoč víceslovné/prázdné tokeny
        start, end = None, None
        for item in cols[9].split("|"):          # MISC pole
            if item.startswith("TokenRange="):
                span = item[len("TokenRange="):]
                if ":" in span:
                    a, b = span.split(":", 1)
                    if a.isdigit() and b.isdigit():
                        start, end = int(a), int(b)
        current.append({
            "form": cols[1], "lemma": cols[2], "upos": cols[3],
            "head": int(cols[6]) if cols[6].isdigit() else 0,
            "deprel": cols[7], "start": start, "end": end,
        })
    if current:
        sentences.append(current)
    return sentences


def make_parse(pipeline, error_cls):
    """Vytvoří handler `/parse` navázaný na UDPipe pipeline.

    Args:
        pipeline: `ufal.udpipe.Pipeline`.
        error_cls: Třída `ufal.udpipe.ProcessingError`.

    Returns:
        callable: funkce(payload) → {"sentences": [...]}.
    """
    def parse(payload):
        error = error_cls()
        conllu = pipeline.process(payload["text"], error)
        if error.occurred():
            raise RuntimeError(error.message)
        return {"sentences": _parse_conllu(conllu)}

    return parse


def main():
    args = parse_args()
    from ufal.udpipe import Model, Pipeline, ProcessingError
    model = Model.load(args.model)
    if model is None:
        raise SystemExit(f"Nelze načíst UDPipe model: {args.model}")
    pipeline = Pipeline(model, "tokenizer=ranges", Pipeline.DEFAULT,
                        Pipeline.DEFAULT, "conllu")
    serve(args.host, args.port, {"/parse": make_parse(pipeline, ProcessingError)})


if __name__ == "__main__":
    main()
