"""Generativní answerer — odpověď složí natrénovaný model (V2b).

Zapadá do stejného rozhraní jako ExtractiveAnswerer, takže je s ním plně
zaměnitelný (pipeline vybírá podle `config.answerer.mode`). Rozdíl je ve filozofii:
extraktivní vrátí větu doslova z textu, generativní ji **složí** — plynuleji, ale
s rizikem, že si model u malého modelu a mála dat trochu „přibarví". Zdroj (pasáž,
z níž bral kontext) uvádí i tak, ať je odpověď dohledatelná.

Model se načítá líně (až při první odpovědi), aby import modulu nevyžadoval
existující checkpoint — hermetické testy zbytku systému tím zůstávají nezávislé.
"""

from jellyai.answerer.base import Answer, Answerer

_NO_ANSWER = "V textu jsem odpověď nenašel."


class GenerativeAnswerer(Answerer):
    """Answerer, který odpověď generuje malým transformerem z nalezené pasáže."""

    def __init__(self, generator_config):
        """Vytvoří answerer; samotný model se načte až při první odpovědi.

        Args:
            generator_config (GeneratorConfig): Konfigurace generátoru
                (cesty k tokenizeru a checkpointu, parametry samplingu).
        """
        self.config = generator_config
        self._runtime = None  # (model, tokenizer, device) — líně

    def _ensure_loaded(self):
        """Načte model+tokenizer při prvním použití a nakešuje je."""
        if self._runtime is None:
            from model.generate import load_generator
            self._runtime = load_generator(self.config)
        return self._runtime

    def answer(self, question, retrieved):
        """Vygeneruje odpověď z top nalezené pasáže.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list[tuple[Passage, float]]): Pasáže a skóre z retrieveru.

        Returns:
            Answer: Vygenerovaná odpověď + zdroj (doc_id#index); při prázdném
                retrievalu poctivé „nenašel jsem".
        """
        if not retrieved:
            return Answer(text=_NO_ANSWER, sources=[], score=0.0)

        from model.generate import generate_answer
        model, tokenizer, device = self._ensure_loaded()
        passage, score = retrieved[0]  # top pasáž = kontext pro generování
        text = generate_answer(model, tokenizer, passage.text, question,
                               self.config, device)
        source = f"{passage.doc_id}#{passage.index}"
        if not text.strip():
            return Answer(text="(model nevygeneroval odpověď)", sources=[source],
                          score=float(score))
        return Answer(text=text, sources=[source], score=float(score))


def explain():
    """Vrátí lidský popis bloku GenerativeAnswerer pro výukovou vrstvu.

    Returns:
        str: Popis toho, jak generativní answerer skládá odpověď.
    """
    return (
        "GenerativeAnswerer vezme top pasáž od Retrieveru jako kontext, sestaví "
        "prompt 'Kontext: … Otázka: … Odpověď:' a nechá malý transformer dopsat "
        "odpověď (sampling). Plynulejší než extraktivní věta, ale u malého modelu "
        "může ustřelovat (zvlášť česká shoda). Uvádí zdroj pasáže."
    )
