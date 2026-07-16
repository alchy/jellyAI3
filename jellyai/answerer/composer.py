"""Kompozitor — ze sady faktů složí čitelný text (víc než jednoslovná odpověď).

Port `Composer` je místo, kam později sedne malá generativní NN. Výchozí je
šablonový: každý fakt poskládá do jednoduché věty (podmět, predikát, doplnění).
Bere kandidátní fakty (třeba z teploty shody) a vrátí souvislý odstavec.
"""

_ROLE_PHRASE = {"num": "roku {}", "time": "{}", "loc": "v {}", "obj": "{}",
                "pred": "{}"}
_TAIL_ORDER = ("obj", "num", "time", "loc", "pred")


class TemplateComposer:
    """Výchozí kompozitor — spojí fakty do vět bez modelu."""

    def compose(self, question, facts):  # pylint: disable=unused-argument
        """Složí čitelný text ze sady faktů.

        `question` výchozí šablona nepoužije — je součástí portu pro budoucí NN
        (která ho k formulaci potřebuje).

        Args:
            question (str): Původní otázka (pro budoucí kontext; zde neužito).
            facts (list[Fact]): Kandidátní fakty.

        Returns:
            str: Souvislý text (prázdný, když nejsou fakty).
        """
        sentences = []
        for fact in facts:
            parts = {p.role: p.node for p in fact.participants}
            subj = parts.get("subj", "")
            tail = [_ROLE_PHRASE[role].format(parts[role])
                    for role in _TAIL_ORDER if role in parts]
            if subj and tail:
                sentences.append(f"{subj} {fact.predicate} {' '.join(tail)}".strip())
        text = ". ".join(s[0].upper() + s[1:] for s in sentences)
        return (text + ".") if text else ""
