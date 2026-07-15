"""Sestavení otázky z věty a vybrané odpovědi.

Trik je jednoduchý a využívá volného slovosledu češtiny: z věty vyřízneme
odpovědní spán, zbytek očistíme a předsadíme tázací slovo. Z „Roboty vynalezl
starý Rossum." (odpověď „starý Rossum") tak vznikne „Kdo roboty vynalezl?".
Není to vždy stylisticky dokonalé (čeština si občas dělá, co chce), ale gramaticky
to drží a hlavně je dvojice (otázka, kontext, odpověď) koherentní — přesně to,
z čeho se má model naučit „najdi odpověď v kontextu".
"""

import re


def build_question(sentence, candidate):
    """Složí otázku odstraněním odpovědi z věty a předsazením tázacího slova.

    Args:
        sentence (str): Původní věta.
        candidate (Candidate): Odpověď (se znakovými offsety) a typ otázky.

    Returns:
        str: Otázka ve tvaru „<tázací slovo> <zbytek věty>?".
    """
    remainder = sentence[:candidate.start] + " " + sentence[candidate.end:]
    remainder = re.sub(r"\s+", " ", remainder).strip()
    remainder = remainder.rstrip(" .!?,;:").strip()
    if remainder:
        remainder = remainder[0].lower() + remainder[1:]
    question = f"{candidate.qtype} {remainder}?"
    return re.sub(r"\s+", " ", question).strip()
