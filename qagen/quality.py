"""Filtr kvality vygenerovaných otázek.

Šablona z věty občas vyrobí paskvil — otázku, co začíná čárkou, skoro bez slov,
nebo odpověď, co je jen fragment. Tenhle filtr takové páry zahodí, aby se do
datasetu dostaly jen ty aspoň trochu smysluplné. Je to poslední síto před zápisem
(a laskavě zabrání tomu, aby se generátor učil z nesmyslů).
"""

# Znaky, kterými by rozumná otázka po tázacím slově začínat neměla.
_LEADING_BAD = set(",.;:)-–—…!?")


def is_acceptable(question, answer):
    """Rozhodne, zda je pár (otázka, odpověď) dost dobrý na zařazení do datasetu.

    Kritéria: odpověď není triviální fragment; otázka za tázacím slovem nezačíná
    interpunkcí; zbytek otázky má aspoň dvě slovná slova a nejsou to převážně
    čísla/interpunkce (podle podílu slovných slov).

    Args:
        question (str): Vygenerovaná otázka (začíná tázacím slovem, končí „?").
        answer (str): Odpovědní spán.

    Returns:
        bool: True, když pár projde všemi kritérii kvality.
    """
    a = answer.strip()
    if len(a) < 2 and not a.isdigit():
        return False

    parts = question.split(" ", 1)
    body = parts[1].rstrip("?").strip() if len(parts) > 1 else ""
    if not body or body[0] in _LEADING_BAD:
        return False

    words = body.split()
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if len(alpha_words) < 2:
        return False
    if len(alpha_words) / len(words) < 0.5:
        return False
    return True
