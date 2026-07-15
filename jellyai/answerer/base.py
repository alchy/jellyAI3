"""Rozhraní bloku Answerer a datový typ odpovědi.

Answerer je poslední článek řetězu: dostane pasáže od retrieveru a má z nich
vyrobit odpověď. Schválně z něj děláme vyměnitelný blok se společným rozhraním —
dnes máme extraktivní verzi (vrátí větu z textu), zítra sem zapojíme generativní
seq2seq model (V2), aniž bychom sáhli na pipeline. Kontrakt zůstává stejný.
"""

from dataclasses import dataclass, field


@dataclass
class Answer:
    """Výsledek odpovědi na dotaz.

    Atributy:
        text (str): Samotná odpověď určená uživateli.
        sources (list[str]): Odkud odpověď pochází, ve tvaru `doc_id#index`.
            Prázdné, když se odpověď nenašla — nechceme si vymýšlet zdroje.
        score (float): Skóre důvěry (vyšší = lepší shoda); 0.0 znamená „nenašel".
    """
    text: str
    sources: list = field(default_factory=list)
    score: float = 0.0


class Answerer:
    """Abstraktní blok, který z nalezených pasáží skládá odpověď.

    Podtřídy implementují :meth:`answer`. Samotná základní třída odpověď neumí —
    je to jen „smlouva", ne pracant.
    """

    def answer(self, question, retrieved):
        """Složí odpověď na dotaz z nalezených pasáží.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list[tuple[Passage, float]]): Pasáže a jejich skóre
                z retrieveru (nejrelevantnější první).

        Returns:
            Answer: Odpověď se zdroji a skóre.

        Raises:
            NotImplementedError: Základní třída odpověď neposkytuje; použij
                konkrétní podtřídu (např. ExtractiveAnswerer).
        """
        raise NotImplementedError
