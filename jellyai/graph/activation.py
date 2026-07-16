"""Aktivační pole — rozsvěcení/pohasínání „těžiště" nad lineárním proudem.

Jeden malý primitiv pro tři použití: (1) při extrakci drží „aktuální subjekt" a
řeší pro-drop/koreferenci (věta bez podmětu → nejteplejší entita); (2) retrieval
podle vzdálenosti (B1); (3) konverzační těžiště napříč dialogem (B2). Model je
prostý: `warm` přičte jas, `step` vše pohasí (× faktor), `hottest` vrátí nejteplejší
klíč. Klíčem může být cokoli hashovatelného (entita, id uzlu).
"""


class ActivationField:
    """Slábnoucí mapa aktivace `klíč → jas` nad sekvencí kroků."""

    def __init__(self, decay=0.55, floor=1e-3):
        """Vytvoří prázdné pole.

        Args:
            decay (float): Faktor pohasínání na krok (0–1); nižší = kratší paměť.
            floor (float): Pod tímto jasem se klíč zapomene (úklid).
        """
        self.decay = decay
        self.floor = floor
        self.scores = {}

    def warm(self, key, amount=1.0):
        """Přičte klíči jas (rozsvítí ho).

        Args:
            key: Hashovatelný klíč (entita, id uzlu).
            amount (float): O kolik zvýšit jas.
        """
        self.scores[key] = self.scores.get(key, 0.0) + amount

    def step(self):
        """Pohasí všechny klíče o `decay` a zapomene ty pod prahem."""
        self.scores = {k: v * self.decay for k, v in self.scores.items()
                       if v * self.decay > self.floor}

    def hottest(self):
        """Vrátí nejteplejší klíč (nebo None při prázdném poli)."""
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def to_dict(self):
        """Serializuje pole do prostého dictu (JSON-friendly)."""
        return {"decay": self.decay, "floor": self.floor, "scores": dict(self.scores)}

    @classmethod
    def from_dict(cls, data):
        """Obnoví pole z dictu (viz `to_dict`)."""
        field = cls(decay=data.get("decay", 0.55), floor=data.get("floor", 1e-3))
        field.scores = dict(data.get("scores", {}))
        return field
