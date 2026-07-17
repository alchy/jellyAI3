"""FocusState — rozpracovaný dialog automatu Iris.

Samotné aktivační pole (jas uzlů) drží answerer (`ActivationField`) — tady
žije jen to, co automat potřebuje MEZI tahy: rozpracované doostření
(`pending`: na co jsme se ptali a jací kandidáti jsou ve hře) a historie
tahů pro pozdější introspekci/ladění.
"""

from dataclasses import dataclass, field


@dataclass
class PendingFocus:
    """Rozpracované doostření — čekáme na volbu uživatele.

    Atributy:
        question (str): Původní otázka, která dialog vyvolala; po volbě se
            přehraje zaostřená (nejednoznačný termín → vybraný kandidát).
        term (str): Nejednoznačný termín otázky („Čapek").
        candidates (list[str]): Nabídnutí kandidáti (id uzlů).
        card (str): Jméno pattern-karty, která dialog vyvolala.
    """
    question: str
    term: str
    candidates: list
    card: str


@dataclass
class FocusState:
    """Stav automatu mezi tahy: rozpracovaný dialog + historie tahů."""
    pending: PendingFocus = None
    history: list = field(default_factory=list)   # [{"text", "kind", …}]

    def remember(self, text, response):
        """Zapíše tah do historie (vstup + druh a jistota výstupu)."""
        self.history.append({"text": text, "kind": response.kind,
                             "assurance": response.assurance})
