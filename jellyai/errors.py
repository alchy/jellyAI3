"""Chyby knihovny — srozumitelné a **akční** (říkají, jak to opravit)."""


class JellyError(Exception):
    """Základ chyb jellyAI3. Hláška má vždy vysvětlit i nápravu."""


class ModelsMissingError(JellyError):
    """Chybí ÚFAL modely."""

    def __init__(self, path):
        super().__init__(
            f"Modely nenalezeny v {path!r} — stáhni je příkazem: ./jelly qa-models")


class CorpusNotStartedError(JellyError):
    """Korpusové služby nejsou nastartované."""

    def __init__(self):
        super().__init__(
            "Korpus není nastartovaný — použij `with jellyai.CorpusTools() as t:` "
            "nebo zavolej `tools.start()`.")
