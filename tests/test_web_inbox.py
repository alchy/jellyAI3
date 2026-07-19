"""DEV most webu: soubor-schránka `web_inbox.txt` — řádky se zpracují, jako
by je uživatel napsal do dialogového okna GUI (vidět otázka, odpověď,
aktivace i růst grafu). Testuje se čistý helper čtení-a-vyprázdnění."""

from cli import _drain_lines


def test_drain_reads_lines_and_empties_file(tmp_path):
    path = tmp_path / "inbox.txt"
    path.write_text("Prší?\n\n  \nKdo je Karel?\n", encoding="utf-8")
    assert _drain_lines(str(path)) == ["Prší?", "Kdo je Karel?"]
    assert path.read_text(encoding="utf-8") == ""      # vyprázdněno
    assert _drain_lines(str(path)) == []               # druhé čtení prázdné


def test_drain_missing_file_is_empty(tmp_path):
    assert _drain_lines(str(tmp_path / "neni.txt")) == []
