from dataprep.clean import clean_text, build_processed


def test_clean_keeps_diacritics_and_punctuation():
    raw = "Příliš   žluťoučký\r\nkůň, úpěl!  "
    out = clean_text(raw)
    assert "ž" in out and "ě" in out
    assert "," in out and "!" in out
    assert "  " not in out  # zdvojené mezery sjednoceny


def test_clean_strips_gutenberg_boilerplate():
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK RUR ***\n"
        "Skutečný text díla.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK RUR ***\n"
        "License blabla"
    )
    out = clean_text(raw)
    assert "Skutečný text díla." in out
    assert "START OF" not in out
    assert "License" not in out


def test_clean_strips_wiki_references_and_headers():
    raw = (
        "Karel Čapek byl spisovatel.\n"
        "== Dílo ==\n"
        "Napsal R.U.R.\n"
        "== Odkazy ==\n"
        "=== Reference ===\n"
        "ISBN 80-7185-332-1\n"
        "Melantrich 1993\n"
    )
    out = clean_text(raw)
    assert "Karel Čapek byl spisovatel." in out
    assert "Napsal R.U.R." in out
    assert "Dílo" not in out       # sekční nadpis zahozen
    assert "ISBN" not in out       # referenční část odříznuta
    assert "Melantrich" not in out


def test_build_processed(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "kniha.txt").write_text("Text   knihy.", encoding="utf-8")
    written = build_processed(str(raw_dir), str(proc_dir))
    assert len(written) == 1
    assert (proc_dir / "kniha.txt").read_text(encoding="utf-8") == "Text knihy."


def test_build_processed_removes_stale(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    proc_dir.mkdir()
    # V processed straší starý text, který v raw už není.
    (proc_dir / "stary.txt").write_text("zbytek", encoding="utf-8")
    (raw_dir / "novy.txt").write_text("Nový text.", encoding="utf-8")
    build_processed(str(raw_dir), str(proc_dir))
    assert (proc_dir / "novy.txt").exists()
    assert not (proc_dir / "stary.txt").exists()  # osiřelý byl odstraněn
