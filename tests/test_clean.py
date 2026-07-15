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


def test_build_processed(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "kniha.txt").write_text("Text   knihy.", encoding="utf-8")
    written = build_processed(str(raw_dir), str(proc_dir))
    assert len(written) == 1
    assert (proc_dir / "kniha.txt").read_text(encoding="utf-8") == "Text knihy."
