from dataprep.wiki import _slug, fetch_articles


def test_slug():
    assert _slug("Karel Čapek") == "karel_čapek"
    assert _slug("R.U.R.") == "r.u.r."


def test_fetch_articles_writes_files(tmp_path):
    fake = {"Karel Čapek": "Text o Čapkovi.", "Prázdný": "  "}
    written = fetch_articles(list(fake), str(tmp_path), fetch=lambda t: fake[t])
    assert len(written) == 1  # prázdný článek se přeskočí
    assert (tmp_path / "wiki_karel_čapek.txt").read_text(encoding="utf-8") == \
        "Text o Čapkovi."
