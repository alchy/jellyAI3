import os

from config import Config, DataConfig
from dataprep.download import book_targets


def test_book_targets_builds_paths():
    cfg = Config(data=DataConfig(
        raw_dir="data/raw",
        books=[("https://example.com/rur.txt", "rur.txt")],
    ))
    targets = book_targets(cfg)
    assert targets == [("https://example.com/rur.txt",
                        os.path.join("data/raw", "rur.txt"))]


def test_book_targets_empty():
    assert book_targets(Config()) == []
