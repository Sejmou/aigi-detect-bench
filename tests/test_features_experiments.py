"""CPU-only tests for the feature cache and experiment plumbing."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aigi_bench.experiments import Corpus, fit_probe, load_corpus
from aigi_bench.features import available_conditions, condition_key


def test_condition_key_clean():
    assert condition_key("clean", None) == "clean"


@pytest.mark.parametrize("name,val,want", [
    ("jpeg", 90, "jpeg_90"),
    ("resize", 0.5, "resize_0p5"),
    ("blur", 2.0, "blur_2"),
    ("crop", 0.75, "crop_0p75"),
])
def test_condition_key_is_filesystem_safe(name, val, want):
    key = condition_key(name, val)
    assert key == want
    assert "." not in key and "/" not in key


def _write_cache(tmp_path, paths, feats, key="clean"):
    np.savez(tmp_path / f"{key}.npz", feats=feats, paths=np.array(paths, dtype=object))


def _manifest(paths, labels):
    return pl.DataFrame({
        "normalized_path": paths,
        "label": pl.Series(labels, dtype=pl.Int8),
        "split": ["train"] * len(paths),
    })


def test_load_corpus_realigns_when_manifest_order_differs(tmp_path):
    """The cache stores its own paths, so a re-sorted manifest must still align."""
    paths = [f"/x/{i}.jpg" for i in range(4)]
    feats = np.arange(8, dtype=np.float32).reshape(4, 2)
    _write_cache(tmp_path, paths, feats)

    # manifest in REVERSE order relative to the cache
    m = _manifest(paths[::-1], [0, 0, 1, 1])
    mp = tmp_path / "m.parquet"
    m.write_parquet(mp)

    c = load_corpus(mp, tmp_path, "clean")
    # row i of the corpus must carry the features of ITS OWN path
    for i, p in enumerate(m["normalized_path"]):
        assert np.array_equal(c.feats[i], feats[paths.index(p)])


def test_load_corpus_raises_on_missing_paths(tmp_path):
    _write_cache(tmp_path, ["/x/0.jpg"], np.zeros((1, 2), dtype=np.float32))
    m = _manifest(["/x/0.jpg", "/x/absent.jpg"], [0, 1])
    mp = tmp_path / "m.parquet"
    m.write_parquet(mp)
    with pytest.raises(KeyError, match="absent"):
        load_corpus(mp, tmp_path, "clean")


def test_available_conditions_lists_cache(tmp_path):
    for k in ("clean", "jpeg_90"):
        _write_cache(tmp_path, ["/a.jpg"], np.zeros((1, 2), dtype=np.float32), key=k)
    assert available_conditions(tmp_path) == ["clean", "jpeg_90"]


def test_corpus_mask_and_subset():
    df = pl.DataFrame({
        "label": pl.Series([0, 1, 0, 1], dtype=pl.Int8),
        "split": ["train", "train", "test", "test"],
    })
    c = Corpus(df=df, feats=np.arange(8, dtype=np.float32).reshape(4, 2))
    m = c.mask(pl.col("split") == "test")
    x, y = c.subset(m)
    assert x.shape == (2, 2)
    assert list(y) == [0, 1]


def test_fit_probe_separates_linearly_separable_data():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(-1, 0.1, (50, 3)), rng.normal(1, 0.1, (50, 3))].astype(np.float32)
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    probe = fit_probe(x, y)
    assert (probe.predict(x) == y).mean() == 1.0
