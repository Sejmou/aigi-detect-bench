import numpy as np
from PIL import Image

from aigi_bench.data import make_splits


def test_splits_are_deterministic_and_disjoint(tmp_path):
    for cls in ("real", "fake"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(20):
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(d / f"{i}.png")
    s1 = make_splits(tmp_path / "real", tmp_path / "fake", seed=1)
    s2 = make_splits(tmp_path / "real", tmp_path / "fake", seed=1)
    assert [str(p) for p in s1["train"].paths] == [str(p) for p in s2["train"].paths]
    all_paths = [str(p) for k in s1 for p in s1[k].paths]
    assert len(all_paths) == len(set(all_paths)) == 40
