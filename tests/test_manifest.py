"""CPU-only tests for the manifest helpers (no corpus, no GPU)."""
from __future__ import annotations

import pytest

from aigi_bench.manifest import SPLITS, assign_split, parse_run_dir


def test_parse_run_dir_text_variant():
    assert parse_run_dir("z-image-turbo_qwen3-vl_text") == {
        "generator": "z-image-turbo",
        "captioner": "qwen3-vl",
        "prompt_variant": "text",
    }


def test_parse_run_dir_no_text_variant():
    # "no-text" is hyphenated, not underscored — that is what keeps the
    # 3-way split unambiguous.
    assert parse_run_dir("dreamshaper-8_qwen3-vl_no-text")["prompt_variant"] == "no-text"


def test_parse_run_dir_keeps_hyphenated_generator_names():
    assert parse_run_dir("qwen-image-2512_qwen3-vl_text")["generator"] == "qwen-image-2512"


@pytest.mark.parametrize("bad", ["justonepart", "a_b", "a_b_c_d", "gen_qwen3-vl_bogus"])
def test_parse_run_dir_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_run_dir(bad)


def test_assign_split_is_deterministic():
    assert assign_split("abc", 42, 0.6) == assign_split("abc", 42, 0.6)


def test_assign_split_depends_on_seed():
    ids = [f"id{i}" for i in range(500)]
    a = [assign_split(i, 1, 0.6) for i in ids]
    b = [assign_split(i, 2, 0.6) for i in ids]
    assert a != b


def test_assign_split_respects_train_frac():
    ids = [f"id{i}" for i in range(5000)]
    got = [assign_split(i, 42, 0.6) for i in ids]
    assert set(got) <= set(SPLITS)
    train = got.count("train") / len(got)
    calib = got.count("calib") / len(got)
    test = got.count("test") / len(got)
    assert train == pytest.approx(0.6, abs=0.03)
    # remainder split evenly between calib and test
    assert calib == pytest.approx(0.2, abs=0.03)
    assert test == pytest.approx(0.2, abs=0.03)


def test_assign_split_is_stable_under_added_keys():
    """Adding IDs must not reshuffle existing assignments (hash, not shuffle)."""
    before = {i: assign_split(i, 42, 0.6) for i in ("a", "b", "c")}
    after = {i: assign_split(i, 42, 0.6) for i in ("a", "x", "b", "y", "c", "z")}
    assert all(after[k] == v for k, v in before.items())
