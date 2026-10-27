"""memopro.sort / memopro.join (E069 V2, 0356): CSV larger than memory, sorted and joined without
disk writes of their own; results equal a plain in-memory sort and join."""

import collections
import random

import pytest

import memopro
from memopro import table

pytest.importorskip("numpy")


@pytest.fixture
def small(monkeypatch):
    """Blocks of 1 KiB (rows cross block edges) and a room of 3 KiB per pass (many passes)."""
    monkeypatch.setattr(table, "BLOCK", 1 << 10)
    monkeypatch.setattr(table, "CHUNK", 1 << 10)
    monkeypatch.setattr(table, "SAMPLE", 2)
    monkeypatch.setattr(table, "_room", lambda ceiling, reserve=0: 3 << 10)


def _rows(n, keys, seed):
    rng = random.Random(seed)
    return [f"{rng.choice(keys)},{i},{'x' * rng.randint(0, 40)}" for i in range(n)]


def test_sort_is_stable_and_equals_a_plain_sort(tmp_path, small):
    rows = _rows(3000, list(range(-50, 400)), 1)  # repeated and negative keys
    path = tmp_path / "a.csv"
    path.write_text("k,i,t\n" + "\n".join(rows) + "\n")
    got = bytearray()
    st = memopro.sort(path, "k", out=got.extend)
    want = sorted(rows, key=lambda r: int(r.split(",")[0]))  # Python's sort is stable
    assert bytes(got).decode() == "k,i,t\n" + "".join(r + "\n" for r in want)
    assert st["passes"] > 3 and st["rows"] == 3000
    memopro.sort(path, "i", out=tmp_path / "b.csv")  # to a file, by another column
    assert (tmp_path / "b.csv").read_text() == path.read_text()


def test_join_equals_a_plain_join(tmp_path, small):
    left = _rows(2000, list(range(300)), 2)
    rng = random.Random(3)
    right = [f"r{j},{rng.randrange(320)},{'y' * rng.randint(0, 9)}" for j in range(900)]
    (tmp_path / "l.csv").write_text("k,i,t\n" + "\n".join(left) + "\n")
    (tmp_path / "r.csv").write_text("name,k,w\n" + "\n".join(right) + "\n")
    got = bytearray()
    st = memopro.join(tmp_path / "l.csv", tmp_path / "r.csv", "k", out=got.extend)
    lines = bytes(got).decode().splitlines()
    assert lines[0] == "k,i,t,name,w"
    by_key = collections.defaultdict(list)
    for r in right:
        name, k, w = r.split(",")
        by_key[int(k)].append(f"{name},{w}")
    want = collections.Counter(
        f"{row},{rest}" for row in left for rest in by_key[int(row.split(",")[0])]
    )
    assert collections.Counter(lines[1:]) == want
    assert st["parts"] > 1


def test_bad_input_is_refused_not_misread(tmp_path, small):
    p = tmp_path / "q.csv"
    p.write_text('k,t\n1,"a,b"\n')
    with pytest.raises(memopro.InvalidArgument, match="quoted"):
        memopro.sort(p, "k", out=lambda b: None)
    p.write_text("k,t\n1,a\nx,b\n")
    with pytest.raises(memopro.InvalidArgument, match="integers"):
        memopro.sort(p, "k", out=lambda b: None)
    with pytest.raises(memopro.InvalidArgument, match="no column"):
        memopro.sort(p, "nope", out=lambda b: None)


def test_a_file_changed_between_passes_is_an_error(tmp_path, small, monkeypatch):
    rows = _rows(500, list(range(100)), 4)
    path = tmp_path / "a.csv"
    path.write_text("k,i,t\n" + "\n".join(rows) + "\n")
    ratio = table._ratio

    def change(src, col):
        out = ratio(src, col)
        path.write_text("k,i,t\n" + "\n".join(reversed(rows)) + "\n")
        return out

    monkeypatch.setattr(table, "_ratio", change)
    with pytest.raises(memopro.IntegrityError):
        memopro.sort(path, "k", out=lambda b: None)


def test_defaults_sort_a_small_file_in_one_bucket(tmp_path):
    rows = _rows(1000, list(range(1000)), 5)
    path = tmp_path / "a.csv"
    path.write_text("k,i,t\n" + "\n".join(rows) + "\n")
    got = bytearray()
    st = memopro.sort(path, "k", out=got.extend)
    want = sorted(rows, key=lambda r: int(r.split(",")[0]))
    assert bytes(got).decode() == "k,i,t\n" + "".join(r + "\n" for r in want)
    assert st["passes"] == 2 and st["runtime"]["written_bytes"] == 0
