"""Unit tests for the sched-comparison CSV loader/grouping."""
import csv
import os

from benchmark.compare_sched import load_tagged, group_by_sched


def _write_csv(path, rows):
    cols = ["backend", "sched", "scene", "width", "compute_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_tagged_reads_sched_column(tmp_path):
    p = os.path.join(tmp_path, "tt.csv")
    _write_csv(p, [{"backend": "tt", "sched": "lpt", "scene": "luigi",
                    "width": 256, "compute_ms": 4.2}])
    rows = load_tagged(p)
    assert rows[0]["sched"] == "lpt"
    assert rows[0]["width"] == 256        # coerced to int
    assert rows[0]["compute_ms"] == 4.2   # coerced to float


def test_load_tagged_falls_back_to_dir_name(tmp_path):
    # CSV with NO sched column -> label from parent dir "sched_segmented".
    d = os.path.join(tmp_path, "sched_segmented")
    os.makedirs(d)
    p = os.path.join(d, "tt.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "scene", "width", "compute_ms"])
        w.writeheader()
        w.writerow({"backend": "tt", "scene": "luigi", "width": 256, "compute_ms": 9.0})
    rows = load_tagged(p)
    assert rows[0]["sched"] == "segmented"   # "sched_" prefix stripped


def test_group_by_sched_partitions_rows():
    rows = [{"sched": "lpt", "scene": "luigi"},
            {"sched": "lpt", "scene": "train"},
            {"sched": "round_robin", "scene": "luigi"}]
    groups = group_by_sched(rows)
    assert set(groups) == {"lpt", "round_robin"}
    assert len(groups["lpt"]) == 2
    assert len(groups["round_robin"]) == 1
