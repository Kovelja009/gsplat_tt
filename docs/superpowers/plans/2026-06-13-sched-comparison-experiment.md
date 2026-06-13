# Scheduling-strategy comparison experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--sched {round_robin,lpt,segmented}` knob to the TT benchmark so the three tile→core scheduling strategies can be measured across both scenes, plus a helper that plots the comparison.

**Architecture:** A new load-blind round-robin assignment builder joins the existing greedy-LPT builder in `backends/tt/lpt.py`. `KernelBackend` gains a `sched` field that selects between the existing single-op path (round_robin / lpt, differing only in which builder runs) and the existing two-phase path (segmented, the default). `benchmark/run.py` exposes `--sched`, tags every record with it, and writes per-config result dirs; `benchmark/compare_sched.py` reads the three CSVs and emits a grouped-bar plot of `compute_ms` plus a combined CSV.

**Tech Stack:** Python 3.12, numpy, ttnn, matplotlib (Agg), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-sched-comparison-experiment-design.md`

---

### Task 1: Round-robin assignment builder

**Files:**
- Modify: `backends/tt/lpt.py` (add `build_round_robin_assignment`)
- Test: `tests/test_lpt.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lpt.py`. Update the import line at the top of the file from
`from backends.tt.lpt import build_tile_assignment` to:

```python
from backends.tt.lpt import build_tile_assignment, build_round_robin_assignment
```

Then append:

```python
def test_round_robin_assigns_by_tile_index_modulo_cores():
    # per-tile counts [10, 0, 5, 20, 7] -> offsets [0,10,10,15,35,42].
    # Non-empty tiles in screen order: 0, 2, 3, 4.
    # Round-robin by ORIGINAL index t % num_cores (num_cores=2):
    #   tile 0 -> core 0, tile 2 -> core 0, tile 3 -> core 1, tile 4 -> core 0.
    offsets = np.array([0, 10, 10, 15, 35, 42], dtype=np.uint32)
    per_core_offset, per_core_count, tile_ids = build_round_robin_assignment(
        offsets, num_tiles=5, num_cores=2)

    def core_tiles(c):
        s, n = int(per_core_offset[c]), int(per_core_count[c])
        return tile_ids[s:s + n].tolist()

    assert core_tiles(0) == [0, 2, 4]   # ascending, t % 2 == 0
    assert core_tiles(1) == [3]         # t % 2 == 1
    # Empty tile 1 dropped.
    assert sorted(core_tiles(0) + core_tiles(1)) == [0, 2, 3, 4]


def test_round_robin_slices_contiguous_and_shapes_match_lpt():
    offsets = np.array([0, 3, 7, 8, 8, 15], dtype=np.uint32)
    num_tiles, num_cores = 5, 3
    rr = build_round_robin_assignment(offsets, num_tiles, num_cores)
    lpt = build_tile_assignment(offsets, num_tiles, num_cores)
    # Same return contract: 3 arrays, per-core arrays sized num_cores,
    # same set of non-empty tiles covered.
    for got, exp in zip(rr, lpt):
        assert got.dtype == exp.dtype
    rr_off, rr_cnt, rr_ids = rr
    assert rr_off.shape == (num_cores,)
    assert rr_cnt.shape == (num_cores,)
    # Slices tile the array with no gaps/overlaps.
    cursor = 0
    for c in range(num_cores):
        assert int(rr_off[c]) == cursor
        cursor += int(rr_cnt[c])
    assert cursor == rr_ids.shape[0]
    # 4 non-empty tiles (tile 3 empty), same coverage as LPT.
    assert sorted(rr_ids.tolist()) == sorted(lpt[2].tolist())


def test_round_robin_all_empty_tiles():
    offsets = np.array([0, 0, 0], dtype=np.uint32)
    per_core_offset, per_core_count, tile_ids = build_round_robin_assignment(
        offsets, num_tiles=2, num_cores=4)
    assert tile_ids.shape[0] == 0
    assert per_core_count.sum() == 0
    assert per_core_offset.shape == (4,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_lpt.py -v`
Expected: the three new `test_round_robin_*` FAIL with `ImportError: cannot import name 'build_round_robin_assignment'`.

- [ ] **Step 3: Implement `build_round_robin_assignment`**

Add to `backends/tt/lpt.py` after `build_tile_assignment` (it reuses the same
core-contiguous packing loop, so keep it adjacent):

```python
def build_round_robin_assignment(offsets: np.ndarray, num_tiles: int, num_cores: int):
    """Load-blind tile->core assignment: non-empty tile t -> core (t % num_cores).

    Baseline against build_tile_assignment's greedy-LPT. Empty tiles are dropped
    (their output slots stay zero via the op's pre-zero, exactly as in the LPT
    builder); each surviving tile t is appended to core (t % num_cores). Tiles
    are walked in ascending tile-id order, so a core's tile_ids slice is
    ascending. Returns the same (per_core_offset, per_core_count, tile_ids)
    tuple as build_tile_assignment, so the single-op path consumes it unchanged.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    loads = offsets[1:num_tiles + 1] - offsets[:num_tiles]

    buckets: list[list[int]] = [[] for _ in range(num_cores)]
    for t in range(num_tiles):
        if loads[t] > 0:
            buckets[t % num_cores].append(t)

    per_core_offset = np.zeros(num_cores, dtype=np.uint32)
    per_core_count = np.zeros(num_cores, dtype=np.uint32)
    tile_ids_list: list[int] = []
    for c in range(num_cores):
        per_core_offset[c] = len(tile_ids_list)
        per_core_count[c] = len(buckets[c])
        tile_ids_list.extend(buckets[c])

    tile_ids = np.asarray(tile_ids_list, dtype=np.uint32)
    return per_core_offset, per_core_count, tile_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/test_lpt.py -v`
Expected: all tests PASS (the original LPT tests plus the three new ones).

- [ ] **Step 5: Commit**

```bash
git add backends/tt/lpt.py tests/test_lpt.py
git commit -m "feat(tt): add load-blind round-robin tile->core assignment

Baseline scheduler against greedy-LPT; same return contract so the
single-op blend path consumes it unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `sched` dispatch in `KernelBackend`

**Files:**
- Modify: `backends/tt/backend.py` (import, `__init__`, `blend()` dispatch)

This task is device code with no pure-unit test (it drives ttnn). Verify by
import + construction + the existing device test suite (skips cleanly with no
TT device). Follow each edit step exactly.

- [ ] **Step 1: Extend the import**

In `backends/tt/backend.py`, change:

```python
from backends.tt.lpt import build_tile_assignment, greedy_lpt
```

to:

```python
from backends.tt.lpt import build_tile_assignment, build_round_robin_assignment, greedy_lpt
```

- [ ] **Step 2: Replace the `_split_enabled` field with `_sched` in `__init__`**

In `__init__`, replace these lines:

```python
        # Intra-tile parallelism: when the segmented schedule splits a heavy tile
        # across cores, render via the two-phase partial+combine ops. On by
        # default; GSPLAT_TT_SPLIT=0 forces the legacy single-op path always.
        self._split_enabled = os.environ.get("GSPLAT_TT_SPLIT", "1") != "0"
```

with:

```python
        # Scheduling strategy: "round_robin" | "lpt" | "segmented" (default).
        # round_robin/lpt use the single-op kernel (differ only in the tile->core
        # assignment builder); segmented uses the two-phase partial+combine ops
        # with intra-tile depth-splitting. Precedence: explicit arg > GSPLAT_TT_SCHED
        # env > GSPLAT_TT_SPLIT=0 alias (-> "lpt", back-compat) > "segmented".
        self._sched = (sched or os.environ.get("GSPLAT_TT_SCHED")
                       or ("lpt" if os.environ.get("GSPLAT_TT_SPLIT") == "0" else "segmented"))
        if self._sched not in ("round_robin", "lpt", "segmented"):
            raise ValueError(
                f"unknown sched {self._sched!r}; expected round_robin | lpt | segmented")
```

- [ ] **Step 3: Add the `sched` parameter to the signature**

Change:

```python
    def __init__(self, verbose: bool = False):
```

to:

```python
    def __init__(self, verbose: bool = False, sched: str | None = None):
```

- [ ] **Step 4: Gate the two-phase path on `_sched`**

In `blend()`, change the segmented-path guard from:

```python
        if self._split_enabled:
```

to:

```python
        if self._sched == "segmented":
```

(The body below it — the `loads`/`target`/`build_segmented_assignment`/
`_blend_two_phase` block — is unchanged.)

- [ ] **Step 5: Select the builder in the single-op path**

In `blend()`, replace:

```python
        # --- LPT schedule (host) ---
        per_core_offset, per_core_count, tile_ids = build_tile_assignment(
            offsets, num_tiles, self.num_cores)
```

with:

```python
        # --- single-op schedule (host): lpt = greedy-LPT, round_robin = load-blind ---
        builder = {"lpt": build_tile_assignment,
                   "round_robin": build_round_robin_assignment}[self._sched]
        per_core_offset, per_core_count, tile_ids = builder(
            offsets, num_tiles, self.num_cores)
```

- [ ] **Step 6: Verify import + construction + dispatch wiring**

Run (no device needed — this only checks construction and validation):

```bash
source venv/bin/activate && python -c "
from backends.tt.backend import KernelBackend
import inspect, backends.tt.backend as m
src = inspect.getsource(KernelBackend.blend)
assert 'self._sched == \"segmented\"' in src, 'segmented gate missing'
assert 'build_round_robin_assignment' in src, 'round_robin builder not wired'
assert not hasattr(KernelBackend, '_split_enabled'), 'stale field on class'
print('dispatch wiring OK')
"
```

Expected: prints `dispatch wiring OK`. (Full `KernelBackend()` construction
needs a TT device; the source-level checks above confirm the wiring without one.)

- [ ] **Step 7: Run the device + cache test suite (skips cleanly with no device)**

Run: `source venv/bin/activate && pytest tests/test_kernel_integration.py tests/test_ttnn_op_cache.py tests/test_tt_segments.py -v`
Expected: PASS or SKIP (skips if no TT device present); no errors/failures.

- [ ] **Step 8: Commit**

```bash
git add backends/tt/backend.py
git commit -m "feat(tt): sched knob selecting round_robin | lpt | segmented

Replaces the GSPLAT_TT_SPLIT bool with an explicit three-way scheduling
strategy (default segmented). GSPLAT_TT_SPLIT=0 kept as an lpt alias.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `--sched` flag + `sched` column in `benchmark/run.py`

**Files:**
- Modify: `benchmark/run.py` (`CSV_COLUMNS`, `_build_record`, `main`)

No standalone unit test (it's a CLI orchestrator); verify via `--help` and a
schema check. The bucket model it relies on is already covered by
`tests/test_benchmark_phases.py`.

- [ ] **Step 1: Add `sched` to `CSV_COLUMNS`**

In `benchmark/run.py`, change the `CSV_COLUMNS` list — insert `"sched"` right
after `"backend"`:

```python
CSV_COLUMNS = [
    "backend", "sched", "scene", "n_gaussians", "height", "width",
    "num_visible", "num_entries",
    "project_ms", "tile_assign_ms", "sort_ms", "blend_ms", "total_ms", "fps",
    "load_ms", "compute_ms", "return_ms", "transfer_ms", "device_kernel_ms",
    "blend_ms_min", "total_ms_min",
]
```

- [ ] **Step 2: Thread `sched` into `_build_record`**

Change the signature:

```python
def _build_record(backend, scene_name, n_gauss, H, W, med, mins, buckets, last):
```

to:

```python
def _build_record(backend, sched, scene_name, n_gauss, H, W, med, mins, buckets, last):
```

and change the opening of the returned dict from:

```python
        "backend": backend, "scene": scene_name, "n_gaussians": n_gauss,
```

to:

```python
        "backend": backend, "sched": sched, "scene": scene_name, "n_gaussians": n_gauss,
```

- [ ] **Step 3: Add the `--sched` arg and out-dir default in `main`**

In `main`, after the `--out` argument line:

```python
    ap.add_argument("--out", default="benchmark/results")
```

change it to default `None` and add `--sched`:

```python
    ap.add_argument("--out", default=None,
                    help="output dir (default: benchmark/results/sched_<sched>)")
    ap.add_argument("--sched", choices=["round_robin", "lpt", "segmented"],
                    default="segmented",
                    help="TT tile->core scheduling strategy (ignored for cpu/cuda)")
```

- [ ] **Step 4: Construct the backend with `sched` (TT only) and resolve `--out`**

Change:

```python
    try:
        backend = get_backend(args.backend)
    except (KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: backend {args.backend!r} unavailable: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
```

to:

```python
    kw = {"sched": args.sched} if args.backend == "tt" else {}
    try:
        backend = get_backend(args.backend, **kw)
    except (KeyError, RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: backend {args.backend!r} unavailable: {e}", file=sys.stderr)
        return 2

    out_dir = args.out or os.path.join("benchmark/results", f"sched_{args.sched}")
    os.makedirs(out_dir, exist_ok=True)
```

- [ ] **Step 5: Pass `sched` into `_build_record` and use `out_dir` for paths**

Change the `_build_record` call:

```python
                rec = _build_record(args.backend, scene_name,
                                    gaussians.num_gaussians, H, W, med, mins, bucket_med, last)
```

to:

```python
                rec = _build_record(args.backend, args.sched, scene_name,
                                    gaussians.num_gaussians, H, W, med, mins, bucket_med, last)
```

Then change the two output-path lines near the end of `main`:

```python
    csv_path = os.path.join(args.out, f"{args.backend}.csv")
    json_path = os.path.join(args.out, f"{args.backend}.json")
```

to:

```python
    csv_path = os.path.join(out_dir, f"{args.backend}.csv")
    json_path = os.path.join(out_dir, f"{args.backend}.json")
```

- [ ] **Step 6: Verify the CLI wiring**

Run:

```bash
source venv/bin/activate && python -m benchmark.run --help 2>&1 | grep -- --sched
```

Expected: a line showing `--sched {round_robin,lpt,segmented}`.

Then verify the record schema is consistent (no device needed):

```bash
source venv/bin/activate && python -c "
from benchmark.run import _build_record, CSV_COLUMNS
class L:  # stand-in for the render result
    num_visible = 1; num_entries = 1
rec = _build_record('tt', 'lpt', 'luigi', 100, 256, 256,
                    {'total': 10.0}, {}, {'load':1,'compute':2,'return':1,'transfer':1,'device_kernel':2}, L())
assert rec['sched'] == 'lpt'
missing = [c for c in CSV_COLUMNS if c not in rec]
assert not missing, f'record missing CSV columns: {missing}'
print('run.py schema OK')
"
```

Expected: prints `run.py schema OK`.

- [ ] **Step 7: Commit**

```bash
git add benchmark/run.py
git commit -m "feat(bench): --sched flag, sched column, per-config out dir

python -m benchmark.run tt --sched {round_robin,lpt,segmented} writes
benchmark/results/sched_<mode>/tt.{csv,json}; every row is tagged with
its sched. Flag is ignored for cpu/cuda.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `benchmark/compare_sched.py` comparison helper

**Files:**
- Create: `benchmark/compare_sched.py`
- Test: `tests/test_compare_sched.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_sched.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_compare_sched.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.compare_sched'`.

- [ ] **Step 3: Implement `benchmark/compare_sched.py`**

Create `benchmark/compare_sched.py`:

```python
"""Compare TT scheduling strategies: plot compute_ms per scene, grouped by sched.

    python -m benchmark.compare_sched \
        benchmark/results/sched_round_robin/tt.csv \
        benchmark/results/sched_lpt/tt.csv \
        benchmark/results/sched_segmented/tt.csv \
        --out benchmark/results/sched_compare

Reads each CSV, tags rows by their `sched` column (falling back to the parent
directory name with any "sched_" prefix stripped), then emits one grouped-bar
plot of compute_ms per scene (x = resolution, one bar per sched) plus a combined
CSV of all rows.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

_INT_FIELDS = ("height", "width", "num_visible", "num_entries", "n_gaussians")
_FLOAT_FIELDS = ("project_ms", "tile_assign_ms", "sort_ms", "blend_ms",
                 "total_ms", "fps", "load_ms", "compute_ms", "return_ms",
                 "transfer_ms", "device_kernel_ms", "blend_ms_min", "total_ms_min")


def load_tagged(path):
    """Read a benchmark CSV; return row-dicts with numerics coerced and a `sched`
    field guaranteed present (from the column, else the parent dir name with any
    'sched_' prefix stripped)."""
    fallback = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if fallback.startswith("sched_"):
        fallback = fallback[len("sched_"):]
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in _INT_FIELDS:
                if k in r and r[k] != "":
                    r[k] = int(float(r[k]))
            for k in _FLOAT_FIELDS:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            if not r.get("sched"):
                r["sched"] = fallback
            rows.append(r)
    return rows


def group_by_sched(rows):
    """Partition rows into {sched: [rows]} preserving input order."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["sched"], []).append(r)
    return groups


def _scenes(rows):
    return sorted({r["scene"] for r in rows})


def plot_compute_by_sched(groups, out_dir):
    """One grouped-bar chart per scene: x = resolution, one bar per sched,
    y = compute_ms. Missing cells render as NaN (visible gap)."""
    scheds = list(groups)
    all_scenes = sorted({r["scene"] for rows in groups.values() for r in rows})
    for scene in all_scenes:
        res_set = sorted({r["width"] for rows in groups.values()
                          for r in rows if r["scene"] == scene})
        x = np.arange(len(res_set))
        n = len(scheds); w = 0.8 / max(1, n)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, sched in enumerate(scheds):
            lut = {r["width"]: r["compute_ms"]
                   for r in groups[sched] if r["scene"] == scene}
            vals = [lut.get(res, np.nan) for res in res_set]
            ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=sched)
        ax.set_xticks(x); ax.set_xticklabels([str(r) for r in res_set])
        ax.set_xlabel("resolution (px)"); ax.set_ylabel("compute (ms)")
        ax.set_title(f"{scene}: kernel compute by scheduling strategy")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(out_dir, f"sched_compare_{scene}_compute.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"wrote {p}")


def write_combined_csv(groups, out_dir):
    """Concatenate all rows (with sched) into one CSV for downstream analysis."""
    all_rows = [r for rows in groups.values() for r in rows]
    if not all_rows:
        return
    cols = list(all_rows[0].keys())
    p = os.path.join(out_dir, "sched_compare.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare TT scheduling strategies from CSVs.")
    ap.add_argument("csv", nargs="+", help="per-config <backend>.csv files")
    ap.add_argument("--out", default="benchmark/results/sched_compare",
                    help="output dir for plots + combined CSV")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = [r for p in args.csv for r in load_tagged(p)]
    groups = group_by_sched(rows)
    plot_compute_by_sched(groups, args.out)
    write_combined_csv(groups, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source venv/bin/activate && pytest tests/test_compare_sched.py -v`
Expected: all three tests PASS.

- [ ] **Step 5: Smoke-test the plotting end-to-end with synthetic CSVs**

Run:

```bash
source venv/bin/activate && python -c "
import csv, os, tempfile
from benchmark.compare_sched import main
d = tempfile.mkdtemp()
def mk(sched, compute):
    p = os.path.join(d, f'{sched}.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['backend','sched','scene','width','compute_ms'])
        w.writeheader()
        for scene in ('luigi','train'):
            for res in (256, 480):
                w.writerow({'backend':'tt','sched':sched,'scene':scene,'width':res,'compute_ms':compute})
    return p
csvs = [mk('round_robin', 9.0), mk('lpt', 6.0), mk('segmented', 3.0)]
out = os.path.join(d, 'cmp')
main(csvs + ['--out', out])
pngs = sorted(f for f in os.listdir(out) if f.endswith('.png'))
assert pngs == ['sched_compare_luigi_compute.png', 'sched_compare_train_compute.png'], pngs
assert os.path.exists(os.path.join(out, 'sched_compare.csv'))
print('compare_sched smoke OK:', pngs)
"
```

Expected: prints `compare_sched smoke OK: ['sched_compare_luigi_compute.png', 'sched_compare_train_compute.png']`.

- [ ] **Step 6: Commit**

```bash
git add benchmark/compare_sched.py tests/test_compare_sched.py
git commit -m "feat(bench): compare_sched plots compute_ms across scheduling strategies

Reads the three sched_*/tt.csv files, groups by sched (falling back to the
parent dir name), emits a grouped-bar compute_ms plot per scene plus a
combined CSV.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run the experiment + document how

**Files:**
- Modify: `benchmark/README.md` if it exists, else `README.md` (add a short "Scheduling-strategy comparison" section)

This task produces the actual experiment artifacts and records the recipe. The
three benchmark commands require a TT device; if none is attached, do Step 1
(docs) and skip Steps 2-3, noting that they must be run on TT hardware.

- [ ] **Step 1: Document the experiment recipe**

Check whether `benchmark/README.md` exists:

```bash
ls benchmark/README.md 2>/dev/null && echo EXISTS || echo "use top-level README.md"
```

Add a section (to `benchmark/README.md` if it exists, otherwise append to the
top-level `README.md` under the benchmarking content) with this exact text:

````markdown
## Scheduling-strategy comparison (TT)

Compare the three tile→core scheduling strategies across both scenes:

```bash
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal

python -m benchmark.run tt --sched round_robin   # load-blind baseline, single-op kernel
python -m benchmark.run tt --sched lpt           # greedy-LPT, single-op kernel
python -m benchmark.run tt --sched segmented     # two-phase + LPT over depth-segments (default)

python -m benchmark.compare_sched \
    benchmark/results/sched_round_robin/tt.csv \
    benchmark/results/sched_lpt/tt.csv \
    benchmark/results/sched_segmented/tt.csv
```

Each `run` writes `benchmark/results/sched_<mode>/tt.{csv,json}`. `compare_sched`
writes a grouped-bar `compute_ms` plot per scene plus a combined CSV to
`benchmark/results/sched_compare/`. `compute_ms` (kernel wall-clock, bound by the
densest core's load) is the metric the scheduling strategy moves.
````

- [ ] **Step 2: Run the three benchmark sweeps (TT device required)**

```bash
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
python -m benchmark.run tt --sched round_robin
python -m benchmark.run tt --sched lpt
python -m benchmark.run tt --sched segmented
```

Expected: each prints per-(scene,res) lines and `Wrote N rows -> benchmark/results/sched_<mode>/tt.csv`.
Verify all three dirs exist:

```bash
ls benchmark/results/sched_round_robin/tt.csv benchmark/results/sched_lpt/tt.csv benchmark/results/sched_segmented/tt.csv
```

- [ ] **Step 3: Generate the comparison plot**

```bash
source venv/bin/activate
python -m benchmark.compare_sched \
    benchmark/results/sched_round_robin/tt.csv \
    benchmark/results/sched_lpt/tt.csv \
    benchmark/results/sched_segmented/tt.csv
```

Expected: prints `wrote .../sched_compare_luigi_compute.png`,
`wrote .../sched_compare_train_compute.png`, `wrote .../sched_compare.csv`.

- [ ] **Step 4: Commit docs (and results if produced)**

```bash
git add README.md benchmark/README.md 2>/dev/null; git add -A benchmark/results/sched_* 2>/dev/null
git commit -m "docs(bench): scheduling-strategy comparison recipe + results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** Round-robin builder (Task 1 = spec §Components.1), backend
  dispatch incl. `_split_enabled` removal + GSPLAT_TT_SPLIT alias (Task 2 =
  §Components.2 + §Back-compat), run.py `--sched`/column/out-dir (Task 3 =
  §Components.3), compare_sched plot+combined CSV with dir-name fallback (Task 4
  = §Components.4 + §Error handling), unit test for the builder (Task 1 Step 1 =
  §Testing), experiment recipe + run (Task 5 = §Goal). All spec sections mapped.
- **Type consistency:** `build_round_robin_assignment(offsets, num_tiles, num_cores)
  -> (per_core_offset, per_core_count, tile_ids)` used identically in Task 1
  (def), Task 2 (call via `builder`). `_build_record` gains `sched` as 2nd param
  in both its def and call (Task 3). `load_tagged` / `group_by_sched` /
  `plot_compute_by_sched` / `write_combined_csv` signatures match between
  definition (Task 4 Step 3) and tests (Task 4 Step 1).
- **No placeholders:** every code/edit step shows full content; every run step
  states the exact command and expected output.
