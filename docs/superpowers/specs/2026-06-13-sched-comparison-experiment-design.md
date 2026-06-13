# Scheduling-strategy comparison experiment (TT backend)

**Date:** 2026-06-13
**Status:** approved, ready for implementation plan

## Goal

Produce a **performance comparison table** (latency / FPS / phase buckets) of
three tile→core scheduling strategies on the Tenstorrent backend, across both
tracked scenes (`scenes/luigi.ply`, `scenes/train.ply`) and the standard
resolution sweep. The three configs:

1. **`round_robin`** — load-blind tile→core assignment, single-op (default) kernel.
2. **`lpt`** — greedy longest-processing-time assignment, single-op kernel.
3. **`segmented`** — two-phase partial+combine kernel with greedy-LPT over
   depth-segments (the current default path). **This is the default config.**

The metric that actually moves across the three is `compute_ms` /
`device_kernel_ms` — kernel wall-clock is bound by the densest core's load, and
that is precisely what load-balancing (LPT) and intra-tile splitting (segmented)
attack. The existing load/compute/return/transfer bucket model already records
it; no new metric is introduced.

## Non-goals

- No new correctness/pytest suite (PSNR regression). Output is the perf table.
- No automatic stitching of the three runs into a single device session — the
  experiment is three separate invocations, one per config (user's choice).

## Approach

Approach A from brainstorming: an explicit `sched` knob on `KernelBackend` plus
a `--sched` CLI flag on the benchmark runner. Rejected alternatives: env-var-only
(implicit/global), and three separate backend classes (95% duplicated `blend()`).

The two single-op configs (`round_robin`, `lpt`) reuse the **entire existing
single-op `blend()` body** — only the assignment-builder function differs.
`segmented` is the existing two-phase path unchanged. The only genuinely new
production code is the round-robin builder.

## Components

### 1. Round-robin assignment builder — `backends/tt/lpt.py`

New function, same return contract as `build_tile_assignment` so the single-op
path consumes it unchanged:

```python
def build_round_robin_assignment(offsets, num_tiles, num_cores):
    """Load-blind tile->core assignment: non-empty tile t -> core (t % num_cores).

    Returns (per_core_offset, per_core_count, tile_ids) identical in shape to
    build_tile_assignment. Empty tiles are dropped (their output slots stay
    zero via the op's pre-zero, exactly as in the LPT builder); each surviving
    tile t is appended to core (t % num_cores). Tiles are walked in screen
    (ascending tile-id) order, so a core's tile_ids slice is ascending.
    """
```

- Walk `t` in `range(num_tiles)`; skip `loads[t] <= 0`.
- Append `t` to `buckets[t % num_cores]`.
- Lay buckets out core-contiguous into `tile_ids`, filling `per_core_offset` /
  `per_core_count` — identical packing loop to `build_tile_assignment`.

Note this is load-blind by *original tile index*, so consecutive non-empty
tiles 64 apart can collide on one core — the genuine naive baseline (vs. LPT's
balanced bins).

### 2. Backend dispatch — `backends/tt/backend.py`

`KernelBackend.__init__` gains a `sched` param:

```python
def __init__(self, verbose=False, sched=None):
    ...
    # Scheduling strategy: "round_robin" | "lpt" | "segmented" (default).
    # Precedence: explicit arg > GSPLAT_TT_SCHED env > GSPLAT_TT_SPLIT=0 alias
    # (-> "lpt", back-compat) > "segmented".
    self._sched = (sched or os.environ.get("GSPLAT_TT_SCHED")
                   or ("lpt" if os.environ.get("GSPLAT_TT_SPLIT") == "0" else "segmented"))
    if self._sched not in ("round_robin", "lpt", "segmented"):
        raise ValueError(f"unknown sched {self._sched!r}; "
                         "expected round_robin | lpt | segmented")
```

The existing `self._split_enabled` field is removed; its role is subsumed by
`self._sched == "segmented"`.

`blend()` dispatch:
- `segmented`: the existing heavy-tile-split check + `build_segmented_assignment`
  + `_blend_two_phase`. Unchanged logic, just gated on `self._sched == "segmented"`
  instead of `self._split_enabled`.
- `lpt` / `round_robin`: fall through to the existing single-op body. The only
  change there is selecting the builder:

  ```python
  builder = {"lpt": build_tile_assignment,
             "round_robin": build_round_robin_assignment}[self._sched]
  per_core_offset, per_core_count, tile_ids = builder(offsets, num_tiles, self.num_cores)
  ```

The `MAX_TILE_IDS_PER_CORE` overflow guard already in the single-op path stays
and now also guards `round_robin` (which can pile many tiles on one core).

Import `build_round_robin_assignment` alongside the existing `build_tile_assignment`.

### 3. Benchmark runner — `benchmark/run.py`

- New arg: `--sched {round_robin,lpt,segmented}`, default `segmented`.
- Construct the backend with it, but only pass `sched` for the TT backend
  (cpu/cuda constructors don't accept it):

  ```python
  kw = {"sched": args.sched} if args.backend == "tt" else {}
  backend = get_backend(args.backend, **kw)
  ```

- Default `--out` becomes `benchmark/results/sched_<sched>` when `--out` is not
  given, so the three invocations don't overwrite each other. Explicit `--out`
  still overrides.
- Add a `sched` field to each record and `"sched"` to `CSV_COLUMNS` (so a
  concatenation of the three CSVs is self-identifying). For non-TT backends the
  column is still written (value = `args.sched`, default `segmented`) — harmless
  and keeps the schema uniform.

Resulting experiment = three commands (both scenes + full res sweep are
defaults):

```bash
python -m benchmark.run tt --sched round_robin
python -m benchmark.run tt --sched lpt
python -m benchmark.run tt --sched segmented
```

→ `benchmark/results/sched_round_robin/tt.csv`, `.../sched_lpt/tt.csv`,
`.../sched_segmented/tt.csv` (+ matching `.json`).

### 4. Comparison helper — `benchmark/compare_sched.py` (new)

Mirrors `benchmark/plot.py` style (matplotlib Agg, same `_load_csv`-style
coercion). CLI:

```bash
python -m benchmark.compare_sched \
    benchmark/results/sched_round_robin/tt.csv \
    benchmark/results/sched_lpt/tt.csv \
    benchmark/results/sched_segmented/tt.csv \
    --out benchmark/results/sched_compare
```

- Reads N CSVs, groups rows by the `sched` column (falls back to the parent dir
  name if the column is absent, for resilience).
- Emits **one grouped-bar plot per scene**: x-axis = resolution, one bar group
  per resolution, N bars per group (one per sched), y-axis = `compute_ms` (the
  metric that moves). Missing cells → `NaN` bar (visible gap, not zero-filled),
  matching `plot_cross_backend`.
- Writes a **combined CSV** (`sched_compare.csv`) = all input rows concatenated
  with the `sched` column, for downstream analysis.
- File naming: `sched_compare_<scene>_compute.png` per scene.

## Data flow

```
build_round_robin_assignment / build_tile_assignment  (lpt.py)
        |  (per_core_offset, per_core_count, tile_ids)
        v
KernelBackend.blend()  --sched--> single-op path (round_robin | lpt)
                              \--> two-phase path  (segmented, default)
        |  image + timings (incl. compute/device_kernel buckets)
        v
benchmark/run.py  --> results/sched_<mode>/tt.{csv,json}  (with `sched` column)
        |
        v
benchmark/compare_sched.py  --> grouped-bar PNG per scene + combined CSV
```

## Error handling

- Invalid `sched` value: `ValueError` at backend construction (fail loud, early).
- `MAX_TILE_IDS_PER_CORE` overflow under round-robin: reuses the existing single-op
  `RuntimeError` guard (lower resolution / raise the cap).
- `compare_sched.py` with a CSV lacking the `sched` column: fall back to the
  parent directory name as the group label; warn but don't crash.

## Testing

- Extend `tests/test_lpt.py` with unit coverage for `build_round_robin_assignment`:
  correct `t % num_cores` placement, empty-tile dropping, return-shape parity
  with `build_tile_assignment`, ascending tile_ids per core, `per_core_offset`/
  `per_core_count` consistency.
- No new device test required (the experiment is a benchmark, not a regression
  gate); the existing `tests/test_kernel_integration.py` continues to exercise
  the default `segmented` path.

## Back-compat

- `GSPLAT_TT_SPLIT=0` continues to force the single-op LPT path (now via the
  `sched` alias). Default behavior (no env, no arg) is unchanged: `segmented`.
- `get_backend("tt")` with no kwargs still constructs the default backend.
