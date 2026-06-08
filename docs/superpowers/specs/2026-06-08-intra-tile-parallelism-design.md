# Intra-tile parallelism (depth-segment + associative combine) — design

**Date:** 2026-06-08
**Status:** approved (pending spec review)

## Goal

Cut the alpha-blend kernel's wall-clock time on dense scenes by parallelizing
the **densest tile** across cores. Today a tile is the atomic unit of work (one
core composites all of a tile's Gaussians front-to-back), so kernel time is set
by the single densest tile's Gaussian count. For `train.ply` that tile holds
~63K genuinely-contributing low-opacity Gaussians, so the work is real and
irreducible per-pixel — the only lever is to spread it across cores.

Split a heavy tile's depth-sorted Gaussian list into **K contiguous segments**,
composite each into a partial `(R,G,B,T)` on its own core, then merge the
partials in depth order via the Porter-Duff **over** operator (which is
associative). This drives the bottleneck core's load toward the
`total_entries / num_cores` ideal.

Scope: forward-pass rendering only. No training. The existing single-tile path
is preserved for scenes with no heavy tiles.

## Background / why this approach (investigated 2026-06-08)

Measured on `train.ply` (Blackhole, 130 cores), confirming the lever:

- Kernel compute time is **linearly proportional to the single densest tile's
  Gaussian count** (256→960px: max_tile 75,325→15,749 = 4.78×; compute
  345→73ms = 4.75×). `max_core_load == max_single_tile` at every resolution —
  the LPT scheduler parks the monster tile alone on a core and cannot do better.
- **Whole-tile early-out does not help.** The CPU reference breaks when
  `T.max() < 1e-4`, but for `train@256` **0 of 64 tiles** ever fully saturate
  (median per-Gaussian α ≈ 0.04; low-opacity Gaussians spread within the tile
  always leave some pixel with `T ≥ 1e-4`). The kernel has no early-out anyway
  (Stage F masks saturated pixels but runs the SFPU at full cost).
- **The tile is not over-assigned.** 83.7% of the densest tile's 75K Gaussians
  contribute ≥ 1/255; only 16.3% are negligible. So culling is a minor (~16%)
  win, kept out of scope here.

Conclusion: the densest tile's ~63K composites are genuine work; spreading it
across cores (this design) is the right and necessary lever.

## Non-goals

- The `(gaussian, tile)` α<1/255 cull (a separate, ~16%, near-lossless host
  optimization — tracked separately, not in this spec).
- Per-pixel early-out (impossible on SIMD lock-step) or whole-tile early-out
  (proven useless for dense scenes above).
- Changing projection / tile-assignment / sort.
- Single-program cross-core combine via semaphores (Approach 2 — rejected as
  deadlock-prone; we use two op dispatches instead).

## Architecture

Two new ttnn ops; the existing `gaussian_alpha_blend` op is untouched.

```
host: project / tile_assign / sort  (CPU, unchanged)
   → prepare_kernel_inputs → build_segmented_assignment
   ├─ no heavy tiles → gaussian_alpha_blend (existing)            → out
   └─ heavy tiles    → gaussian_alpha_blend_partial → partials DRAM
                     → gaussian_alpha_blend_combine                → out
   → readback out → image
```

The split path runs only when the schedule contains at least one split tile, so
small scenes (e.g. luigi, no heavy tiles) keep the existing single-op fast path
and pay no combine cost.

## The math (associative `over`)

A segment composited with its own transmittance starting at 1 produces a partial
`(C_i, T_i)` where `C_i = Σ within-segment α·T_local·color` and
`T_i = Π within-segment (1-α)`. Merging segments 0..K-1 in depth order:

```
C_acc = 0 ; T_acc = 1
for i in 0..K-1:           # depth order
    C_acc += T_acc * C_i
    T_acc *= T_i
out = C_acc                # T_acc discarded
```

This reconstructs the exact monolithic front-to-back result in real arithmetic.
Only bf16 rounding differs (partials stored bf16; merge accumulates fp32 in Dst).

## Components

### 1. Host scheduler — `backends/tt/lpt.py`

`build_segmented_assignment(offsets, num_tiles, num_cores, target=None)`:

- `target = ceil(total_entries / num_cores)` when not given (the measured ideal).
- For each non-empty tile with load `L`: `n_seg = max(1, ceil(L / target))`;
  split its Gaussian range `[offsets[t], offsets[t+1])` into `n_seg` contiguous
  depth-ordered segments (sizes differ by ≤1).
- Each segment is a **job** `(tile_id, gseg_start, gseg_count)`. Jobs are
  LPT-balanced across cores by `gseg_count` (heaviest first), exactly like the
  current tile-level LPT.
- Partial slots: assign each job a `partial_slot` such that **all segments of a
  given tile occupy contiguous slots in depth order** (segment 0 first). This
  lets the combine read `partials[first_slot .. first_slot+K]` directly.
- Returns:
  - `per_core_offset`, `per_core_count`: into a concatenated **job table**
    (one entry = `(tile_id, gseg_start, gseg_count, partial_slot)`),
  - the **job table** itself (flattened to device buffers/attributes),
  - the **combine plan**: per non-empty output tile `(out_tile_id, first_slot, K)`.

The existing `build_tile_assignment` stays for the no-split fast path. Keep the
empty-tile filtering and the per-core L1 cap check (now on jobs-per-core).

### 2. Phase-1 partial op — `gaussian_alpha_blend_partial`

Derived from the existing op (kernels copied + minimally changed):

- **Reader**: per job, read the tile's `px/py` and the Gaussian sub-range
  `[gseg_start, gseg_start+gseg_count)` (offset into `packs`/`offsets`), pushing
  `gseg_count` packs. (Today's reader reads a whole tile; now a sub-range.)
- **Compute**: unchanged composite loop (already inits `T=1, color=0`); after the
  loop, additionally pack `CB_T_STATE` as a 4th output tile alongside R,G,B.
- **Writer**: write 4 tiles `(R,G,B,T)` to `partials[partial_slot*4 ..]`.
- Output tensor `partials` shape `(num_jobs*4, 1024)` bf16, DRAM page 2048 B
  (matches the existing tile page-size contract).

### 3. Phase-2 combine op — `gaussian_alpha_blend_combine`

New, small op:

- Inputs: `partials` (from phase 1), the combine plan (per-tile
  `first_slot, K`, distributed to cores).
- **Compute**: per assigned output tile, run the `over` scan above over its K
  partials (read R,G,B,T per segment in slot order), accumulating in fp32 Dst;
  pack the final R,G,B. K=1 → straight copy.
- **Writer**: write 3 tiles to `out[out_tile_id*3 ..]` (the existing output
  layout / page-size contract).
- Output-tile → core distribution: LPT by K (or round-robin; K is tiny so cost
  is dominated by the per-tile read/write, near-uniform).

### 4. Backend — `backends/tt/backend.py`

- Build the segmented schedule; if it has no splits, call the existing op
  (current code path, unchanged).
- If it has splits: allocate/reuse a `partials` DRAM tensor sized to `num_jobs`;
  upload the job table + combine plan as op attributes (hash-excluded, like the
  current schedule); enqueue partial then combine; `synchronize_device` between
  phase boundaries for honest timing (consistent with the existing sync fix).
- New sub-timings: `partial_kernel`, `combine_kernel` (the existing
  `prep/upload/kernel/download` model extends; `benchmark/phases.py` maps
  `compute = partial_kernel + combine_kernel` for the split path).

## Data flow

```
packs/offsets/px/py (device, per frame)
        │  job table (tile_id, gseg_start, gseg_count, partial_slot) per core
        ▼
phase-1 partial op  ──► partials DRAM  (num_jobs × 4 tiles: R,G,B,T)
        │  combine plan (out_tile_id, first_slot, K) per core
        ▼
phase-2 combine op  ──► out DRAM       (num_tiles × 3 tiles: R,G,B)
        ▼
ttnn.to_torch → image
```

## Error handling

- **Per-core job cap**: the reader/writer hold a core's job list in a fixed L1
  array; keep the existing `MAX_TILE_IDS_PER_CORE`-style cap (now jobs-per-core)
  and fail loud before dispatch rather than overflow L1.
- **Partials buffer**: sized to `num_jobs` at schedule time; validate its DRAM
  page size (2048 B) at dispatch, like the existing `validate_page_sizes`.
- **Empty tiles**: filtered from the schedule (no job, no combine entry);
  `out` is pre-zeroed so their slots stay background.
- **K=1**: combine is an identity copy; no special-casing beyond the loop bound.
- **Degenerate split**: a tile with `load ≤ target` yields `n_seg=1` (no split),
  so the split path is only entered when it helps.

## Testing (TDD; this order is the build order)

1. **Python combine reference** (`tests/`): composite a tile's Gaussians (a)
   monolithically and (b) as K segments merged via `over`; assert max abs
   pixel diff ≈ 0 (float). Proves the math before any kernel work.
2. **Scheduler unit tests**: `build_segmented_assignment` splits a synthetic
   heavy tile into the right number of contiguous, depth-ordered segments;
   jobs are balanced (max-core-load ≤ ~target); the combine plan reconstructs
   each tile's segments in depth order; no-split tiles pass through unchanged.
3. **Phase-1 kernel**: emits a correct 4th (T) plane and respects segment
   sub-ranges (compare a single-segment partial against the existing op's RGB +
   a known T).
4. **Phase-2 combine op**: device combine of K partials matches the Python
   combine reference.
5. **Backend integration**: two-phase TT output vs CPU reference PSNR ≥ 35 dB
   (and ≈ the current single-phase TT PSNR, ~41 dB) on `train@256`.
6. **Perf**: via the benchmark harness (`benchmark/run.py tt`) — `train@256`
   `compute` should drop from ~345ms toward the `entries/cores` ideal (ballpark
   ~10×); `luigi` unchanged (no-split path).

## Open risks

- **bf16 partials**: products of K bf16 `T` values + summed bf16 colors may erode
  PSNR. Mitigation: fp32 accumulation in the combine (Dst is fp32). Milestone 5
  validates; if PSNR drops below target, consider fp32 partials for `T`.
- **Two-dispatch overhead**: the partials DRAM round-trip adds latency; expected
  to be dwarfed by the compute saved on heavy scenes, and avoided entirely on
  no-split scenes. Quantified in milestone 6.
- **Effort**: 6 milestones spanning host scheduler + two ttnn ops; large but one
  coherent feature. The Python reference + scheduler (milestones 1–2) de-risk the
  math and scheduling before the C++/kernel work.
