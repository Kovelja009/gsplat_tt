# Intra-tile Parallelism — Phase 2 (device ops + backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the densest tile's alpha-blend work split across cores on-device, using Phase 1's tested scheduler + combine math: a `gaussian_alpha_blend_partial` op composites depth-segments into `(R,G,B,T)` partials, and a `gaussian_alpha_blend_combine` op merges them via the associative `over` operator.

**Architecture:** Two new ttnn ops mirroring the existing `gaussian_alpha_blend` op's structure (types / device_operation / program_factory / nanobind / build wiring). The existing op is untouched and stays the no-split fast path. The backend builds the segmented schedule (Phase 1) and runs partial→combine only when the schedule contains splits.

**Tech Stack:** C++ / tt-metal (ttnn op + reader/compute/writer kernels), nanobind, CMake; Python backend orchestration; pytest + the `benchmark/` harness for validation. Requires the Blackhole/Wormhole device and `sudo ninja` builds.

**Scope:** Phase 2 of the spec `docs/superpowers/specs/2026-06-08-intra-tile-parallelism-design.md` (milestones 3–6). Phase 1 (host math + `build_segmented_assignment`, in `backends/tt/segments.py`) is merged and is the contract this phase implements.

**Reality note for the implementer:** tt-metal compute/dataflow kernels are JIT-compiled and developed **iteratively** — exact SFPU/NoC op sequences are arrived at by building (`sudo ninja -C .../build metal install` or the ttnn ext build) and validating against the Phase-1 reference + PSNR, not written perfectly up front. Tasks 3 and 6 below therefore specify the **math, CB additions, arg layout, and pass/fail gate**, and direct you to copy the existing kernel as the template and iterate to green. The scaffolding tasks (1, 2, 4, 5, 7) are concrete mirrors of existing files.

---

## File Structure

New op dir (mirrors the existing `alpha_blend/`):
`backends/tt/tt-metal/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine/`
- `alpha_blend_combine.{hpp,cpp}` — op entry (mirror `alpha_blend.{hpp,cpp}`).
- `alpha_blend_combine_nanobind.{hpp,cpp}` — binding (mirror).
- `device/alpha_blend_combine_types.hpp` — `CombineParams` / `CombineInputs`.
- `device/alpha_blend_combine_device_operation.{hpp,cpp}` — device op (output = `(num_tiles*3,1024)` bf16, zero-init).
- `device/alpha_blend_combine_program_factory.{hpp,cpp}` — CB setup + kernels + per-core combine args.
- `device/kernels/{dataflow/reader_combine.cpp, dataflow/writer_combine.cpp, compute/combine.cpp}`.

Partial op: extend the **existing** `alpha_blend/` op with a partial mode rather than a third tree (smaller blast radius):
- `alpha_blend/device/kernels/.../*` — modified reader (segment ranges via job table), compute (emit T), writer (4-tile partial output).
- `alpha_blend/device/alpha_blend_types.hpp` — add `job_table` input + `partial` flag/`num_jobs`.

Host:
- `backends/tt/backend.py` — build segmented schedule; gate single-op vs partial+combine; allocate partials buffer; new sub-timings.
- `backends/tt/segments.py` — (Phase 1, done) provides the schedule.

Build wiring (per `setup.sh` step 3 / `.gitignore` tracked subtree): add the new op dir to `ttnn/CMakeLists.txt`, `ttnn/sources.cmake`, and the experimental nanobind registration — and update `setup.sh`'s injection so a fresh vendor re-applies it.

---

### Task 1: Combine op scaffolding (types + device op + entry + nanobind, no kernels yet)

**Files:** create the `alpha_blend_combine/` tree listed above (except kernels — Task 6).

Mirror the existing op exactly, changing only the I/O shape and attributes.

- [ ] **Step 1: `device/alpha_blend_combine_types.hpp`**

```cpp
#pragma once
#include <cstdint>
#include <tuple>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
namespace ttnn::operations::experimental::gaussian_splatting::alpha_blend_combine {

struct CombineParams {
    uint32_t num_tiles;                       // output tiles
    // Per-core combine schedule (hash-excluded, like AlphaBlendParams): each
    // core owns a contiguous slice of the combine plan rows.
    std::vector<uint32_t> per_core_offset;    // into the flattened plan
    std::vector<uint32_t> per_core_count;
    static constexpr auto attribute_names = std::forward_as_tuple("num_tiles");
    auto attribute_values() const { return std::forward_as_tuple(num_tiles); }
};

struct CombineInputs {
    ttnn::Tensor partials;   // (num_jobs*4, 1024) bf16 : per-job R,G,B,T tiles
    ttnn::Tensor plan;       // (num_nonempty_tiles, 4) u32 rows [out_tile, first_slot, K, pad]
};
}  // namespace
```

(Plan rows padded to 4 u32 → 16-byte page, matching the `tile_ids` 64-byte / u32-page idiom; the program factory passes per-core offset/count into `plan`.)

- [ ] **Step 2: `device/alpha_blend_combine_device_operation.{hpp,cpp}`**

Mirror `alpha_blend_device_operation.*`. `compute_output_specs` / `create_output_tensors` return `(num_tiles*3, 1024)` bf16 ROW_MAJOR DRAM, **zero-initialised** (`ttnn::zeros`) — identical to the existing op's output (so empty tiles stay background). `validate_*`: `partials` bf16 on device, `plan` u32 on device. Add the `ttnn::prim::gaussian_alpha_blend_combine(partials, plan, num_tiles, per_core_offset, per_core_count)` launcher.

- [ ] **Step 3: `alpha_blend_combine.{hpp,cpp}` + `*_nanobind.{hpp,cpp}`**

Mirror `alpha_blend.{hpp,cpp}` / `alpha_blend_nanobind.cpp`. Bind as `ttnn.experimental.gaussian_alpha_blend_combine(partials, plan, *, num_tiles, per_core_offset, per_core_count)`.

- [ ] **Step 4: Build wiring**

Add the new dir to `ttnn/CMakeLists.txt` (`add_subdirectory` + link), `ttnn/sources.cmake` (nanobind source), and the experimental nanobind registration file (include + `detail::bind_gaussian_alpha_blend_combine(mod)` call). Update `setup.sh`'s injection block so a fresh vendor re-applies these.

- [ ] **Step 5: Build (no kernels referenced yet — compile the scaffolding)**

Run: `sudo ./backends/tt/tt-metal/build_metal.sh --build-programming-examples`
Expected: compiles clean; `python -c "import ttnn; print(hasattr(ttnn.experimental,'gaussian_alpha_blend_combine'))"` → `True`.

- [ ] **Step 6: Commit**

```bash
git add backends/tt/tt-metal/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine backends/tt/tt-metal/ttnn/CMakeLists.txt backends/tt/tt-metal/ttnn/sources.cmake setup.sh
git commit -m "feat(tt): scaffold gaussian_alpha_blend_combine op (no kernels yet)"
```

---

### Task 2: Combine program factory (CBs + kernels wiring + per-core args)

**Files:**
- Create: `alpha_blend_combine/device/alpha_blend_combine_program_factory.{hpp,cpp}`

Mirror `alpha_blend_program_factory.cpp`. CBs: input `CB_PARTIAL` (depth 4, bf16 tile pages — holds one job's R,G,B,T), `CB_COLOR_OUT` (depth 3), plus fp32 accumulators `CB_C_ACC` (3) and `CB_T_ACC` (1). Validate page sizes (partials 2048 B, out 2048 B, plan 16 B). Per-core runtime args: `partials_addr, plan_addr, out_addr, plan_start, plan_count`. `ordered_cores` + `override_runtime_arguments` patch addresses + per-core plan slice on cache hits (verbatim pattern from the existing factory).

- [ ] **Step 1–5:** implement the factory mirroring the existing one; build with `sudo ./backends/tt/tt-metal/build_metal.sh --build-programming-examples`; expected: clean compile (kernels referenced by path exist as stubs from Task 6 or create empty stubs first). Commit `feat(tt): combine op program factory`.

(Detailed CB/arg code is a direct port of `alpha_blend_program_factory.cpp` lines 110–210, substituting the CB set and the 5 runtime args above. Build-verify after writing.)

---

### Task 3 (iterative): Partial op — emit T + segment ranges

**Files:**
- Modify: `alpha_blend/device/alpha_blend_types.hpp` — add `ttnn::Tensor job_table;` (`(num_jobs,4)` u32) and `uint32_t num_jobs;`; add a `partial` bool to `AlphaBlendParams` (structural → in the hash, so partial vs normal compile separately).
- Modify: `alpha_blend/device/.../reader_alpha_blend.cpp` — when `partial`, read per-job rows from `job_table` (`tile_id, gseg_start, gseg_count, partial_slot`): load `px/py[tile_id]`, push `gseg_count` packs from `packs[gseg_start..]`, push `g_count=gseg_count` meta. (No `offsets` lookup — the segment range is explicit.)
- Modify: `alpha_blend/device/.../compute/alpha_blend_compute.cpp` — after the per-tile gaussian loop, additionally `pack_tile(CB_T_STATE → CB_COLOR_OUT)` as a 4th output tile (the T plane). Gate the 4th push on a compile-time `partial` flag so the normal op still emits 3.
- Modify: `alpha_blend/device/.../writer_alpha_blend.cpp` — when `partial`, write 4 tiles to `partials[partial_slot*4 + {0,1,2,3}]` (output buffer shape `(num_jobs*4,1024)`); else the existing 3-tile path.
- Modify: device op `compute_output_specs`/`create_output_tensors` — partial mode → `(num_jobs*4,1024)`.

**Gate (how to know it's right):**

- [ ] **Step 1: Single-segment equivalence test (device)**

For a synthetic tile run as ONE segment (K=1), the partial op's R,G,B must match the existing op's R,G,B bit-for-bit, and its T plane must match a host `composite_tile(...)[1]` (from Phase 1) to bf16 tolerance.

Run a `/tmp` script: build inputs for one tile, call the existing op and the partial op (single job covering the whole tile), compare. Expected: RGB identical; `max|T_dev − T_host_bf16| < 2^-7`.

- [ ] **Step 2: Iterate the kernel edits until Step 1 passes**, rebuilding with
`sudo ./backends/tt/tt-metal/build_metal.sh --build-programming-examples` between edits.

- [ ] **Step 3: Commit** `feat(tt): partial alpha-blend — segment ranges + T-plane output`.

---

### Task 4: Backend — segmented schedule + partials buffer + gating

**Files:**
- Modify: `backends/tt/backend.py`

- [ ] **Step 1: Build the schedule and gate**

In `blend(...)`, after `prepare_kernel_inputs` and computing `offsets`, call
`from backends.tt.segments import build_segmented_assignment` →
`sched = build_segmented_assignment(offsets, num_tiles, self.num_cores)`.
If `sched.num_jobs == num_nonempty_tiles` (no splits), take the existing
single-op path unchanged. Else take the split path (Step 2).

- [ ] **Step 2: Split path**

Upload `job_table` (u32, `(num_jobs,4)` → page 16 B) and `plan` (u32,
`(num_nonempty,4)` → page 16 B) like the existing `tile_ids` upload. Allocate a
`partials` ttnn tensor `(num_jobs*4, 1024)` bf16 (reuse a scratch like
`_packs_scratch`). Call:
```python
partials = ttnn.experimental.gaussian_alpha_blend_partial(
    packs_dev, px_dev, py_dev, job_table_dev,
    image_height=H, image_width=W, num_jobs=sched.num_jobs,
    per_core_offset=[...], per_core_count=[...])
ttnn.synchronize_device(dev)                      # honest partial_kernel timing
out = ttnn.experimental.gaussian_alpha_blend_combine(
    partials, plan_dev, num_tiles=num_tiles,
    per_core_offset=[...combine...], per_core_count=[...])
ttnn.synchronize_device(dev)
```
Add sub-timings `partial_kernel`, `combine_kernel`. Readback `out` as today.

- [ ] **Step 3: Map buckets**

In `benchmark/phases.py`, extend the TT branch: `compute = kernel + partial_kernel + combine_kernel` (whichever keys are present), so the split path reports correctly. Add a unit test in `tests/test_benchmark_phases.py` for the split-path keys.

- [ ] **Step 4: Commit** `feat(tt): two-phase (partial+combine) dispatch in backend`.

---

### Task 5: Integration — correctness (PSNR)

- [ ] **Step 1: PSNR test** (`tests/test_kernel_integration.py`, new case, guarded by device presence + a `timeout=` on any subprocess): render `train`-like dense synthetic scene at 256 with the split path; assert PSNR vs the CPU `alpha_blend` reference ≥ 35 dB, and within ~1 dB of the existing single-op path. Run on the real device.

- [ ] **Step 2: If PSNR < target**, the suspect is bf16 partial `T`. Mitigate by storing the partial `T` plane fp32 (separate fp32 partials buffer) and re-test. Commit `test(tt): split-path PSNR` once green.

---

### Task 6 (iterative): Combine kernel

**Files:**
- Create: `alpha_blend_combine/device/kernels/{dataflow/reader_combine.cpp, dataflow/writer_combine.cpp, compute/combine.cpp}`

**Reader**: per assigned plan row `(out_tile, first_slot, K)`, read the K jobs'
4 tiles each from `partials[first_slot*4 .. (first_slot+K)*4]` into `CB_PARTIAL`.
**Compute** (`combine.cpp`): the `over` scan, fp32 in Dst —
```
C_acc=0; T_acc=1
for i in 0..K-1:  C_acc += T_acc * C_i ;  T_acc *= T_i   (per channel)
```
read R,G,B,T per segment from `CB_PARTIAL`; pack R,G,B of `C_acc` to `CB_COLOR_OUT`.
**Writer**: 3 tiles → `out[out_tile*3 + {0,1,2}]` (verbatim from existing writer).

**Gate:**
- [ ] **Step 1: Device combine matches Phase-1 reference.** Feed known partials
(from Task 3's partial op on a multi-segment tile), run the combine op, compare
the merged RGB to host `combine_over(...)` (Phase 1) to bf16 tolerance.
- [ ] **Step 2: Iterate kernel + rebuild until Step 1 passes.**
- [ ] **Step 3: Commit** `feat(tt): combine kernel (associative over merge)`.

---

### Task 7: End-to-end perf validation

- [ ] **Step 1: Clean + sweep** the corrected harness:
```bash
rm -f benchmark/results/tt.csv
venv/bin/python -m benchmark.run tt --scenes scenes/train.ply --res 256 480 640 960
```
- [ ] **Step 2: Compare** `compute_ms` for `train@256` to the pre-split baseline
(~345 ms). Expected: large drop toward the `entries/cores` ideal (the Phase-0
investigation measured ideal ≈ 6.7k/75k of max-tile → ballpark ~10×). Confirm
`luigi` is unchanged (no-split path) and PSNR still ≥ 35 dB.
- [ ] **Step 3: Record** the result in `docs/plan_progress.md` and commit
`docs: intra-tile parallelism perf result`.

---

## Self-Review

**Spec coverage (milestones 3–6):**
- M3 phase-1 partial op (segment ranges + T plane) → Task 3 (+ output spec). ✓
- M4 phase-2 combine op → Tasks 1, 2, 6. ✓
- M5 backend wiring → Task 4. ✓
- M6 validation (PSNR + perf) → Tasks 5, 7. ✓

**Placeholder scan:** scaffolding tasks carry concrete interface code; the two
kernel tasks (3, 6) intentionally specify math + CB/arg layout + a binary
pass/fail gate rather than fabricated SFPU sequences, per the reality note —
this is a deliberate, flagged choice for JIT-compiled device code, not a gap.

**Type consistency:** `job_table` cols `[tile_id, gseg_start, gseg_count,
partial_slot]` and `plan` cols `[out_tile, first_slot, K, pad]` match Phase 1's
`SegmentedSchedule.job_table` (4 cols) and `combine_plan` (3 cols + pad). The
combine math (`C += T_acc*C_i; T_acc *= T_i`) is exactly Phase 1's tested
`combine_over`. Op/binding names (`gaussian_alpha_blend_partial`,
`gaussian_alpha_blend_combine`) are consistent across factory, device op,
nanobind, and backend.

**Open dependency:** Task 4 (`per_core_offset/count` for the combine) needs the
Phase-1 scheduler to also expose per-core *combine-plan* slicing. Phase 1 emits
the flat `combine_plan`; the backend slices it round-robin/LPT across cores
before the op call (cheap — combine cost ∝ K). If a device-side per-core combine
balance is wanted, add it to `build_segmented_assignment` — noted, not required
for correctness.
