# Gaussian Splatting Alpha-Blend tt-metal Kernel — Plan Progress

Living document. Each section locked in as we resolve it.

---

## Research Summary (condensed)

### Hardware (Wormhole)
- **Cores**: ~72 usable Tensix cores (N150 has 1 row harvested; grid is logical 8×9).
- **L1/core**: 1.5 MB total (~1.4 MB usable after kernel code + CB metadata).
- **SFPU**: 32-lane vector; native ops include `exp`, `mul`, `add`, `recip`, `sqrt`, `min/max`, `addcmul` (fused `a + b*c`). `exp` uses polynomial approximation; fp32-accurate variant exists.
- **Tile format**: Native 32×32 elements (= 2 KB bf16 / 4 KB fp32). Composed of 4 faces of 16×16. Processed in 4 SFPU passes of 32 lanes per tile.
- **DRAM**: 12 GB, 12 banks, ~190 GB/s effective (interleaved).
- **NoC**: Dual unidirectional torus (NoC0/NoC1) for bidirectional async reads/writes.
- **Precision**: bf16 is fast path. `fp32_dest_acc_en=true` gives fp32-accurate SFPU + Dst accumulation but halves Dst tile slots (8 → 4). Use `HiFi3` (not HiFi4 on WH B0 due to bug #38306).

### Programming Model
- **3 kernels per core**: Reader (NoC→L1 via CBs), Compute (SFPU/FPU), Writer (L1→NoC→DRAM).
- **Circular buffers**: Declared on host with `CreateCircularBuffer`; kernel-side `cb_reserve_back`/`cb_push_back` (producer) and `cb_wait_front`/`cb_pop_front` (consumer). Double-buffer (depth 2) is default.
- **Compute register lifecycle**: `tile_regs_acquire` → op (via SFPU/FPU) → `tile_regs_commit`/`tile_regs_wait` → `pack_tile` → `tile_regs_release`.
- **Multi-core**: `split_work_to_cores(grid, num_units)` returns two core groups with work counts differing by ≤1. Per-core `SetRuntimeArgs` passes offsets. Runtime args limit ~16–20 values; larger metadata goes to DRAM.
- **No fused quadratic form** — must chain 3 `mul_tiles` + 2 `add_tiles` for `a·dx² + 2b·dx·dy + c·dy²`.

### Reference Implementations
- **INRIA CUDA (`diff-gaussian-rasterization`)**: one block per 16×16 tile, 256 threads (one per pixel). Cooperative shared-mem batch of 256 Gaussians. `__syncthreads_count(done)` block-wide early exit. Conic packed as 3 floats + opacity into `float4`. Thresholds: `power>0` skip, `alpha<1/255` skip, `T<1e-4` terminate.
- **WebGPU port (`cvlab-epfl/gaussian-splatting-web`)**: closest analog to tt-metal. 16×16 workgroup, 256-Gaussian batch in `var<workgroup>`. No block-wide exit (WGSL lacks the primitive). Per-lane `break`.
- **tt-metal closest template**: `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp` — per-row SFPU streaming with exp + reductions. No graphics/rasterization precedent exists in tt-metal.

### bf16 Precision Analysis
- Per-pixel power, exp, color-accumulate are tolerable in bf16.
- **Transmittance T** is the risk: multiplicative chain → 4% RMS drift over 100 Gaussians in pure bf16.
- **Recommendation**: compile with `fp32_dest_acc_en=true`, `HiFi3`. bf16 storage in L1, fp32 in Dst register for T + color_accum.
- Expected PSNR: 38–42 dB (pure bf16 would be 30–34 dB).

---

## Locked-In Decisions

### Decision 1 — Screen tile size: **32×32** (Option B)

**Why**: 1:1 map to Wormhole's native 32×32 hardware tile. One hw tile = one screen tile = 1024 pixels = 4 SFPU passes. Simpler addressing, best utilization. Only cost is a `tile_size=32` change in the CPU binning call — the pipeline code is already parametric.

**Impact**:
- `get_tile_assignments(..., tile_size=32)`
- `sort_and_bin(..., tiles_x = (W+31)//32, ...)`
- `alpha_blend(..., tile_size=32)`
- Image dimensions should be multiples of 32 for clean boundaries (pad if not; crop on host).

---

### Decision 2 — Data movement: **Host pre-gather per-tile Gaussian attributes** (Option A)

**Why**: easiest to implement, highest NoC bandwidth, easiest to debug. Converts the indirect `gaussian_attrs[sorted_ids[idx]]` pattern into pure sequential streaming on device.

**Mechanics** (refined during spec self-review):
- After `sort_and_bin` on CPU, host builds **9 SoA (Structure-of-Arrays) bf16 buffers**, one per scalar Gaussian attribute: `mean_x`, `mean_y`, `cov_inv_{a,b,c}`, `color_{r,g,b}`, `opacity`.
- Each buffer has shape `(total_entries, 32, 32)` bf16. Each 32×32 tile is a full-tile broadcast of one scalar (1024 copies of that Gaussian's attribute value). Enables plain `sub_tiles`/`mul_tiles` semantics with no scalar-intrinsic complexity.
- Host uploads 9 SoA buffers + `tile_offsets` + `px_tiles` + `py_tiles`.
- Reader kernel streams sequentially from its tile's range, pushing 9 tiles per Gaussian (one to each attribute CB).

**Why SoA-broadcast not packed-single-tile**: the earlier "one packed tile per Gaussian" design had a semantic bug — a 32×32 tile with 9 populated lanes per row doesn't broadcast for per-pixel ops. The SoA-broadcast design trades 9× memory for correctness and simple kernel code.

**Cost**: 18 KB × total_entries. For 10K Gaussians × ~10 tile memberships = 100K entries → 1.8 GB (fits 12 GB DRAM for PoC). Full scenes (100K+ Gaussians) need v1.5 optimization.

---

### Decision 3 — Scope: **Single-core PoC first** (Option A)

**Why**: correctness before performance. Debugging reader/compute/writer + CB plumbing + SFPU math chains is hard enough without also debugging parallelism. Once single-core matches CPU reference, multi-core is a host-side `split_work_to_cores` change (~1 day).

**Milestone ladder**:
1. **PoC**: core (0,0) renders a single 32×32 tile to verify SFPU math. Hardcoded inputs, small Gaussian count.
2. **Single-core full-image**: same one core loops over all tiles serially. Full PSNR validation vs CPU reference.
3. **Multi-core**: `split_work_to_cores` distributes tiles across ~72 cores. Benchmark speedup.
4. (Stretch) Kernel-level optimizations: deeper CB pipelining, SFPU op fusion, etc.

Thesis narrative benefits from this progression (CPU → single-core → multi-core speedup curve).

---

### Decision 4 — Precision: **`fp32_dest_acc_en=true` + `HiFi3` from day one** (Option A)

**Why**: one-line host flag; zero perf cost for our kernel (we only need ~4 Dst slots and the flag halves from 8 to 4); buys 8–10 dB PSNR vs pure bf16. Meets thesis target of 30–40 dB with margin.

**Host kernel config**:
```cpp
WormholeComputeKernelConfig compute_config {
    .math_fidelity = MathFidelity::HiFi3,  // NOT HiFi4 — WH B0 bug #38306
    .math_approx_mode = false,             // use accurate exp()
    .fp32_dest_acc_en = true,              // fp32 SFPU + Dst accumulation
    .packer_l1_acc = false,
};
```

**Data formats**:
- DRAM / L1 (CBs): `bfloat16` for all inputs (Gaussian attributes, output tiles).
- Dst registers: fp32 (implicit via `fp32_dest_acc_en`).
- Running state (`T`, `color_accum`) stays in Dst across Gaussian iterations; `pack_tile` converts back to bf16 when writing the output tile.

Expected PSNR vs CPU fp32: **38–42 dB**. Indistinguishable after 8-bit output quantization.

---

### Decision 5 — Culling: **Block-wide `T<1e-4` exit only** (Option A)

**Why**: the block-wide transmittance exit changes asymptotic cost (30–80 effective Gaussians blended vs a 500-Gaussian sorted list — ~6× savings). Per-lane `power>0` and `alpha<1/255` skips are cheap to add later but don't help on vectorized SFPU (all 32 lanes execute every op regardless).

**Implementation sketch (v1 day-one)**:
- Track `T` for all 1024 pixels in Dst (32 tile-rows × 32 cols, fp32 via `fp32_dest_acc_en`).
- After each Gaussian, take `T_max = max_reduce(T)`; if `T_max < 1e-4`, break outer Gaussian loop for this tile.
- Reduction uses `reduce_tile<PoolType::MAX>` + a scalar compare.

**Deferred to v2 (optimization phase)**:
- `power > 0` skip (outside ellipse).
- `alpha < 1/255` skip (sub-visible contribution).

Note: correctness is equivalent between v1 and v2 — just performance.

---

### Decision 6 — Output layout: **Tiled bf16 on device + device-side `ttnn.untilize` → host** (Option A)

**Why**: canonical tt-metal pattern. Compute kernel writes natural 32×32 bf16 tiles; a trivial `untilize` pass produces row-major; host does bf16→uint8 conversion (one-liner). All example ops in tt-metal work this way; easier debugging. <1 ms overhead for typical image sizes.

**Shape**:
- Device output buffer: `(tiles_y * tiles_x, 32, 32, 3)` bf16 in tile layout — or equivalently `(H, W, 3)` bf16 padded to 32 multiples.
- After untilize: `(H_padded, W_padded, 3)` bf16 row-major.
- Host: crop to `(H, W, 3)`, convert bf16 → uint8 with clamp/scale.

---

### Decision 7 — Build integration: **Develop inside `tt-metal/tt_metal/programming_examples/gaussian_splatting/`** with `.gitignore` exception

**Why**: zero build setup (tt-metal's CMake auto-picks up new subdirs), matches idiomatic tt-metal layout, all env vars/headers/linkage already configured, easy to copy patterns from neighboring examples (`eltwise_sfpu`, `matmul_multi_core`).

**Git strategy**: since the vendored `tt-metal/` is untracked, add a narrow `.gitignore` exception:
```gitignore
tt-metal/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/**
```
Only our kernel subdir is tracked; the rest of the vendored repo stays untracked.

**File layout**:
```
tt-metal/tt_metal/programming_examples/gaussian_splatting/
├── CMakeLists.txt
├── alpha_blend.cpp                              (host driver)
└── kernels/
    ├── dataflow/
    │   ├── reader_alpha_blend.cpp
    │   └── writer_alpha_blend.cpp
    └── compute/
        └── alpha_blend_compute.cpp
```

**Python integration**: defer pybind11 until after PoC. For validation phase, use a standalone C++ binary that reads/writes `.npy` files dumped from Python.

---

## All Decisions Locked — Spec Writing Next

All seven design tensions resolved. Next step: write full design spec to `docs/superpowers/specs/` and get user review before transitioning to implementation plan.

---

## Spec Revision 2 — Post-Feasibility Review (2026-04-18)

After writing Rev 1 of the spec, a 10-agent parallel feasibility review was conducted against the tt-metal repo. Found 6 critical issues requiring spec revision. Rev 2 of the spec addresses all of them.

### Findings Summary

**🔴 Critical (all fixed in Rev 2)**:

1. **`_to_cb` suffix ops don't exist**: `sub_tiles_to_cb`, `mul_tiles_to_cb`, `add_tiles_to_cb` are not real APIs. Must use `acquire → sub_tiles(cb_a, cb_b, ia, ib, dst) → pack_tile → release`.
2. **Dst can't hold long-lived accumulators across many iterations**: tt-metal's math↔packer double-buffer semaphore requires frequent `tile_regs_commit`. Kernel must cycle acquire/commit/release per Gaussian; running state spills to L1 CBs.
3. **Dst-to-Dst binary ops not supported**: all binary ops require CB operands. Running-state accumulators live in L1 CBs (`cb_color_R_state`, `cb_T_state`, etc.) and reload via `copy_tile`.
4. **Early termination via scalar-from-Dst infeasible**: no API extracts a scalar from Dst to RISC-V. Replaced with **sentinel-mask approach**: every 16 Gaussians compute `sat_mask = (T ≥ 1e-4)`, multiply subsequent `contrib` by mask. Saturated pixels stop accumulating without break.
5. **9-SoA full-tile-broadcast wildly wasteful**: `mul_unary_tile` / `sub_unary_tile` / `add_unary_tile` exist and take scalars directly. Replaced with flat `(N, 9)` fp32 scalar array — **500× DRAM reduction** (1.8 GB → 3.6 MB for 100K entries). Unlocks full scenes without v1.5 rework.
6. **9-CB wait-on-all deadlock risk**: auto-resolved by fix #5 (single scalar CB per Gaussian).

**🟡 Significant (all addressed in Rev 2)**:

- `T *= (1-alpha)` round-trips through bf16 L1 — accepted as dominant precision loss; still within 35-42 dB target.
- Missing/wrong op names fixed: `zero_tile` → `fill_tile(0.0f)`; `min_tile`/`max_tile` → `binary_min_tile`/`binary_max_tile`.
- Op count revised from ~15 to ~22-25 per Gaussian; perf projection updated: ~600 ms single-core at 640×640, ~40 ms multi-core target.
- v1 decomposed into v1a (1 Gaussian, 1 tile) → v1b (full scene, single core) → v1c (multi-core). Multi-core target softened to ≥15× floor / 20× stretch.
- Tier 0 smoke tests added (pass-through, byte-count, non-NaN, single-Gaussian identity).
- Numeric sanity notebook added to Phase 0.

**🟢 Validated (no changes)**:
- `fp32_dest_acc_en=true + HiFi3 + math_approx_mode=false` is correct.
- Runtime args limit is 341 per core (not 16-20 as initially believed).
- MeshDevice + EnqueueMeshWorkload + TensorAccessorArgs are idiomatic.
- CMakeLists.txt can be minimal (8 lines).
- Output buffer `(N, 3, 32, 32)` works with `ttnn.untilize` (with explicit writer tile-ID math).
- CB L1 budget (~45 KB after revision) is far under 1.4 MB available.

### Revision Scope
- §2: updated Decision 2 and Decision 5 to reflect scalar-unary approach and sentinel mask.
- §4: diagram updated (scalar CB, not 9 attribute CBs).
- §5.1: rewritten — single flat fp32 `(N, 9)` array instead of 9 SoA tile-broadcast buffers.
- §5.5: memory budget updated (~8 MB total, vs 1.8 GB).
- §6.1: reader kernel simplified to 4 CBs (cb_px, cb_py, cb_scalars, cb_tile_meta).
- §6.2: full rewrite — state CBs, scalar-unary ops, sentinel mask, correct op names, honest op count, acquire/commit cycles.
- §9.0: new Tier 0 smoke tests section.
- §9.1: added numeric sanity notebook.
- §9.5: decomposed into v1a/b/c with tiered success criteria.
- §10: risk register expanded from 7 to 14 risks.
- §12: next steps restructured as Phase 0-5 with concrete Phase 0 device-sanity tasks.

---

## Spec Revision 3 — Second Feasibility Review (2026-04-18)

10-agent parallel review of Rev 2 against tt-metal. Verdict: **READY WITH MINOR EDITS** — no design rework needed, 10 tactical edits applied.

### Findings & Applied Fixes

**Validated (no change):**
- `mul_unary_tile`, `sub_unary_tile`, `add_unary_tile` exist with claimed signatures. `ge_binary_tile` exists.
- Scalar CB with 64 B pages and `get_read_ptr` + `float*` cast pattern is idiomatic.
- acquire/commit/release per Gaussian (5–7 blocks) is low-overhead and matches production kernels (layernorm/softmax).
- CB L1 budget (~47 KB) far under 1.4 MB available.
- Tier 0 → Tier 3 ordering is correct.

**Applied Rev 3 optimizations:**

1. **`rsub_unary_tile` exists** → eliminated `cb_const_ones`. Stage E uses `copy_tile(cb_alpha) + rsub_unary_tile(0, 1.0f)`. Saves 2 KB L1.
2. **`unary_ge_tile` exists** → eliminated `cb_const_1e-4`. Stage F uses `copy_tile(cb_T_state) + unary_ge_tile(0, 1e-4f)`. Saves 2 KB L1.
3. **`addcmul_tile` fuses Stage D** across all 3 channels in one acquire block (4 Dst slots: R/G/B state + contrib). Saves ~50% of Stage D lifecycle overhead. Op count down from 25 → 20 per Gaussian.
4. **Perf projection corrected**: `exp_tile` is ~800 cycles (polynomial), not 128. New estimate: ~4500 cycles/Gaussian, ~700 ms single-core at 640×640, ~50–80 ms multi-core. Still 3–10× single-core / 30–100× multi-core speedup vs Python CPU baseline. Thesis narrative holds.
5. **§6.2 pseudocode cleanup**: removed leftover "wait — dx is a tile" editing note, fixed CB name mismatches, explicit `copy_tile + binary_min_tile` flow for clamps, pinned `fill_tile` semantics. Ops list tightened with full signatures + unused ops removed (`reduce_tile` not needed).
6. **Documentation gaps filled**:
   - `mean_x/y` and `px/py` are both in **global screen coordinates** (§5.1, §5.3).
   - `two_cov_b = 2 * cov_b` **precomputed on host** in the pack (§5.1).
   - `px_tiles`/`py_tiles` uploaded **once per resolution change**, reused across frames (§5.3).
   - Stage F sat_mask refreshes at **top of loop before Stage D**, skipping g=0 (§6.2 Stage F).
7. **§10 cross-reference fixed**: "§8 precision analysis" → "§3 precision analysis".
8. **Untilize fallback designed** (§5.4): 3 separate single-channel output buffers if `ttnn.untilize` rejects the `(N, 3, 32, 32)` layout. 5-line writer kernel delta.
9. **Tier 0 expanded** (§9.0): added T0.0 (hello tt-metal), T0.5 (two-Gaussian blend), T0.6 (saturation/sat_mask). Pinned exact params for T0.4 single-pixel identity test. Named CB push/pop counter mechanism (L1 scratch + `read_hex_vec_from_core`).
10. **Phase 0-5 tightened** (§12): swapped order of tasks 1 and 2 (notebook before build — parallelizable); split task 9 into 9a/9b/9c; explicit C++ test harness task added; DeviceProfiler integration added; pybind11 deferred to post-thesis; benchmark scope narrowed to 2 scenes × 1 resolution × 3 backends + analysis chapter. **v1b-is-contract, v1c-is-stretch** called out explicitly.

**Deferred to v2 (tracked as stretch):**
- **Reader-signal block-wide early termination** (save ~80% of cycles on saturated-tile-dominated scenes). Feasible via `noc_semaphore_inc` from compute to reader. Better ROI than per-lane skips — promoted ahead of them in the v2 optimization list.

### Post-Rev 3 Status
- **14 risks** in §10, all with concrete mitigations (no handwaves).
- **~20 SFPU ops per Gaussian** (down from ~25 in Rev 2).
- **~700 ms single-core / ~50-80 ms multi-core** projected for 640×640.
- **L1 budget ~47 KB** (vs 1.4 MB available).
- **Total memory footprint**: ~8 MB for PoC scene, ~40 MB for full 100K-Gaussian scene. No v1.5 optimization needed.
- All ops exist and signatures validated against tt-metal headers.

### Ready for Implementation Plan
Spec Rev 3 is approved for transition to `writing-plans` skill.

---

## Next Steps (after all decisions locked)

1. Write full design spec to `docs/superpowers/specs/YYYY-MM-DD-alpha-blend-kernel-design.md`
2. User reviews and approves spec
3. Transition to writing-plans skill for implementation plan
4. Implement kernel in `tt_metal_kernels/alpha_blend/` (dir TBD)
