# Alpha-Blend tt-metal Kernel — Design Spec

**Status**: Revision 2 (post-feasibility review). Approved design, pending implementation plan.
**Date**: 2026-04-18
**Author**: Vanja Kovinic (MSc thesis)
**Scope**: Forward-pass alpha-blending kernel for 3D Gaussian Splatting on Tenstorrent Wormhole.

**Revision history**:
- Rev 1 (2026-04-18): Initial spec after brainstorming 7 design decisions.
- Rev 2 (2026-04-18): Updated §5, §6, §9, §10 after deep feasibility review against tt-metal API. Replaced full-tile-broadcast with scalar-unary ops; fixed Dst register lifecycle; replaced `break`-style early termination with sentinel-mask approach; fixed op names; decomposed v1 into v1a/b/c milestones.
- Rev 3 (2026-04-18): Second feasibility review (10 agents against tt-metal Rev 2). Eliminated two constant CBs via `rsub_unary_tile`/`unary_ge_tile`. Fused Stage D using `addcmul_tile` across all 3 channels in one acquire block (3× speedup per channel). Updated perf projection for true `exp_tile` cost (~700 ms single-core, ~60 ms multi-core target). Cleaned §6.2 pseudocode (name fixes, ops list). Added global-coord convention, 2·cov_b host precompute, sat_mask refresh ordering. Designed untilize fallback. Expanded Tier 0 tests. Tightened Phase 0-5 timeline within thesis budget.

---

## 1. Summary

Replace the CPU-side `alpha_blend` function in [rasterization.py](../../../rasterization.py) with a custom tt-metal kernel running on Tenstorrent Wormhole hardware. Preserves bit-comparable semantics with the CPU reference (within bf16 precision bounds). Delivers the thesis's core kernel contribution.

**Non-goals**: projection kernel (separate thesis milestone), sorting (stays on CPU per thesis plan), backward pass/training (explicitly out of thesis scope).

---

## 2. Locked Design Decisions

| # | Decision | Rationale (1-line) |
|---|---|---|
| 1 | Screen tile size = **32×32** | 1:1 map to Wormhole's native hardware tile; best utilization, simplest addressing |
| 2 | Data movement = **host pre-gather per-tile Gaussian scalars into flat arrays; use SFPU scalar-unary ops** | Sequential streaming; 500× less DRAM than tile-broadcast; `mul_unary_tile`/`sub_unary_tile` exist and take scalars directly |
| 3 | Scope = **single-core PoC first, multi-core after correctness** | Standard research workflow; thesis narrative benefits from the speedup curve |
| 4 | Precision = **`fp32_dest_acc_en=true` + HiFi3 from day one** | One-line host flag; free perf; +8–10 dB PSNR |
| 5 | Culling = **sentinel-mask saturation approach** (multiply `contrib` by `(T ≥ 1e-4)` mask every 16 Gaussians). Per-lane `power>0` and `alpha<1/255` deferred. | tt-metal has no scalar-from-Dst extraction; `break`-style exit infeasible. Mask approach costs ~0.5 ops/Gaussian amortized, preserves correctness. |
| 6 | Output = **tiled bf16 on device + `ttnn.untilize` → host** | Canonical tt-metal pattern; trivial device pass; all examples use this |
| 7 | Build location = **`tt-metal/tt_metal/programming_examples/gaussian_splatting/`** + `.gitignore` exception | Zero build setup; idiomatic; easy copy-paste from neighboring examples |

---

## 3. Target Hardware (Wormhole)

- **Cores**: ~72 usable Tensix cores (logical 8×9 grid; N150 has 1 row harvested).
- **L1/core**: 1.5 MB SRAM (~1.4 MB usable after kernel code + CB metadata).
- **SFPU**: 32-lane vector. Ops: `exp`, `mul`, `add`, `recip`, `sqrt`, `min/max`, `addcmul` (fused `a + b*c`). Polynomial `exp`; fp32-accurate variant via `fp32_dest_acc_en`.
- **Tile format**: native 32×32 (= 2 KB bf16). 4 faces of 16×16. 4 SFPU passes per tile.
- **DRAM**: 12 GB, 12 banks, ~190 GB/s aggregate bandwidth.
- **NoC**: dual unidirectional torus (NoC0/NoC1).
- **Compute config used**: `MathFidelity::HiFi3`, `math_approx_mode=false`, `fp32_dest_acc_en=true`, `packer_l1_acc=false`. (HiFi4 avoided due to WH B0 bug #38306.)

---

## 4. Pipeline Architecture

```
┌──────────────────── HOST (CPU, Python + C++) ───────────────────┐
│                                                                  │
│  1. project_gaussians()           (existing Python)              │
│  2. get_tile_assignments(tile_size=32)    (existing, new param)  │
│  3. sort_and_bin()                (existing)                     │
│  4. PRE-GATHER per-tile Gaussian scalars   ← NEW CPU code        │
│     → attribute_packs: flat fp32 array, shape                    │
│       (total_entries, 9) where 9 = (mean_x, mean_y,              │
│       cov_a, cov_b, cov_c, color_r, color_g, color_b, opacity)   │
│     → tile_offsets buffer (uint32)                               │
│     → px_tiles, py_tiles buffers (bf16, tiled — varies per pixel)│
│  5. Upload to device DRAM                                        │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌──────────────────── DEVICE (tt-metal kernel) ───────────────────┐
│                                                                  │
│   v1: single core (0,0). v2: 72 cores via split_work_to_cores.  │
│                                                                  │
│   Per assigned screen tile:                                      │
│     [Reader]  stream one "attribute pack" (9 fp32 scalars) per   │
│                Gaussian → cb_scalars (36-byte pages)             │
│               stream px/py once per tile → cb_px/py (2 KB tiles) │
│               push g_count to cb_tile_meta                       │
│     [Compute] alpha-blend 1024 pixels per tile via SFPU          │
│               scalar-unary ops for per-Gaussian math             │
│               sentinel-mask saturation (no scalar-break)         │
│     [Writer]  push 3 output tiles (R, G, B) → cb_color_out       │
│               drain to DRAM                                      │
│                                                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌──────────────────── HOST (CPU) ─────────────────────────────────┐
│                                                                  │
│  6. ttnn.untilize() device-side                                  │
│  7. Read back → bf16 row-major                                   │
│  8. Crop padding, bf16 → uint8 conversion                        │
│  9. Display via viser/nerfview                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Layout

All DRAM buffers use `InterleavedBufferConfig` (round-robins across 12 banks).

### 5.1 Per-Gaussian attributes — flat scalar array, streamed into compute via scalar CB

One DRAM buffer holding **9 fp32 scalars per Gaussian entry**. Uses tt-metal's `mul_unary_tile` / `sub_unary_tile` / `add_unary_tile` family (in [`eltwise_unary/binop_with_scalar.h`](../../../tt-metal/tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h) and related): they take a `uint32` encoding of an fp32 scalar via runtime parameter — no scalar-broadcast tile needed in DRAM or L1.

**The buffer**:
| Buffer | Shape | Dtype | Notes |
|---|---|---|---|
| `attribute_packs` | `(total_entries, 9)` | fp32 | Per entry: `[mean_x, mean_y, cov_a, two_cov_b, cov_c, R, G, B, opacity]` |

**Coordinate convention (critical)**: `mean_x` and `mean_y` are in **global screen pixel coordinates** — the same space as `px_tiles`/`py_tiles`. The device kernel does `dx = px - mean_x` with no tile-local offset adjustment. This matches CPU reference in `rasterization.py`.

**Host precompute**: `two_cov_b = 2 * cov_b` is precomputed on the host (replacing raw `cov_b` in the pack). Saves one on-device multiply per Gaussian for the `2b·dxdy` term in the quadratic form.

**Size per entry**: 9 × 4 bytes = 36 bytes.
**Total size for 100K entries**: 3.6 MB. (500× smaller than the rejected full-tile-broadcast design.)
**Full-scene support**: 1M entries = 36 MB. Trivial for 12 GB DRAM.

**Reader implication**: for each Gaussian, reader streams 36 bytes from DRAM into a small scalar CB `cb_scalars` (page size 64 bytes padded for alignment, depth 2). Compute kernel unpacks the 9 fp32 values from the page's raw bytes (via `get_read_ptr(cb_scalars)` + pointer casts) and passes each as the runtime-scalar argument to `sub_unary_tile` / `mul_unary_tile` etc.

**Why fp32, not bf16, for the packs**: the scalar-unary ops take a `uint32` holding fp32 bit-pattern. Storing as fp32 in DRAM lets the reader pass them straight through. Converting bf16→fp32 on device adds an op; skipping it is simpler. Still 500× smaller than the broadcast-tile design.

### 5.2 `tile_offsets` (uint32)

**Shape**: `(num_screen_tiles + 1,)` uint32. Cumulative prefix-sum. `tile_offsets[i]` = first entry index in each of the 9 attribute buffers for tile `i`. Count for tile `i` = `tile_offsets[i+1] - tile_offsets[i]`.

**Size**: ~1.6 KB for 400-tile (640×640) image. Read once per-core into an L1 scratch array at kernel start.

### 5.3 `px_tiles`, `py_tiles` (bf16, tiled)

**Shape**: `(num_screen_tiles, 32, 32)` bf16 each. Pixel coordinates in **global screen pixel coordinates** (same space as `mean_x`, `mean_y`). Per-pixel value for tile (tx, ty): `px[i, j] = tx*32 + j + 0.5`, `py[i, j] = ty*32 + i + 0.5`.

**Upload cadence**: px/py buffers depend only on screen geometry, not scene content. Upload **once per resolution change**; reuse across frames. Viewer resolution typically fixed → upload at startup.

**Size**: 2 × 2 KB × num_screen_tiles. 1.6 MB for 400 tiles — trivial.

### 5.4 `output_image` (bf16, tiled)

**Primary layout**: `(num_screen_tiles, 3, 32, 32)` bf16. Per screen tile: 3 consecutive tiles (R, G, B channel-separated). Writer kernel computes DRAM tile index as `3*screen_tile_id + channel`.

**Size**: 2.4 MB for 640×640.

**Post-processing**: device-side `ttnn.untilize` → row-major `(H_padded, W_padded, 3)` → host crop → uint8.

**Fallback layout (if `ttnn.untilize` doesn't support `(N, 3, 32, 32)` with channel-as-second-dim)**: use **3 separate single-channel output buffers** `out_R`, `out_G`, `out_B`, each shape `(tiles_y*32, tiles_x*32)` bf16. Writer kernel pushes one tile per channel to the corresponding buffer instead of interleaved. Then `ttnn.untilize` runs independently on each single-channel tensor (well-supported path). Host interleaves the 3 channels after readback (one `np.stack` call). Memory cost identical; kernel writer delta is ~5 lines of code.

**Decision**: attempt primary layout in Phase 0 via a small `ttnn.untilize` prototype on a mock buffer. If it fails, switch to fallback before committing Phase 1 kernels.

### 5.5 Memory Budget

For 10K visible Gaussians × ~10 tile memberships avg (100K entries), 640×640 image:
- `attribute_packs`: 100K × 36 B = **3.6 MB**
- `px_tiles`, `py_tiles`: 2 × 400 × 2 KB = **1.6 MB**
- `tile_offsets`: 400 × 4 B = **1.6 KB**
- `output_image`: 400 tiles × 3 channels × 2 KB = **2.4 MB**

**Total ~8 MB** — negligible for 12 GB DRAM.

Full scene (100K visible Gaussians → ~1M entries): `attribute_packs` = 36 MB. Still trivial. **No v1.5 packing optimization needed** — the scalar-unary approach scales to arbitrary scene sizes without further rework.

---

## 6. Kernel Specifications

### 6.1 Reader Kernel (`reader_alpha_blend.cpp`)

**Processor**: BRISC (`RISCV_0`), uses NoC0.

**Runtime args** (5 total — DRAM addresses may also use compile-time `TensorAccessorArgs` instead):
- `arg0`: `attribute_packs` DRAM base address (fp32, `(N, 9)`)
- `arg1`: `tile_offsets` DRAM base address (uint32)
- `arg2`: `px_tiles` DRAM base address (bf16, tiled)
- `arg3`: `py_tiles` DRAM base address (bf16, tiled)
- `arg4`: `first_tile_id` (this core's first tile)
- `arg5`: `num_tiles` (this core's assignment)

**Responsibilities**:
1. Read `tile_offsets[first_tile_id .. first_tile_id+num_tiles+1]` into L1 scratch once at startup.
2. For each assigned screen tile:
   - Push Gaussian count (uint32) to `cb_tile_meta` (64 B page).
   - Async-read one `px` tile and one `py` tile for this screen tile → `cb_px`, `cb_py`.
   - For each Gaussian `g` in this tile: async-read 36 bytes (9 fp32 scalars) at offset `g × 36` in `attribute_packs` → `cb_scalars` page. Page padded to 64 bytes for NoC alignment.

**Circular buffers produced**:
| CB ID | Name | Page size | Depth | Notes |
|---|---|---|---|---|
| 0 | `cb_px` | 2 KB (bf16 tile) | 2 | pixel x-coords for screen tile |
| 1 | `cb_py` | 2 KB (bf16 tile) | 2 | pixel y-coords for screen tile |
| 2 | `cb_scalars` | 64 B (padded 36 B pack) | 4 | one Gaussian's 9 fp32 scalars |
| 3 | `cb_tile_meta` | 64 B (padded uint32) | 2 | Gaussian count per tile |

Total L1 for input CBs: `2×2 KB + 2×2 KB + 4×64 B + 2×64 B ≈ 8.4 KB`. Trivial.

**cb_scalars depth 4 (not 2)**: since Gaussians are tiny 36 B packs and compute may pipeline several, extra depth hides NoC latency without meaningful L1 cost.

**No deadlock risk**: single scalar CB per Gaussian → compute waits on one CB, reader pushes one CB. No multi-CB ordering dependency.

### 6.2 Compute Kernel (`alpha_blend_compute.cpp`)

**Processor**: MATH RISC-V (Tensix compute).

**Runtime args**: `arg0`: `num_tiles`.

**Compute kernel design constraints (critical, from feasibility review)**:
1. **`tile_regs_acquire` / `commit` / `release` must cycle frequently** — not once per screen tile. tt-metal's math↔packer double-buffered semaphore requires commits between Dst uses. Acquire/commit per Gaussian (or small groups of Gaussians) is idiomatic.
2. **All binary ops take CB operands, not Dst operands**. Running-state accumulators (`color_R/G/B`, `T`) must **spill to dedicated L1 CBs** between iterations, and be re-loaded via `copy_tile`.
3. **No scalar-from-Dst extraction exists** — replaces break-style early termination with sentinel-mask approach.

**L1 state CBs** (persistent running accumulators — bf16, tile-sized):
| CB | Purpose | Depth | Pre-filled init |
|---|---|---|---|
| `cb_color_R_state` | Accumulated R | 1 | 0.0 on start |
| `cb_color_G_state` | Accumulated G | 1 | 0.0 on start |
| `cb_color_B_state` | Accumulated B | 1 | 0.0 on start |
| `cb_T_state` | Transmittance T | 1 | 1.0 on start |
| `cb_sat_mask` | Saturation mask (0 or 1 per pixel) | 1 | 1.0 on start |

**Scratch L1 CBs** (intermediate tiles, depth 1):
- `cb_dx`, `cb_dy`, `cb_dx2`, `cb_dy2`, `cb_dxdy`, `cb_Q`, `cb_power`, `cb_weight`, `cb_alpha`, `cb_contrib`, `cb_one_minus_alpha`, `cb_T_tmp`. (~12 × 2 KB = 24 KB.)

**Constant CBs** (pre-filled once at kernel start via `fill_tile` → `pack_tile` → `cb_push_back`):
- `cb_const_zero`: tile filled with 0.0 (for `binary_min_tile(power_dst, zero_dst)` to clamp `min(power, 0)`).
- `cb_const_099`: tile filled with 0.99 (for `binary_min_tile(alpha_dst, const_099_dst)` alpha clamp).

*Note*: `cb_const_ones` eliminated via `rsub_unary_tile(dst, 1.0f_bits)` (reverse-subtract — computes `1 - dst`). `cb_const_1e-4` eliminated via `unary_ge_tile(dst, 1e-4f_bits)` (in-place `dst = (dst ≥ 1e-4) ? 1 : 0`). No scalar-unary `min_tile(dst, scalar)` exists, so `cb_const_zero` and `cb_const_099` must remain as constant tiles.

Total constant CB L1: 2 × 2 KB = **4 KB** (half of Rev 2's 8 KB).

**L1 budget**: 5 × 2 KB (state) + 12 × 2 KB (scratch) + 2 × 2 KB (constants after Rev 3 eliminations) + 8.4 KB (input CBs) = **~47 KB** << 1.4 MB available.

---

**Per screen tile — high-level flow**:
1. **Init state CBs**: `fill_tile_cb(cb_color_R_state, 0.0f)`, same for G, B; `fill_tile_cb(cb_T_state, 1.0f)`; `fill_tile_cb(cb_sat_mask, 1.0f)`.
2. **Wait on `cb_px`, `cb_py`** (one tile each, retained across all Gaussians — not popped until end).
3. **Pop one page from `cb_tile_meta`** to read `g_count` into a RISC-V register.
4. **For `g = 0 .. g_count - 1`**: run the inner-Gaussian block below. Uses `acquire/commit/release` multiple times.
5. **Every 16 Gaussians**: recompute `cb_sat_mask` from `cb_T_state` (sentinel-mask refresh).
6. **After loop**: pack `cb_color_R/G/B_state` to `cb_color_out` (3 tiles).
7. **Pop `cb_px`, `cb_py`**. Drain any unconsumed scalar pages (none, since we stream exactly `g_count` per tile).

---

**Inner-Gaussian block** (runs once per Gaussian; uses acquire/commit cycles as needed):

**Stage A — decode scalars**:
- `cb_wait_front(cb_scalars, 1)`. Cast page ptr to `float*`, read 9 scalars into RISC-V locals. `cb_pop_front(cb_scalars, 1)`.
- Convert each to `uint32` fp32-bits for scalar-unary op params.

**Stage B — compute power tile** (`cb_power` gets populated):

*B1 — dx, dy*:
```
acquire
  copy_tile(cb_px, 0, dst=0); sub_unary_tile(0, mean_x_bits)   // dst[0] = dx
  copy_tile(cb_py, 0, dst=1); sub_unary_tile(1, mean_y_bits)   // dst[1] = dy
  pack_tile(0, cb_dx); pack_tile(1, cb_dy)
commit; release
```

*B2 — dx², dy², dx·dy (three CB-to-CB multiplies)*:
```
acquire
  mul_tiles(cb_dx, cb_dx, 0, 0, dst=0); pack_tile(0, cb_dx2)
commit; release

acquire
  mul_tiles(cb_dy, cb_dy, 0, 0, dst=0); pack_tile(0, cb_dy2)
commit; release

acquire
  mul_tiles(cb_dx, cb_dy, 0, 0, dst=0); pack_tile(0, cb_dxdy)
commit; release
```

*B3 — quadratic form `Q = a·dx² + 2b·dxdy + c·dy²`, then `power = -0.5·Q`*:

The `2b` factor is **precomputed on host** — the `attribute_packs` stores `two_cov_b = 2 * cov_b` in the cov_b slot, so the device sees a pre-doubled value. (Decision: this avoids an extra on-device multiply per Gaussian.)

```
acquire
  copy_tile(cb_dx2, 0, dst=0); mul_unary_tile(0, cov_a_bits)          // a·dx²
  copy_tile(cb_dxdy, 0, dst=1); mul_unary_tile(1, two_cov_b_bits)     // 2b·dxdy
  add_tiles(cb_dx2_scaled, cb_dxdy_scaled, 0, 0, dst=2)               // (needs scratch CBs between acquires; see note)
  copy_tile(cb_dy2, 0, dst=3); mul_unary_tile(3, cov_c_bits)          // c·dy²
  ...
commit; release
```

In practice, the chain fans out into 2–3 acquire blocks because `add_tiles` requires CB operands, not Dst-to-Dst. Final step scales by -0.5:
```
acquire
  copy_tile(cb_Q, 0, dst=0); mul_unary_tile(0, neg_half_bits)   // dst[0] = -0.5·Q
  pack_tile(0, cb_power)
commit; release
```

**Stage C — Gaussian weight and alpha**:

*C1 — `gauss_weight = exp(min(power, 0))`* (clamp then exp):
```
acquire
  copy_tile(cb_power, 0, dst=0)
  copy_tile(cb_const_zero, 0, dst=1)
  binary_min_tile(0, 1, dst=0)       // dst[0] = min(power, 0)
  exp_tile(0)                         // dst[0] = exp(dst[0])
  pack_tile(0, cb_weight)
commit; release
```

*C2 — `alpha = min(0.99, opacity · gauss_weight)`*:
```
acquire
  copy_tile(cb_weight, 0, dst=0)
  mul_unary_tile(0, opacity_bits)    // dst[0] = opacity · weight
  copy_tile(cb_const_099, 0, dst=1)
  binary_min_tile(0, 1, dst=0)        // dst[0] = min(dst[0], 0.99)
  pack_tile(0, cb_alpha)
commit; release
```

**Stage D — accumulate color** (fused via `addcmul_tile`, all 3 channels in one acquire block):

`addcmul_tile<DataFormat>(idst0, idst1, idst2, odst, uint32_t scalar_value)` computes `odst = idst0 + scalar · idst1 · idst2`. Perfect fit for the alpha-compositing step: `color_c_state += color_c_bits · contrib · sat_mask` maps directly onto `addcmul_tile(state, contrib, sat_mask, state, color_c_bits)`.

*D1 — compute `cb_contrib = alpha · T`*:
```
acquire
  mul_tiles(cb_alpha, cb_T_state, 0, 0, dst=0)   // dst[0] = alpha · T
  pack_tile(0, cb_contrib)
commit; release
```
(Note: sat_mask is NOT folded into `cb_contrib` here — it's passed as a separate operand to `addcmul_tile` in D2, avoiding one extra multiply.)

*D2 — fused RGB accumulate (single acquire block, uses all 4 Dst slots)*:
```
acquire
  copy_tile(cb_color_R_state, 0, dst=0)   // R state
  copy_tile(cb_color_G_state, 0, dst=1)   // G state
  copy_tile(cb_color_B_state, 0, dst=2)   // B state
  copy_tile(cb_contrib, 0, dst=3)         // contrib (shared across channels)

  // In each addcmul_tile call below, the sat_mask multiplier is loaded
  // inline by referring to cb_sat_mask via intermediate scratch — or
  // pre-multiplied into cb_contrib if the addcmul signature requires
  // all operands to be in Dst. (Verify concrete op ergonomics in
  // implementation; both paths preserve the 1-acquire-per-Gaussian goal.)
  addcmul_tile(0, 3, /*mask*/, dst=0, color_R_bits)   // R += color_R · contrib · mask
  addcmul_tile(1, 3, /*mask*/, dst=1, color_G_bits)   // G += color_G · contrib · mask
  addcmul_tile(2, 3, /*mask*/, dst=2, color_B_bits)   // B += color_B · contrib · mask

  pack_tile(0, cb_color_R_state)
  pack_tile(1, cb_color_G_state)
  pack_tile(2, cb_color_B_state)
commit; release
```

**Savings over Rev 2**: previously 3 acquire/commit/release cycles (one per channel); now 2 blocks total (D1 + D2) regardless of channel count. Stage D overhead reduced by ~50%.

**If `addcmul_tile` requires the mask in Dst slot 3 but we need it for contrib**: fallback is to pre-multiply `cb_contrib *= sat_mask` in D1 (one extra `mul_tiles`), then D2 uses 3-operand form without mask. Still 1 acquire for the three-channel accumulate.

**Stage E — update T** (uses `rsub_unary_tile` to eliminate `cb_const_ones`):
- `T ← T · (1 - alpha) · sat_mask`:
```
acquire
  copy_tile(cb_alpha, 0, dst=0)
  rsub_unary_tile(0, 1.0f_bits)              // dst[0] = 1 - alpha
  pack_tile(0, cb_one_minus_alpha)
commit; release

acquire
  mul_tiles(cb_T_state, cb_one_minus_alpha, 0, 0, dst=0)  // T *= (1-alpha)
  mul_tiles(cb_sat_mask, /*requires CB operand*/, ...)
  // In practice: need two mul steps because mul_tiles is CB-to-Dst
  pack_tile(0, cb_T_tmp)
commit; release

acquire
  mul_tiles(cb_T_tmp, cb_sat_mask, 0, 0, dst=0)  // T *= sat_mask
  pack_tile(0, cb_T_state)
commit; release
```

*Potential fusion*: if `cb_sat_mask` is already in Dst from D2, we could skip reloading, but D2's `release` frees Dst. Keeping Stage E as 3 blocks is cleanest.

**Stage F — sat_mask refresh (every 16 Gaussians)**:

**Ordering**: refresh at the **top** of the per-Gaussian loop when `(g % 16 == 0 && g > 0)`, *before* Stage D of the current Gaussian. This ensures saturated pixels don't accumulate one extra Gaussian. Skip when g=0 (mask is already all-ones from init).

Uses `unary_ge_tile` to eliminate `cb_const_1e-4`:
```
if ((g & 15) == 0 && g > 0) {
    acquire
      copy_tile(cb_T_state, 0, dst=0)
      unary_ge_tile(0, 1e-4f_bits)       // dst[0] = (T >= 1e-4) ? 1 : 0
      pack_tile(0, cb_sat_mask)
    commit; release
}
```

**Semantics**: saturated pixels (T < 1e-4) get mask=0 and stay that way (T×0=0 on next Stage E, so future Stage F refreshes keep producing 0). Subsequent Gaussians' `contrib` gets multiplied by `cb_sat_mask`, so saturated pixels stop accumulating.

**Latency note**: up to 15 Gaussians may continue to blend into a nominally-saturated pixel between refreshes. Bounded contribution per stale Gaussian: `alpha · T · color ≤ 1 × 1e-4 × 1 = 1e-4`. Over 15 stale Gaussians: ~1.5e-3 per channel, below bf16 noise floor (~8e-3). Acceptable. **Do not** "fix" this by refreshing every Gaussian — the ~3 SFPU ops per refresh would cost more than the stale-Gaussian leakage saves.

---

**SFPU/FPU ops used** (verified in tt-metal Rev 2 review):

| Op | Signature | Purpose |
|---|---|---|
| `sub_tiles(cb_a, cb_b, i_a, i_b, idst)` | CB→Dst | Not used (replaced by scalar-unary) |
| `mul_tiles(cb_a, cb_b, i_a, i_b, idst)` | CB→Dst | dx²=dx·dx, T·(1-alpha), etc. |
| `add_tiles(cb_a, cb_b, i_a, i_b, idst)` | CB→Dst | quadratic-form sum |
| `sub_unary_tile(idst, fp32_bits)` | Dst in-place | `dx = px - mean_x` |
| `mul_unary_tile(idst, fp32_bits)` | Dst in-place | scalar scaling (`a·dx²`, `opacity·weight`, `-0.5·Q`) |
| `rsub_unary_tile(idst, fp32_bits)` | Dst in-place | `1 - alpha` (eliminates `cb_const_ones`) |
| `addcmul_tile<DataFormat>(i0, i1, i2, o, fp32_bits)` | Dst ternary + scalar | `o = i0 + scalar·i1·i2` — fused color accumulate |
| `exp_tile(idst)` | Dst in-place | polynomial exp |
| `copy_tile(cb, idx, idst)` | CB→Dst | load CB tile into Dst |
| `pack_tile(idst, cb)` | Dst→CB | spill Dst tile back to CB |
| `fill_tile(idst, scalar)` | Dst fill | constant tile fill (then `pack_tile` to store) |
| `binary_min_tile(i_a, i_b, idst_out)` | Dst×Dst→Dst | `min(power, 0)`, `min(alpha, 0.99)` |
| `unary_ge_tile(idst, fp32_bits)` | Dst in-place | `dst = (dst ≥ scalar) ? 1 : 0` — sat_mask (eliminates `cb_const_1e-4`) |
| `tile_regs_acquire/commit/wait/release` | lifecycle | Dst section handshake |
| `cb_wait_front/pop_front/reserve_back/push_back(cb, n)` | CB sync | producer/consumer handshake |
| Init: `binary_op_init_common`, `add_tiles_init`, `mul_tiles_init`, `sub_tiles_init`, `copy_tile_init`, `exp_tile_init`, `binop_with_scalar_tile_init`, `binary_max_min_tile_init`, `rsub_unary_tile_init`, `unary_ge_tile_init` | once per kernel |

**Ops explicitly NOT used** (corrected from Rev 1/2): `sub_tiles_to_cb`/`mul_tiles_to_cb`/`add_tiles_to_cb` (don't exist); `zero_tile` (use `fill_tile(0.0f)`); `min_tile`/`max_tile` (use `binary_min_tile`/`binary_max_tile`); `reduce_tile` (not needed — sentinel mask uses `unary_ge_tile`, not reduction).

**Ops explicitly NOT used (corrected from Rev 1)**: `sub_tiles_to_cb`, `mul_tiles_to_cb`, `add_tiles_to_cb` (don't exist); `zero_tile` (use `fill_tile(0.0f)`); `min_tile`/`max_tile` (use `binary_min_tile`/`binary_max_tile`).

---

**Per-Gaussian op count (Rev 3, post-fusion)**: ~20 SFPU/FPU tile operations per Gaussian (reduced from ~22–25 by Stage D fusion via `addcmul_tile`). Throughput analysis:

| Op class | Count | Cycles/op | Subtotal |
|---|---|---|---|
| Simple binary (`mul_tiles`, `add_tiles`, `sub_tiles`) | ~10 | 128 | 1280 |
| Scalar-unary (`*_unary_tile`) | ~6 | 128 | 768 |
| `exp_tile` (fp32-accurate polynomial) | 1 | ~800 | 800 |
| `copy_tile` / `pack_tile` | ~8 | 64 | 512 |
| `addcmul_tile` (fused, ~1.5× base cost) | 3 | 192 | 576 |
| Other (`binary_min_tile`, `rsub_unary_tile`, `unary_ge_tile`) | ~3 | 128 | 384 |
| Lifecycle overhead (acquire/commit/release × ~7 blocks) | 7 | ~30 | 210 |
| **Total per Gaussian** | | | **~4500 cycles** |

**Per 500-Gaussian tile**: ~2.25 M cycles ≈ **2.25 ms at 1 GHz** (conservative; `exp` may be faster in practice).

**Single-core full image (400 tiles × 640×640)**: ~900 ms worst case, realistically **~700 ms** accounting for overlap with NoC streaming and sat_mask early-culling on typical tiles with 30-80 effective blends.

**Multi-core (72 cores × 12–15× realistic speedup with LPT load balancing)**: **~50-80 ms per frame**, i.e. 12-20 FPS for 640×640.

**Comparison to CPU baseline**: Python+NumPy `alpha_blend` at 640×640 with 10K Gaussians takes 2–8 s per frame. Even single-core Wormhole is a **3-10× speedup**; multi-core is **30-100×**. Thesis narrative holds strongly.

**Optimization paths (v2)**: reader-signal block-wide early termination (80% savings on saturated-tile-dominated scenes); deeper CB pipelining; SFPU op fusion; sharded DRAM placement.

### 6.3 Writer Kernel (`writer_alpha_blend.cpp`)

**Processor**: NCRISC (`RISCV_1`), uses NoC1.

**Runtime args**:
- `arg0`: `output_image` DRAM base address
- `arg1`: `first_tile_id`
- `arg2`: `num_tiles`

**Responsibilities**: For each assigned screen tile, wait on 3 pages from `cb_color_out`, async-write tiles `3·tile_id + {0,1,2}` back to DRAM, barrier, pop.

---

## 7. Host Orchestration

**Entry point**: `alpha_blend.cpp` (C++ driver binary).

**Phases**:
1. Device setup: `MeshDevice::create_unit_mesh`, `Program`, core selection.
2. DRAM buffer allocation (5 buffers: gaussians, offsets, px, py, output).
3. Upload host-prepared bytes via `EnqueueWriteMeshBuffer`.
4. Circular buffer creation (see §6).
5. Kernel creation with `ComputeConfig { HiFi3, fp32_dest_acc_en=true, math_approx_mode=false }`.
6. `SetRuntimeArgs` per core.
7. `EnqueueMeshWorkload(..., blocking=true)`.
8. `EnqueueReadMeshBuffer` → untilize on host → bf16-to-uint8.

**Multi-core wiring (v2)**: `split_work_to_cores(grid, num_screen_tiles)` returns two core groups; iterate with accumulating `first_tile_id` offset.

**Python integration (deferred to post-PoC)**: pybind11 module `gsplat_tt_kernel.alpha_blend(attribute_packs, tile_offsets, px_tiles, py_tiles, H, W) → np.ndarray`, where `attribute_packs` is the flat `(N, 9)` fp32 array described in §5.1. Until then, standalone binary reads/writes `.npy` files.

---

## 8. File Layout

```
tt-metal/tt_metal/programming_examples/gaussian_splatting/
├── CMakeLists.txt                              (tt-metal picks up via parent add_subdirectory)
├── alpha_blend.cpp                             (host driver + main())
├── alpha_blend_host.h                          (shared constants: CB indices, tile sizes)
└── kernels/
    ├── dataflow/
    │   ├── reader_alpha_blend.cpp
    │   └── writer_alpha_blend.cpp
    └── compute/
        └── alpha_blend_compute.cpp
```

**Git tracking**: add to project `.gitignore`:
```gitignore
tt-metal/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/
!tt-metal/tt_metal/programming_examples/gaussian_splatting/**
```

**Python-side additions**:
- `rasterization.py`: new function `prepare_kernel_inputs(means_2d, cov_inv, colors, opacities, sorted_gaussian_ids, tile_ranges, H, W) → (attribute_packs, tile_offsets, px_tiles, py_tiles)` where `attribute_packs` is the flat `(N, 9)` fp32 array (§5.1). Dumps to `.npy` for now; later returns arrays directly to pybind module.

---

## 9. Validation Strategy

### 9.0 Tier 0 — Smoke tests (before any kernel correctness work)

Catch 80% of "device boots but output is black" issues:

**T0.0 — Hello tt-metal**: empty kernel on core (0,0) that does `DPRINT << "Hello from (" << my_x << "," << my_y << ")" << ENDL();`. Validates toolchain, device enumeration, DPRINT server. Without this, any device failure looks like a kernel bug.

**T0.1 — Pass-through kernel**: reader writes known pattern (e.g. `0xDEADBEEF` filling a 2KB tile) to CB; compute kernel is `copy_tile` identity; writer dumps to DRAM. Host readback asserts pattern match. Confirms reader→compute→writer plumbing end-to-end.

**T0.2 — CB push/pop count**: reserve a small L1 scratch counter region per kernel; bump on each push/pop; read back via `tt::llrt::read_hex_vec_from_core` after `Finish()`. Confirms no silent overruns. (DPRINT-based alternative works but disables for timing.)

**T0.3 — Non-NaN, non-zero output**: after a single-Gaussian render, host checks `not np.isnan(img).any() and img.max() > 0 and img.min() >= 0`. Catches NaN injections and stuck-zero bugs.

**T0.4 — Single-Gaussian single-pixel identity**:
- Scene: 1 Gaussian at `mean=(16.5, 16.5)` (pixel center), `cov_inv = [[100, 0], [0, 100]]` (sharp Gaussian, `exp(-50) ≈ 2e-22` at distance 1 pixel).
- Color: `(1, 0, 0)` red. Opacity: 1.0.
- Camera: orthographic, 32×32 image.
- Expected: pixel `(16, 16)` ≈ `(1, 0, 0)` within ±0.01 (bf16 precision). All other pixels < 1e-3 on all channels.
- After-render T: ≈ 0 at center (saturated), ≈ 1 elsewhere.

**T0.5 — Two-Gaussian alpha blend**: two overlapping Gaussians at the same pixel, α=0.5 each, colors red and blue (red in front). Expected output: `(0.5, 0, 0.25)` — verifies front-to-back compositing, not just single-splat math.

**T0.6 — Saturation / sat_mask kick-in**: 50 opaque (opacity=1.0) Gaussians stacked at one pixel. Expected: color converges to the nearest Gaussian's color, T drops below 1e-4 after ~10 Gaussians, sat_mask takes over. Validates Rev 3 sentinel-mask logic end-to-end.

### 9.1 Tier 1 — Unit tests per math step (single-Gaussian)

Standalone C++ tests feeding hand-computed `(mean, cov_inv, color, opacity)` for one Gaussian + known pixel coords. Verify each kernel intermediate against NumPy ground truth. Target: ≤ 1e-3 absolute error (bf16 tolerance).

**Also add**: a "numeric sanity notebook" — host-only NumPy port of the compute kernel math that reads the same `.npy` inputs Tier 2 uses. Gives a golden fp32 reference AND a bf16-simulated reference before the device is ever loaded. Isolates algorithm bugs from tt-metal bugs.

### 9.2 Tier 2 — End-to-end PSNR vs CPU reference

Python harness dumps `(attribute_buffers × 9, tile_offsets, px_tiles, py_tiles)` for a chosen scene + camera to `.npy`. Standalone C++ binary loads, runs kernel, writes `output.npy`. Python compares against CPU `alpha_blend()`:

- **Target PSNR**: ≥ 35 dB (research projection: 38–42 dB).
- **SSIM**: ≥ 0.98.
- **Visual diff**: side-by-side PNGs saved; spot-check for banding, black pixels, color shifts.

Tested on ≥3 camera poses per scene, ≥2 scenes (one small ~10K Gaussians, one medium ~50K).

### 9.3 Tier 3 — Interactive viewer integration

Wire the kernel into [viewer.py](../../../viewer.py) via subprocess (phase 1) or pybind11 (phase 2). Orbit camera; confirm no NaN/black/artifacted frames.

### 9.4 Debugging support

- `DPRINT` macros in kernels (dev builds only).
- CB readback from host harness for mid-pipeline inspection.
- CPU-equivalent path in C++ binary (`--cpu-fallback` mode) for isolating tt-metal bugs from algorithm bugs.

### 9.5 Success criteria (decomposed, tiered)

**v1a — single Gaussian, single tile, single core** (earliest real milestone):
- One 32×32 tile rendered with a single hand-placed Gaussian.
- Visual output matches hand-computed Gaussian footprint.
- Confirms: acquire/commit cycle, scalar-unary ops, mul/sub/add tiles, exp, accumulator spill, output pack, NoC writeback.

**v1b — full scene, single core** (main PoC milestone):
- 640×640 image of ~10K-Gaussian scene renders end-to-end via core (0,0).
- PSNR ≥ 35 dB vs CPU reference on 3 camera poses.
- SSIM ≥ 0.98 (catches structural artifacts PSNR may miss).
- No visible banding, gaps, or wrong-color Gaussians.

**v1c — multi-core** (main performance milestone):
- Same PSNR and SSIM as v1b.
- Speedup over single-core: **≥15× floor / ≥20× stretch target**. (Research shows 20× is realistic but not guaranteed without work-stealing; LPT load balancing on Gaussian counts brings 20× in reach.)

**v2 — optimization pass** (stretch, only if v1c hits floor):
- Per-lane `power > 0` and `alpha < 1/255` skips.
- Deeper CB pipelining.
- Target: ≥30× speedup over single-core.

---

## 10. Known Risks

| Risk | Mitigation |
|---|---|
| Scratch CB L1 overflow | Budget: ~45 KB total (state + scratch + constants + inputs). Far under 1.4 MB. No concern. |
| Dst register lifecycle (no long-lived acquire) | Kernel cycles `acquire/commit/release` per Gaussian; running state lives in L1 state CBs, reloaded via `copy_tile` each iteration. Explicit in §6.2. |
| Dst-to-Dst binary ops don't exist | All binary math uses CB operands; state-CB spills documented. |
| Sentinel-mask cost | Per-Gaussian `mul_tiles(cb_contrib, cb_sat_mask)` costs 1 op; sat-mask refresh every 16 Gaussians costs ~3 ops. Net ~0.5 ops/Gaussian overhead. |
| Tilize padding mismatch | Image H,W forced to multiples of 32 on host; crop after readback. |
| WH B0 HiFi4 bug #38306 | Locked to HiFi3 + `fp32_dest_acc_en=true`; validated via `verify_numerical_configuration` pattern from production ttnn ops. |
| `fp32_dest_acc_en` not actually giving fp32 exp | Tier 0 numeric sanity test: feed known scalar through `exp_tile`, read back, compare to `np.exp` fp32. |
| `ttnn.untilize` shape compatibility with `(N, 3, 32, 32)` | Phase 0 prototype: feed a mock buffer through untilize and verify output shape. Fall back to 3 separate single-channel output buffers if untilize doesn't handle the 4-D shape. |
| Device-boot / toolchain friction | "Hello tt-metal" phase (eltwise_binary copy) before touching our kernel; `source tt-metal/venv/activate` + `./build_metal.sh` in Phase 0. |
| DPRINT perf trap | DPRINT disabled for all timing runs; explicit `#ifdef DEBUG_DPRINT` guards. |
| Silent CB underflow / hang | Wrap test runs with wall-clock timeouts (30 s/frame). Log per-CB push/pop counts. |
| Straggler tiles (irregular Gaussian counts) | v1c uses LPT load balancing on host; tiles sorted by cost, dispatched greedy-longest-first. Reader reads tile_id_list indirect instead of contiguous range. |
| Precision loss on T accumulator | T spills to bf16 CB between iterations; 7-bit mantissa on T compounds over ~100 Gaussians. Bounded by sat_mask cutoff at T<1e-4. Acceptable for 35 dB PSNR target per §3 precision analysis (see `fp32_dest_acc_en` discussion). |
| Op count understated (was 15, now ~22–25) | Perf projection updated in §6.2. Target: ~600 ms single-core, ~40 ms multi-core for 640×640. |
| Early-termination infeasibility (scalar-from-Dst) | Replaced with sentinel-mask approach in §6.2. Still delivers most of the asymptotic savings. |

---

## 11. Dependencies

- **tt-metal**: already vendored at `tt-metal/`. Must be buildable (see `tt-metal/INSTALLING.md`).
- **Pre-built**: `source tt-metal/venv/activate` (to set `TT_METAL_HOME`, `ARCH_NAME`, etc.) before `./build_metal.sh`.
- **Python**: existing deps (torch, plyfile, viser, nerfview).
- **Test**: NumPy for `.npy` serialization + PSNR/SSIM computation (`scikit-image` for SSIM if not already present).

---

## 12. Next Step

After user review + approval of this spec, transition to the `writing-plans` skill to produce a step-by-step implementation plan.

**Thesis budget alignment**: v1b is the **contract** deliverable; v1c is the **stretch**. If Phase 0 or Phase 2 blows up, cut Phase 4/5 and ship v1b + performance analysis chapter as the thesis artifact.

**Phase 0 — Numeric sanity + environment (parallelizable)**
1. Write **numeric sanity notebook**: NumPy fp32 + bf16-simulated port of the compute kernel, reads `.npy` inputs from Python, produces golden output. *Start this first — no device dependency; runs while tt-metal builds in background.*
2. Build tt-metal (`source tt-metal/venv/activate` + `./build_metal.sh`). Run `eltwise_binary` upstream example to verify device + toolchain + DPRINT server.
3. Prototype `ttnn.untilize` on mock `(N, 3, 32, 32)` bf16 buffer. If untilize fails on this shape, switch to the 3-separate-single-channel-buffer fallback from §5.4 before writing any kernel code.

**Phase 1 — Scaffold + smoke (Tier 0)**
4. Create `tt-metal/tt_metal/programming_examples/gaussian_splatting/` + CMakeLists (~8 lines, modeled on `eltwise_binary/CMakeLists.txt`).
5. Add `.gitignore` exception for this subdir.
6. DPRINT + iteration-loop setup (~1h; document the rebuild+rerun commands).
7. Build **standalone C++ test harness** that loads `.npy`, launches a program, dumps `.npy` output. This harness drives all Tier 0-2 tests.
8. Implement Tier 0 tests T0.0 through T0.3 (hello tt-metal, pass-through, CB counter, non-NaN).

**Phase 2 — v1a (single Gaussian, single tile)**
9. Implement host CPU pre-gather: `prepare_kernel_inputs()` in `rasterization.py` (~2h; most logic already exists).
10. Implement reader kernel: scalar CB + px/py tile CBs + meta CB.
11. **Split compute kernel bring-up into sub-phases**:
    - 11a. Compute writes solid color tile (verifies `fill_tile` + `pack_tile` + writer wiring).
    - 11b. Compute does a single `sub_unary_tile` on px CB (verifies scalar-unary path).
    - 11c. Full single-Gaussian math chain (Stages A-E, no loop yet).
12. Implement writer kernel: 3-tile RGB output (or fallback 3-buffer variant).
13. Validate Tier 0 T0.4 (single-pixel identity) + Tier 1 unit tests against numeric sanity notebook.

**Phase 3 — v1b (full scene, single core)**
14. Extend compute kernel with per-Gaussian loop + state-CB spills + Stage F sat_mask refresh every 16 Gaussians.
15. Validate Tier 0 T0.5 (two-Gaussian blend) + T0.6 (saturation).
16. Integrate **DeviceProfiler / Tracy timeline** into harness for cycle-accurate breakdown.
17. Validate Tier 2: PSNR ≥ 35 dB, SSIM ≥ 0.98 on 3 camera poses of 1 small scene.
18. Performance baseline: time per frame at 640×640, single-core.

**v1b checkpoint**: if on-track, proceed to Phase 4. If over budget, stop here and go to Phase 5 with v1b only.

**Phase 4 — v1c (multi-core, stretch)**
19. Host-side `split_work_to_cores` + LPT load balancing on Gaussian counts.
20. Reader reads indirect `tile_id_list` instead of contiguous range.
21. Benchmark speedup: ≥15× floor, ≥20× stretch.

**Phase 5 — Integration + thesis deliverables**
22. Wire into `viewer.py` via subprocess call to the standalone harness (Tier 3). **pybind11 binding deferred to post-thesis** — saves 4-6h.
23. Thesis benchmarks: **2 scenes × 1 resolution (640×640) × 3 backends (CPU / v1b / v1c)**. Narrower than Rev 2's matrix — depth of analysis beats breadth for MSc.
24. Performance analysis chapter: op count per Gaussian, measured cycle breakdown, roofline against SFPU peak, precision vs fp32 PSNR/SSIM.
