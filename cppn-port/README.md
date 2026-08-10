# GlassOS CPPN — CUDA port

A WebGL fragment shader ported to CUDA, then optimised and measured properly.

The original is the renderer inside [`glassos-webgl`](../../glassos-webgl):
a full-screen shader that evaluates up to 6 evolved **CPPN** organisms per
pixel — 6 hidden nodes each, activation function selected per node from
{sin, gaussian, tanh, triangle} by the genome — draws the analytic zero-contour
of each field with chromatic dispersion, and composites that over a refractive
glass sphere with 5-octave fBm caustics, Fresnel rim, tone map and hash dither.
Every pixel is independent; there is no inter-thread communication anywhere.
It is a pure ALU-throughput problem, which makes it a clean subject for a port.

![CUDA output](out_variant_C.png)

*1920×1080, rendered by `cppn.exe` on an RTX 2080. Pixel-for-pixel comparisons
against both a CPU reference and the original GLSL are in
[`bench.md`](bench.md).*

---

## Why this port is interesting

**GLSL uniform array → CUDA `__constant__` memory.** The shader packs each
organism into exactly 16 `vec4` uniform slots (pos/radius/fade, tint/timeRate,
12 slots of hidden-layer weights, 2 slots of output weights + bias + line
width). 6 × 16 × 16 B = 1536 bytes, read identically by every invocation.
`__constant__` is the textbook CUDA analogue: small, read-only, warp-uniform,
broadcast from a dedicated cache. So I built it that way — **and measured it
5–10 % slower than just leaving the data in global memory.** With ~690 FFMA
against ~34 loads and a 1536-byte working set that never leaves L1, the "global"
path is already a cache broadcast, while `LDC` gives up the 128-bit vector load
that fetches a whole `vec4` slot in one instruction. The textbook answer was
wrong for this kernel, and the ablation table shows it.

**Fragment early-out → warp-level work skipping.** The shader's
`if (lr > 1.6) continue;` is a per-fragment branch. In CUDA it becomes a
per-lane predicate, and it only saves anything when *all 32 lanes* of a warp
agree. They usually do: warp lanes are spatially adjacent, and most of a
1920×1080 frame is outside all six organism discs. This one line is worth
**1.60×** — the single biggest win in the whole port, larger than everything
else combined. Divergence only costs on the ~1-warp-wide silhouette of each disc.

**Occupancy is a register-count cliff, not a knob.** The naive kernel uses 73
registers, which is over the 64-register line Turing needs to keep 1024 threads
resident per SM: 75 % theoretical occupancy, and 1024-thread blocks will not
launch at all. Restructuring drops it to 64 registers and 100 %. Meanwhile the
block-size sweep separates purely by *threads per block* — every 64- and
128-thread configuration lands at ~1.05 ms, every 256-thread one at ~0.93 ms —
even though all of them are at 100 % theoretical occupancy. Occupancy was not
the limiter; block scheduling overhead was.

---

## Measured results

Full method, caveats and raw logs: **[`bench.md`](bench.md)**.
RTX 2080 (sm_75) · driver 610.74 · CUDA 12.6 · 1920×1080 · min of 5 interleaved
rounds × 100 launches, CUDA events, no `-use_fast_math`.

| configuration | ms | Mpix/s | vs baseline |
|---|---|---|---|
| **A** global genomes, no early-out *(naive baseline)* | 1.647 | 1259 | 1.00× |
| genomes in `__constant__`, no early-out | 1.742 | 1191 | 0.95× |
| global + early-out | 1.028 | 2017 | 1.60× |
| **B** `__constant__` + early-out | 1.140 | 1819 | 1.44× |
| **C** B + SFU intrinsics + `#pragma unroll` | 0.922 | 2248 | 1.79× |
| **D** global + early-out + intrinsics *(fastest found)* | **0.862** | **2406** | **1.91×** |

Best block size 32×8 (256 threads); the sweep table is in `bench.md`.
Store bandwidth at the best variant is 9.6 GB/s against 448 GB/s theoretical —
**~2 % of peak, so this kernel is compute-bound**, which is exactly why the
memory-layout change did nothing and the arithmetic change did everything.

**Correctness**

- vs an independent double-precision CPU reference, rendered in **float**:
  **max 1 LSB over all 6.2 M channels**, mean 0.0006 LSB, SSIM 1.000000.
- vs the **actual GLSL shader** — lifted verbatim from the live page into a
  standalone WebGL2 harness, fed the identical 384 packed uniform floats, run in
  headless Edge on the same RTX 2080 through ANGLE/D3D11:
  **max 4 LSB, mean 0.091 LSB, SSIM 0.999965.**
- fast-math cost: **max 1 LSB, 0.0024 % of channels differ** on this workload.
  The intrinsics' accuracy caveats (large `sin` arguments, `__powf` near zero)
  are spelled out in `bench.md`; they do not bite here, and I say why.

**Not measured:** Nsight Compute refuses to run — `ERR_NVGPUCTRPERM`, the
driver gates performance counters behind administrator rights on Windows and
this session is not elevated. So there is **no achieved occupancy, no warp
execution efficiency, no memory-pipe utilisation** in this report, and none is
guessed at. The fallback is theoretical occupancy from the CUDA occupancy API,
`ptxas -v` register/spill counts, static SASS instruction mix, measured store
bandwidth, and an empirical block-size sweep.

**On CUDA vs WebGL speed:** I measured both, on the same GPU, at the same
resolution, with the same uniforms (WebGL via `EXT_disjoint_timer_query_webgl2`).
Best WebGL 0.921 ms vs best CUDA 0.862 ms. **That is a tie within the noise on
this machine, and I am not claiming CUDA beats the shader.** Both paths
plausibly end up at very similar SASS.

---

## Files

| | |
|---|---|
| `cppn.cu` | all kernel variants, benchmark harness, PNG output |
| `genome.h` | deterministic re-implementation of the JS genome factory + the 16-`vec4` packing |
| `cpu_ref.cpp` | independent reference, templated on `float`/`double`, plus the diff/SSIM report |
| `png.h` | dependency-free PNG writer (stored deflate) |
| `build.bat` | `nvcc -O3 -arch=sm_75` |
| `make_harness.py` | builds `glsl_harness.html` from the live shader + `scene.json`; `--compare` diffs the screenshot |
| `sass_summary.py` | instruction-mix table from `cuobjdump --dump-sass` |
| `bench.md` | every measurement, with method and caveats |
| `bench_raw.txt`, `correctness_raw.txt`, `glsl_compare.txt`, `sass_summary.txt` | verbatim tool output |
| `out_variant_A/B/C.png`, `out_cpu_ref.png`, `glsl_shot.png` | rendered frames |

## Build and run

```bat
build.bat
cppn.exe 1920 1080 100      :: benchmark tables + out_variant_*.png + scene.json
cpu_ref.exe 1920 1080       :: correctness table + out_cpu_ref.png
python make_harness.py      :: GLSL harness (screenshot command is in the docstring)
python make_harness.py --compare
```

`build.bat` points at the CUDA 12.6 toolkit and the VS2019 Build Tools host
compiler by absolute path; edit the two variables at the top if yours differ.
