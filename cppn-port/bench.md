# Measurements — GlassOS CPPN, CUDA port

Every number on this page was produced on the machine described below, by the
programs in this directory. Raw console logs are kept verbatim in
`bench_raw.txt`, `correctness_raw.txt` and `glsl_compare.txt`.

## Hardware / software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080, Turing, **sm_75**, 46 SMs, 8 GB |
| Clocks | SM 1800 MHz nominal (1905 MHz observed under load), memory 7000 MHz, 256-bit → **448 GB/s** theoretical |
| Driver | 610.74 (WDDM), CUDA driver API 13030 |
| Toolkit | CUDA 12.6, `nvcc` V12.6.20, host compiler MSVC 14.29 (VS2019 Build Tools) |
| Flags | `-O3 -arch=sm_75 -lineinfo --ptxas-options=-v` — **no `-use_fast_math`**; the fast-math variants call intrinsics explicitly |
| OS | Windows 11 (26200) |
| Scene | fixed seed 20260811, `u_time = 7.5`, 6 organisms, 6 hidden nodes each |
| Resolution | 1920×1080 unless stated |

**Caveat that applies to every timing here:** this is a desktop GPU that is also
driving the display, with a browser and Teams running. Individual runs vary by
2–4× when something else grabs the GPU. Two things are done about that:

1. Each configuration is timed as **5 rounds × 100 launches**, and the
   **minimum** per-launch mean is kept (the least-interfered estimate).
2. All configurations in a table are timed **round-robin in a single loop**, so
   clock drift over the run cannot masquerade as a difference between them.
   This was not cosmetic — before interleaving, sequential timing blocks made
   the *same kernel* look 33 % slower at the end of a run than at the start.

Runs taken while the GPU was busy with other work were discarded; the numbers
below are from a run started with `nvidia-smi` reporting ~1 % utilisation.
Repeatability across four clean runs: variant A 1.61–1.69 ms, B 1.12–1.19 ms,
C 0.90–0.99 ms, i.e. roughly ±5 %.

---

## 1. Static resource usage (`cudaFuncGetAttributes`, and `ptxas -v`)

| variant | registers | local mem | constant | max threads/block |
|---|---|---|---|---|
| A naive (global genomes, no early-out) | 73 | 32 B | 1536 B | 768 |
| B constant + early-out | 64 | 32 B | 1536 B | 1024 |
| C B + intrinsics | **64** | **0 B** | 1536 B | 1024 |

`ptxas` reported **0 bytes of spill stores/loads** for all three, so the 32 B of
"local" in A/B is addressable local storage the compiler chose, not spilling.

## 2. Theoretical occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)

| threads/block | A naive | B constant | C fast |
|---|---|---|---|
| 64   | 12 blocks, 75 % | 16 blocks, 100 % | 16 blocks, 100 % |
| 128  | 6 blocks, 75 %  | 8 blocks, 100 %  | 8 blocks, 100 % |
| 256  | 3 blocks, 75 %  | 4 blocks, 100 %  | 4 blocks, 100 % |
| 512  | 1 block, 50 %   | 2 blocks, 100 %  | 2 blocks, 100 % |
| 1024 | **0 — will not launch** | 1 block, 100 % | 1 block, 100 % |

A's 73 registers put it over the 64-register line that lets Turing keep a full
1024 threads resident per SM, so A is capped at 75 % occupancy and cannot launch
1024-thread blocks at all.

## 3. Block-size sweep (variant C, 1920×1080, min of 5 interleaved rounds × 100)

| block | threads/block | ms | Mpix/s |
|---|---|---|---|
| 8×8    | 64  | 1.054 | 1967 |
| 16×4   | 64  | 1.059 | 1958 |
| 16×8   | 128 | 1.050 | 1975 |
| 32×2   | 64  | 1.065 | 1947 |
| 32×4   | 128 | 1.053 | 1969 |
| **32×8** | **256** | **0.923** | **2246** |
| 64×2   | 128 | 1.061 | 1954 |
| 64×4   | 256 | 0.930 | 2231 |
| 128×2  | 256 | 0.943 | 2199 |
| 256×1  | 256 | 0.964 | 2151 |

The interesting part is that the sweep separates cleanly by **total threads per
block, not by shape**: every 64- and 128-thread block lands at ~1.05 ms, every
256-thread block at 0.92–0.96 ms — a ~13 % step. Theoretical occupancy is 100 %
for all of them, so this is not an occupancy effect; the most likely cause is
per-block scheduling overhead (8100 blocks at 256 threads vs 32400 at 64) plus
better instruction-level parallelism within a larger block. I could not confirm
that hypothesis without a profiler (see §6). Within the 256-thread group, wider
blocks (32×8) beat 256×1 slightly, consistent with better 2D locality in the
`uchar4` stores.

## 4. Kernel configurations (1920×1080, block 32×8, min of 5 interleaved rounds × 100)

Six configurations from the 2×2×2 of {global | `__constant__`} × {no early-out |
early-out} × {libm | intrinsics}, so the three axes can be attributed separately.

| configuration | ms | Mpix/s | vs A | store BW (GB/s) |
|---|---|---|---|---|
| **A** global + no early-out *(naive baseline)* | 1.647 | 1259 | 1.00× | 5.04 |
| global + no early-out, genomes in `__constant__` | 1.742 | 1191 | **0.95×** | 4.76 |
| global + early-out | 1.028 | 2017 | 1.60× | 8.07 |
| **B** `__constant__` + early-out | 1.140 | 1819 | 1.44× | 7.28 |
| **C** B + SFU intrinsics + unroll | 0.922 | 2248 | 1.79× | 8.99 |
| **D** global + early-out + intrinsics *(fastest found)* | **0.862** | **2406** | **1.91×** | 9.62 |

### What the ablation actually says

- **The early-out is the whole memory-side win** (1.60×). It is the shader's
  `if (lr > 1.6) continue;`, and it works in CUDA for the same reason it works
  in a fragment shader: most of a 1920×1080 frame is outside all six organism
  discs, and the lanes of a warp are spatially adjacent, so whole warps skip the
  CPPN evaluation together. Divergence only costs anything on the ~1-warp-wide
  boundary of each disc.
- **`__constant__` memory made it *slower*, by ~5–10 %.** This was the
  hypothesis the port was built to test, and the measurement says no. This is
  an honest negative result, not a tuning failure: it reproduced across every
  clean run, in both the early-out and no-early-out arms.
- Therefore **D (global + early-out + intrinsics) is the fastest kernel here,
  not C** — 0.862 ms vs 0.922 ms. Variant C is kept because the brief specified
  the constant-memory progression; D is what I would ship.

### Why constant memory loses (evidence, and where the evidence stops)

From SASS (`cuobjdump --dump-sass`, summarised in `sass_summary.txt`):

| kernel | instrs | LDG | LDC | MUFU | FFMA |
|---|---|---|---|---|---|
| A global + no early-out | ≥4096 | 34 | 0 | 51 | 694 |
| `__constant__` + no early-out | ≥4096 | 13 | 6 | 51 | 684 |
| global + early-out | ≥4096 | 34 | 0 | 51 | 692 |
| B `__constant__` + early-out | ≥4096 | 13 | 18 | 51 | 684 |
| C `__constant__` + early-out + intrinsics | 3424 | **0** | 22 | **100** | 611 |
| D global + early-out + intrinsics | **3376** | 16 | 0 | **100** | 611 |

(`instrs` is a lower bound at 4096 — the disassembly's 4-hex-digit offsets wrap
there. The intrinsic variants carry `MUFU.SIN`, `MUFU.COS` and `MUFU.LG2`; the
others have only `MUFU.EX2/RCP/RSQ`.)

The kernel issues ~690 FFMAs against ~34 genome loads. The genome block is
**1536 bytes** — it is resident in L1 after the first warp touches it and stays
there, so the "global" path is already a cache broadcast. Swapping it for `LDC`
buys nothing and costs something: the constant path cannot fold a runtime-indexed
`float4` fetch into an instruction operand the way an immediate-offset constant
bank access can, and 128-bit `LDG.E.128` moves a whole `vec4` slot in one
instruction. That is the explanation I believe; I could not *prove* it without
per-instruction stall sampling, which needs the profiler (§6).

The store bandwidth column is the load-bearing number for "what kind of kernel
is this": the best variant writes 8.3 MB per frame in 0.862 ms = **9.6 GB/s
against 448 GB/s of theoretical peak, about 2 %**. This kernel is
overwhelmingly **compute-bound**, which is why the memory-layout optimisation
did nothing and the arithmetic optimisation did everything.

## 5. Fast-math: what it bought and what it cost

Applied as explicit intrinsics (`__sinf`, `__cosf`, `__expf`, `__powf`), never
`-use_fast_math`, so the choice is visible in the source.

- **Speed:** 1.60× → 1.91× (global+early-out arm), i.e. the intrinsics alone are
  worth **1.19×**. SASS shows why: real `MUFU.SIN` / `MUFU.COS` / `MUFU.LG2`
  instructions replace multi-instruction software sequences, and the whole
  kernel shrinks (variant C is the only one of the five under 4096
  instructions).
- **Accuracy:** measured, not assumed. Rendering the same frame with and
  without intrinsics and differencing all 6.2 M channels:
  **max difference 1 LSB, mean 0.000024 LSB, 0.0024 % of channels differ at all.**
  Visually and numerically free *for this workload*.
- **The caveat that belongs with that number:** `__sinf`/`__cosf` lose accuracy
  rapidly for large arguments (the hardware range reduction is limited), and
  `__powf(x,y)` = `__expf(y*__logf(x))` is poor near `x→0`. Here the arguments
  are small — CPPN pre-activations of order ±10, tone-map exponents in [0,1] —
  so the error stays under one 8-bit quantisation step. On a workload with large
  phase arguments (long animation times feeding straight into `sin`) this
  substitution would visibly break.

## 6. Nsight Compute — **not available on this machine**

```
ncu --section SpeedOfLight --section Occupancy .\cppn.exe
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

Nsight Compute 2024.3.0 is installed at
`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\`, but on Windows
the driver gates performance counters behind administrator rights, and this
session is not elevated (`IsInRole(Administrator)` → `False`). Lifting the gate
means either running elevated or changing the driver's `RmProfilingAdminOnly`
setting — a system security setting I am not going to change.

**So there are no achieved-occupancy, warp-execution-efficiency, or
memory-throughput counters in this report, and I am not going to guess at
them.** The fallback data actually collected instead:

- theoretical occupancy from the CUDA occupancy API (§2)
- register/local/spill counts from `ptxas -v` and `cudaFuncGetAttributes` (§1)
- static instruction mix from SASS (§4)
- achieved store bandwidth from CUDA-event timings and known bytes written (§4)
- a block-size sweep as an empirical stand-in for occupancy tuning (§3)

What is genuinely missing: which pipe (FMA vs SFU vs LSU) is the actual
throughput limiter, and how much of the early-out win is warp-skip vs cache.

---

## 7. Correctness

### 7a. Against an independent CPU reference

`cpu_ref.cpp` is a second, separate transcription of the GLSL, templated on
scalar type. It shares only the genome-packing header with the CUDA code, not
the shading math.

| reference | compared with | max abs diff | mean abs diff | % channels >1 LSB | SSIM (luma) |
|---|---|---|---|---|---|
| **double**, full shader | CUDA A / B / C | 28 | 2.211 | 56.4 % | 0.997857 |
| **double**, dither off | CUDA B / C | 25 | 1.685 | 40.7 % | 0.998495 |
| **float**, dither off | CUDA B / C | **1** | **0.0006** | **0.0000 %** | **1.000000** |

Read those three rows together — they are the whole story:

- Against a **float** reference the CUDA output is **exact to within 1 LSB on
  every one of 6.2 million channels**, mean error 0.0006 LSB. The port is
  correct.
- Against a **double** reference the difference is much larger, and that is
  expected rather than alarming. The shader's value noise is
  `fract(p.x*p.y)` of coordinates that reach ~10³ after the ×123.34 / ×345.45
  multiplies. `fract` of a large product is a chaotic function of the last bits
  of its argument, so float and double disagree essentially at random there,
  and 5 octaves of fBm amplify it. The 0.53 LSB that rows 1→2 differ by is the
  shader's own ±0.015 hash dither, isolated by rendering both with it disabled.
- A/B/C agree with each other to 1 LSB, i.e. the optimisations did not change
  the image.

### 7b. Against the real GLSL shader — done, apples to apples

I did manage an honest side-by-side. `glsl_harness.html` is generated by
`make_harness.py`: it lifts the **fragment shader source verbatim** out of
`D:\Projects\glassos-webgl\index.html` (only the `${MAXG}` / `${H}` template
substitutions are resolved), and feeds it the **same 384 packed floats** the
CUDA program used, dumped to `scene.json` by `cppn.exe`. No uniform is
re-derived on the JS side, so there is nothing to drift.

Rendered in headless Edge at 1920×1080, DPR pinned to 1, screenshotted:

```
UNMASKED=ANGLE (NVIDIA, NVIDIA GeForce RTX 2080 (0x00001E87) Direct3D11 vs_5_0 ps_5_0, D3D11)
```

— i.e. the real shader, on the same GPU, through the ANGLE/D3D11 backend, not a
software rasteriser.

| | max abs diff | mean abs diff | % >1 LSB | % >2 LSB | SSIM (luma) |
|---|---|---|---|---|---|
| CUDA (A, B, C) vs WebGL2 | **4** | **0.0910** | 1.81 % | 0.62 % | **0.999965** |

That is a stronger agreement than against the double reference, exactly as
predicted: GLSL `highp float` and CUDA `float` are both IEEE binary32, so they
track each other through the chaotic hash almost perfectly. The residual 4 LSB
outliers are isolated pixels on the value-noise lattice boundaries.

### 7c. GLSL vs CUDA *performance* — measured, but read the caveats

The harness also runs `EXT_disjoint_timer_query_webgl2`: 200 draws of the same
full-screen quad, GPU time, same GPU, same resolution, same uniforms.

| | best of 6 samples | samples (ms/draw) |
|---|---|---|
| WebGL2 / ANGLE-D3D11 | **0.921 ms** | 0.999, 1.195, 1.695, 1.332, 0.921, 1.061 |
| CUDA variant C | 0.902–0.988 ms | (see §4) |
| CUDA variant D | **0.862 ms** | |

**The honest conclusion is that they are the same speed.** Best-case WebGL
0.921 ms vs best-case CUDA C 0.902 ms is a 2 % difference — inside the noise on
this machine. Variant D is ~7 % ahead of best-case WebGL, which is real but
small. **I am not claiming CUDA beats the GLSL version.** The plausible reading
is that both end up executing very similar SASS: ANGLE compiles GLSL → HLSL →
D3D bytecode → the same NVIDIA shader compiler backend, and the driver applies
its own fast-math substitutions by default, which is roughly what variant C/D
do by hand.

Further caveats on that comparison: WebGL timer-query samples were far noisier
than CUDA events (0.92–1.70 ms across six browser launches, each a fresh
process); the WebGL number includes whatever fixed per-draw state cost ANGLE
adds; and CUDA event timing excludes the device→host copy, which the WebGL path
also excludes since nothing is read back during the timed loop.

---

## Reproducing

```
build.bat            :: nvcc -O3 -arch=sm_75, both binaries
cppn.exe 1920 1080 100   :: §1-§5, writes out_variant_*.png, cuda_*.raw, scene.json
cpu_ref.exe 1920 1080    :: §7a, writes out_cpu_ref.png and the diff table
python make_harness.py   :: §7b, builds glsl_harness.html from the live shader + scene.json
```
then screenshot the harness (the exact headless Edge invocation is at the top of
`make_harness.py`) and run the compare snippet in the same file.
