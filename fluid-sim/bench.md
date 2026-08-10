# Benchmarks and correctness measurements

Every number on this page was produced on the machine described below by the
executable in this repository. Nothing is copied from a paper, a tutorial, or
another GPU. Where a result is unflattering it is reported as measured.

## Test system

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 (Turing, **sm_75**), 46 SMs, 8 GB GDDR6 |
| Memory interface | 256-bit @ 7000 MHz -> **448.0 GB/s** theoretical peak |
| Driver | 610.74 (WDDM, GPU also drives the desktop) |
| CUDA | Toolkit 12.6.20 (`nvcc`), driver API reports 13.3 |
| Host compiler | MSVC 14.29.30133 (VS 2019 Build Tools) |
| Build | `-O3 -arch=sm_75 -lineinfo` |
| Precision | `float` throughout |

`N` always denotes the **interior** (simulated) resolution. The allocation adds
a one-cell ghost border, so `N=512` allocates 514x514 per field.

## Timing methodology

- All timings use **CUDA events** (`cudaEventRecord` / `cudaEventElapsedTime`).
- **Warm-up is excluded**: 10 untimed frames precede every measurement.
- Each measurement averages **>= 30 frames** inside a single event pair, and the
  whole measurement is **repeated 3-6 times with the minimum kept**.
- Kernel-level numbers (`kbench`) use 300 back-to-back launches per trial, 6
  trials, and report min / mean / max so the spread is visible.

### Why minimum-of-trials, and a warning about this machine

This GPU also drives the Windows desktop (Chrome, Teams, VS Code compositing).
Background GPU work intermittently steals SM time, and it is not small: during
this work a configuration whose clean time is **4.28 ms** was measured once at
**35.10 ms**, and `N=2048, 80 sweeps` (true value ~17.2 ms) read **53.17 ms** on
a run where the interference persisted across all six trials.

Two guards were added after seeing this:
1. minimum-of-N-trials rather than a single average;
2. a monotonicity sanity check on every sweep (time must increase with grid size
   and with iteration count) before a table is accepted.

A quiet, non-display GPU would not need either. **Treat these absolute
milliseconds as good to roughly +/-5%, and the A/B ratios (measured back-to-back
in one process) as considerably tighter than that.**

---

## 1. Correctness

### 1.1 Manufactured solution: does the projection operator actually work?

The field `u = grad(cos(k*pi*x) * cos(k*pi*y))` is a **pure gradient**, so it is
entirely curl-free and an exact Helmholtz-Hodge projection must reduce it to
**u = 0**. Whatever velocity survives is precisely the solver error. `cos` (not
`sin`) is required so that `u.n = 0` on all four walls, making the field
compatible with the no-flux Neumann pressure boundary — with `sin`, part of the
field is unremovable by construction and the test measures nothing.

`N=512`, RMS|u| remaining after one projection:

| mode k | RMS abs u initial | Jacobi 20 | Jacobi 80 | Jacobi 320 | MG 1 cyc | MG 4 cyc | MG 16 cyc |
|---|---|---|---|---|---|---|---|
| 1  | 4.339e-03 | 0.04% | 0.15% | **0.60%** | 31.9% | 78.4% | **99.78%** |
| 4  | 1.736e-02 | 0.60% | 2.38% | 9.19% | 97.2% | 99.97% | **99.99%** |
| 16 | 6.942e-02 | 9.18% | 31.96% | 78.47% | 98.4% | 99.76% | 99.76% |
| 64 | 2.777e-01 | 76.45% | 96.02% | 96.19% | 96.06% | 96.19% | 96.19% |

(percentages = fraction of the field removed; higher is better)

Reading this table:

- **The implementation is correct** — high-frequency error is annihilated
  exactly as the theory says it should be.
- **Jacobi is a smoother, not a solver.** It removes 76% of mode 64 in 20
  sweeps but only 0.6% of mode 1 in 320 sweeps. This is the textbook behaviour
  of a local relaxation scheme and it is the single most important fact about
  the classic Stam solver: at the 20-40 iterations everyone uses, the smooth
  component of the divergence is essentially untouched.
- **Multigrid fixes exactly that mode.** 99.78% of mode 1 removed in 16 cycles
  versus 0.60% for 320 Jacobi sweeps.
- **Mode 64 plateaus at ~96% for both solvers.** That floor is *discretisation*
  error, not solver error: at k=64 on a 512 grid there are only 8 cells per
  wavelength, so the analytic gradient is not exactly a discrete gradient. Both
  solvers converge to the same floor, which is the expected signature of a
  discretisation limit rather than a bug.

### 1.2 Divergence and dye mass in the real simulation

Buoyant plume, `N=512`, 300 frames, dye injection stopped after frame 60 so that
mass conservation can be measured on a closed system. Statistics over frames
60-299.

| solver | worst max abs div before | worst max abs div after | mean RMS before | mean RMS after | RMS reduction | dye mass drift |
|---|---|---|---|---|---|---|
| Jacobi, 40 sweeps | 8.894e-01 | 6.142e-01 | 1.883e-02 | 1.732e-02 | **1.1x** | **-92.07%** |
| red-black GS, 40 sweeps | 4.629e-01 | 4.105e-01 | 1.344e-02 | 1.271e-02 | **1.1x** | **-93.92%** |
| multigrid, 4 V-cycles | 6.544e-02 | **1.125e-03** | 1.867e-03 | **4.861e-05** | **38.4x** | **-15.97%** |

**This is the proof that the projection works** — and equally, the proof that
the textbook iteration count does not. With 40 Jacobi sweeps the projection
removes about 10% of the divergence, the velocity field stays measurably
compressible, and the dye field loses 92% of its mass over 240 frames because a
compressible flow does not preserve the volume the dye occupies. Switching only
the linear solver, with no change to advection or boundaries, takes divergence
down by 38x and mass loss down to 16%.

The residual **-16% mass drift under multigrid is genuine and expected**: it is
the numerical dissipation of semi-Lagrangian advection, which is stable and
unconditionally bounded but *not* conservative. Fixing that requires a different
advection scheme (MacCormack / BFECC / a conservative flux form), not a better
pressure solve. It is not fixed here.

Per-frame divergence and dye mass for the shipped 300-frame run are logged in
`results/sim_log.csv` (columns: frame, dye_mass, maxdiv_before_projection,
maxdiv_after_projection). Representative values from that run:

| frame | dye mass | max abs div before | max abs div after |
|---|---|---|---|
| 0   | 588.12    | 4.504e-02 | 1.510e-04 |
| 50  | 19125.78  | 9.220e-02 | 1.614e-03 |
| 100 | 21688.61  | 2.986e-02 | 7.856e-04 |
| 150 | 20498.81  | 2.016e-02 | 4.398e-04 |
| 200 | 19820.84  | 1.460e-02 | 3.468e-04 |
| 250 | 19454.51  | 1.580e-02 | 4.353e-04 |

Across frames 60-299 the per-frame `max|div|` reduction factor ranges from
**36.3x to 58.5x, median 43.5x** — consistent, with no frame where the projection
fails.

### 1.3 Pressure-solver convergence on a fixed field

One frozen, fully developed turbulent velocity field (`N=512`, RMS|div| =
2.552e-02, max|div| = 4.106e-01); projection run at varying sweep counts, timed
in isolation over 50 reps.

| solver | sweeps | RMS abs div after | reduction | max abs div | ms |
|---|---|---|---|---|---|
| Jacobi | 10 | 2.487e-02 | 1.0x | 3.728e-01 | 0.182 |
| Jacobi | 40 | 2.341e-02 | 1.1x | 3.119e-01 | 0.598 |
| Jacobi | 160 | 2.014e-02 | 1.3x | 2.178e-01 | 2.230 |
| Jacobi | 640 | 1.538e-02 | 1.7x | 1.235e-01 | 8.744 |
| red-black GS | 10 | 2.435e-02 | 1.0x | 3.503e-01 | 0.237 |
| red-black GS | 40 | 2.203e-02 | 1.2x | 2.687e-01 | 0.905 |
| red-black GS | 160 | 1.788e-02 | 1.4x | 1.679e-01 | 3.350 |
| red-black GS | 640 | 1.283e-02 | 2.0x | 8.752e-02 | 13.267 |

Note that **red-black GS at 320 sweeps lands on exactly the same residual as
Jacobi at 640** (1.5383e-02 vs 1.5381e-02). That is a clean empirical
confirmation of the textbook result that Gauss-Seidel converges at twice the
rate of Jacobi per sweep — and it is also the reason RBGS is *not* a win here
(see optimisation 3 below): it costs about twice as much per sweep on this
hardware, so the two effects cancel.

---

## 2. Stage breakdown

Per-stage CUDA events inside the frame loop; minimum over 4 trials of 50 frames.
The per-stage synchronisation serialises stages that would otherwise overlap, so
the instrumented total runs slightly above the clean single-event whole-frame
number. Both are given.

### N=512, Jacobi 40 sweeps — whole frame **1.272 ms** (786 FPS)

| stage | ms/frame | % |
|---|---|---|
| sources + buoyancy | 0.0403 | 3.1% |
| diffusion (20 Jacobi sweeps) | 0.5767 | 44.4% |
| advect velocity | 0.0207 | 1.6% |
| **projection (40 Jacobi sweeps)** | **0.5943** | **45.8%** |
| boundary | 0.0364 | 2.8% |
| advect dye | 0.0295 | 2.3% |
| instrumented total | 1.2980 | |

### N=2048, Jacobi 40 sweeps — whole frame **11.916 ms** (84 FPS)

| stage | ms/frame | % |
|---|---|---|
| sources + buoyancy | 0.1913 | 1.6% |
| diffusion (20 Jacobi sweeps) | 5.5265 | 46.3% |
| advect velocity | 0.2637 | 2.2% |
| **projection (40 Jacobi sweeps)** | **5.7421** | **48.1%** |
| boundary | 0.0266 | 0.2% |
| advect dye | 0.1863 | 1.6% |
| instrumented total | 11.9364 | |

### N=2048, multigrid — whole frame **16.459 ms** (61 FPS)

| stage | ms/frame | % |
|---|---|---|
| sources + buoyancy | 0.1938 | 1.2% |
| diffusion (20 Jacobi sweeps) | 5.6064 | 34.9% |
| advect velocity | 0.2675 | 1.7% |
| **projection (4 V-cycles)** | **9.7625** | **60.9%** |
| boundary | 0.0263 | 0.2% |
| advect dye | 0.1869 | 1.2% |
| instrumented total | 16.0433 | |

**The elliptic solves dominate completely.** Projection plus diffusion — both
just repeated 5-point stencil sweeps — are 90.2% of the frame at N=512 and 94.4%
at N=2048. The two advections together, which is where all the interesting
physics lives, are under 4%. Anyone optimising this solver should ignore
advection entirely and attack the linear solves; that is exactly what the
multigrid work below does.

(The diffusion stage is 20 Jacobi sweeps on each of `u` and `v` = 40 sweeps
total, which is why it costs almost exactly what the 40-sweep pressure solve
does. With `visc = 0` it disappears entirely.)

---

## 3. Grid-size and iteration-count sweep

Whole-frame ms, minimum of 4 trials of 30 frames, block 32x8.

### Jacobi pressure solve

| N | 20 sweeps | 40 sweeps | 80 sweeps |
|---|---|---|---|
| 256  | 0.9198 ms (1087 FPS) | 1.1852 ms (844 FPS) | 1.8041 ms (554 FPS) |
| 512  | 0.9602 ms (1041 FPS) | 1.2476 ms (802 FPS) | 1.7837 ms (561 FPS) |
| 1024 | 2.6204 ms (382 FPS)  | 3.3714 ms (297 FPS) | 4.8807 ms (205 FPS) |
| 2048 | 9.2189 ms (108 FPS)  | 11.8789 ms (84 FPS) | 17.2023 ms (58 FPS) |

**256 and 512 cost the same.** 512 has 4x the cells but is no slower — below
about 1024 the solver is bound by kernel-launch overhead and latency, not by
memory bandwidth, so the GPU is simply not full. From 512 to 1024 to 2048 the
time then scales close to the 4x cell count (2.70x, then 3.52x at 40 sweeps),
which is the signature of a properly bandwidth-bound kernel.

### Multigrid pressure solve (4 V-cycles, 6 levels, coarse=10)

| N | ms/frame | FPS |
|---|---|---|
| 256  | 3.8870 | 257 |
| 512  | 4.2007 | 238 |
| 1024 | 5.8483 | 171 |
| 2048 | 17.0615 | 59 |

Multigrid's cost is nearly flat from 256 to 1024 because it is dominated by the
fixed number of kernel launches down the V-cycle, not by the fine-grid work.
**The relative cost of multigrid therefore falls sharply as the grid grows**: it
is 3.4x the price of Jacobi-40 at N=512, but only 1.44x at N=2048 — while
delivering a 38x better divergence reduction. On large grids it is close to
free in relative terms.

---

## 4. Optimisations, measured

Four candidates were implemented and measured. **Two of the three
micro-optimisations made it slower**, and the only large win was algorithmic.

Kernel-level isolation (`kbench`, `N=2048`, 300 launches x 6 trials). "GB/s" is
compulsory traffic divided by min time — it counts each cell's loads and stores
once, so cache hits can push it above the 448 GB/s DRAM peak.

| kernel | min us | mean us | max us | GB/s |
|---|---|---|---|---|
| advect: manual bilerp | 170.56 | 174.82 | 188.25 | 393.5 |
| advect: tex2D + array copy | 260.49 | 263.43 | 274.58 | 322.0 |
| advect: tex2D (copy excluded) | 170.45 | 187.37 | 199.82 | 393.7 |
| jacobi: naive global | 129.00 | 136.06 | 143.74 | 390.2 |
| jacobi: shared-mem halo | 140.87 | 142.85 | 144.17 | 357.3 |
| gauss-seidel: red-black | 259.86 | 260.38 | 261.17 | 193.7 |

### Optimisation 1 — texture object with hardware bilinear filtering: **SLOWER (-2.9%)**

Replacing the manual 4-tap bilinear read with `tex2D<float>` on a
`cudaTextureObject_t` (`cudaFilterModeLinear`).

| N | manual bilerp | tex2D + copy | tex2D, copy excluded |
|---|---|---|---|
| 512  | 8.37 us | 19.33 us (**2.31x slower**) | 9.95 us (1.19x slower) |
| 1024 | 48.26 us | 73.62 us (**1.53x slower**) | 47.51 us (1.02x *faster*) |
| 2048 | 170.56 us | 260.49 us (**1.53x slower**) | 170.45 us (1.00x, tie) |

Whole-frame effect: **0.971x at N=2048** (12.335 ms vs 11.980 ms) — i.e. 2.9%
slower.

Why it lost: filtered texture reads on Turing require the data to live in a
`cudaArray`, and the simulation state lives in linear global memory that changes
every frame. So every advected field must be staged into the array with
`cudaMemcpy2DToArray` first, and that copy costs more than the filtering saves.
With the copy excluded the texture path is a dead tie with hand-written bilinear
at 1024 and 2048 — **the texture unit's filtering is not actually faster than
four L1-cached loads and three FMAs on this hardware**, so there was never
enough headroom to pay for the copy. This would only pay off if the whole solver
were restructured around surface/array storage.

### Optimisation 2 — shared-memory tiling with halo for the Jacobi stencil: **SLOWER (-2.7%)**

| N | naive global | shared + halo | result |
|---|---|---|---|
| 512  | 7.65 us | 8.96 us | **1.17x slower** |
| 1024 | 35.81 us | 37.23 us | **1.04x slower** |
| 2048 | 129.00 us | 140.87 us | **1.09x slower** |

Whole-frame effect: **0.973x at N=2048** (12.311 ms vs 11.980 ms).

Why it lost: this is the classic shared-memory-tiling result on modern hardware.
The 5-point stencil re-reads each neighbour at most 4 times, and those re-reads
already hit L1 — the naive kernel sustains **390 GB/s of the 448 GB/s peak, or
87% of theoretical bandwidth**, so there is at most 13% of headroom in the whole
kernel. Against that, the tiled version adds halo loads (a 32x8 tile loads 34x10
= 340 values for 256 outputs, **33% extra traffic**), a `__syncthreads()`, and 8
extra registers (24 vs 16). Its measured throughput drops to 357 GB/s. Shared
memory helps when a kernel is *not* already near bandwidth; here it was, so
tiling could only add overhead. Reported as a negative result.

### Optimisation 3 — red-black Gauss-Seidel: **converges 2x better, costs 2x more, net wash**

RBGS needs no second buffer and converges at twice Jacobi's rate per sweep —
confirmed exactly by the convergence table in section 1.3 (RBGS@320 == Jacobi@640).
But per equivalent sweep it is measurably slower:

| N | jacobi (2 sweeps' work) | red-black GS (1 iter = 2 half-sweeps) |
|---|---|---|
| 512  | 15.30 us | 14.31 us |
| 1024 | 71.62 us | 68.84 us |
| 2048 | 258.00 us | 259.86 us |

Effective bandwidth tells the story: **193.7 GB/s vs 390.2 GB/s at N=2048** — RBGS
achieves barely half the throughput of the Jacobi kernel. Each half-sweep touches
every *other* cell in a row, so a 32-thread warp requesting 32 stride-2 floats
pulls in 256 bytes of cache lines to use 128 bytes of it: **half of every memory
transaction is discarded.**

Net: the 2x convergence advantage and the 2x bandwidth penalty cancel almost
exactly, and RBGS is not worth using as a standalone solver on this hardware
(confirmed in section 1.2: 40 RBGS sweeps gives the same 1.1x divergence
reduction as 40 Jacobi sweeps and slightly *worse* mass drift). It is retained
only as the multigrid smoother, where its in-place operation and superior
smoothing per sweep matter more than raw bandwidth.

### Optimisation 4 — geometric multigrid V-cycle: **38x better divergence for 1.44x the cost at N=2048** (the only real win)

This is the optimisation that mattered, and it is algorithmic rather than a
memory trick. V-cycle with red-black GS smoothing, full-weighting restriction
(summed, which supplies the factor-4 operator rescaling for spacing 2h) and
prolongation by injection.

Against Jacobi-40 at N=2048: projection cost **5.742 ms -> 9.763 ms** (1.70x),
whole frame **11.916 ms -> 16.459 ms** (1.38x), in exchange for divergence
reduction **1.1x -> 38.4x** and dye mass drift **-92.07% -> -15.97%**.

To match multigrid's accuracy with plain Jacobi is not a matter of a few times
more sweeps: 640 Jacobi sweeps (8.744 ms, 14.6x the cost of 40) still only
achieves a 1.7x divergence reduction, against multigrid's 38x.

### Optimisation 4b — truncating the multigrid hierarchy: **1.87x faster for no accuracy loss**

The first working V-cycle coarsened all the way to an 8x8 grid and ran 40
smoothing sweeps at the bottom, costing 8.10 ms/frame at N=512 — far more than
expected from the arithmetic, which pointed at kernel-launch overhead on the
tiny coarse levels rather than real work. Measured (N=512, 4 V-cycles, whole
frame, and RMS|div| after projection over frames 60-199):

| levels | coarse sweeps | ms/frame | mean RMS abs div after |
|---|---|---|---|
| 3 | 40 | 5.432 | 2.054e-03 |
| 3 | 10 | 3.084 | 7.594e-03 |
| 4 | 40 | 6.227 | 3.665e-04 |
| 4 | 10 | 3.270 | 1.898e-03 |
| 5 | 40 | 6.560 | 5.957e-05 |
| 5 | 10 | 3.821 | 3.387e-04 |
| **6** | 40 | 6.924 | 1.633e-05 |
| **6** | **10** | **4.323** | **5.607e-05** |
| 7 (full) | 40 | 8.097 | 1.573e-05 |
| 7 (full) | 10 | 5.619 | 1.573e-05 |

Going from 6 levels to the full 7 costs **+1.18 ms/frame and buys nothing**
(1.633e-05 vs 1.573e-05 — within noise of each other). The bottom two levels are
16x16 and 8x8 grids: a few hundred cells each, where every kernel launch is pure
overhead and the actual arithmetic is free. Cutting the hierarchy at 6 levels
and the coarse solve to 10 sweeps takes **8.097 ms -> 4.323 ms, a 1.87x
speedup**, and costs only a 3.6x increase in an already-negligible residual
(1.57e-05 -> 5.61e-05, against a pre-projection value of 1.87e-03). These are the
shipped defaults.

**Honest caveat on this trade.** Truncation is free *for this flow*, but not in
general — and the manufactured-solution test in section 1.1 shows exactly where
it bites. With the full 7-level hierarchy, 4 V-cycles remove **99.991%** of the
smoothest mode (k=1); with the shipped 6-level hierarchy the same 4 cycles remove
only **78.4%**. Truncating the hierarchy removes precisely the coarsest grids,
which are the only ones that can see the domain-scale mode. The plume simulation
does not care because its divergence has little energy in the global mode and
the residual is re-solved every frame, but a problem driven at the domain scale
(a lid-driven cavity, or a long-time steady state) would need the full depth.
The 6-level default is tuned to this benchmark, not universally correct.

### Optimisation 5 — block-size sweep: **1.17x at N=512, nothing at N=2048**

Whole-frame ms, Jacobi-40, min of 4 trials.

| block | N=512 | N=2048 |
|---|---|---|
| 8x8    | 1.4673 | 12.3344 |
| 16x8   | 1.2749 | 12.7299 |
| 16x16  | 1.2580 | 12.0074 |
| 32x4   | 1.2851 | 12.0228 |
| 32x8   | 1.2678 | **11.8871** |
| 32x16  | 1.2788 | 12.5767 |
| 64x4   | 1.2548 | 12.0848 |
| 64x8   | 1.2623 | 12.0196 |
| 128x8  | **1.2513** | 12.0435 |

At N=512 only 8x8 is clearly bad (1.4673 vs 1.2513, **1.17x**) — a 64-thread
block wastes half of each 128-byte cache line per row and needs 4x the blocks.
Every configuration with a row of at least 16 threads is within 2% of every
other. At N=2048 the entire sweep spans **7.1%** (11.887 to 12.730 ms) with no
coherent pattern — 8x8 is no longer the worst, and the ordering does not
reproduce the N=512 ordering. Given this machine's ~5% run-to-run variation,
**that spread is largely noise and no block size is defensibly better than
another at N=2048**: the kernel is purely bandwidth-bound, so how the threads are
grouped stops mattering. `32x8` is kept as the default.

---

## 5. Occupancy and profiling

### Nsight Compute could not be used

`ncu` 2024.3.0 is installed at
`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\` and runs, but
every profiling attempt fails:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0.
```

This is NVIDIA's counter-permission lockout. It applies to `--set full` and
equally to the minimal `--section LaunchStats --section Occupancy`, so **no
Nsight section is available at all** on this machine. Lifting it requires
changing a driver-level security setting (or running elevated), which was out of
scope here. Recorded verbatim in `results/ncu_error.txt`.

**Achieved** occupancy, memory throughput counters, and the hardware bottleneck
classification for the Jacobi kernel are therefore *not* available, and no such
numbers are claimed anywhere in this document.

### Fallback A — theoretical occupancy via the CUDA occupancy API

`cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` give
the launch-configuration limits without any hardware counters. Full table in
`results/occupancy.txt`; the summary is that **every kernel reaches 100%
theoretical occupancy at every block size tested (8x8 through 128x8)**:

| kernel | registers | static smem | dyn smem @32x8 | blocks/SM @32x8 | theoretical occupancy |
|---|---|---|---|---|---|
| k_advect_manual | 20 | 0 | 0 | 4 | 100% |
| k_advect_tex | 16 | 0 | 0 | 4 | 100% |
| k_jacobi_naive | 16 | 0 | 0 | 4 | 100% |
| k_jacobi_shared | 24 | 0 | 1360 B | 4 | 100% |
| k_gs_redblack | 15 | 0 | 0 | 4 | 100% |
| k_residual | 18 | 0 | 0 | 4 | 100% |

Register pressure is tiny (15-24 of 255) and shared use is at most 5.2 KB of the
64 KB available, so occupancy is **never** the limiter — which is consistent with
the block-size sweep being flat and with the kernels sitting at 87% of peak
bandwidth. Note this is *theoretical* occupancy computed from the launch
configuration; it is not a measurement of achieved occupancy.

### Fallback B — `--ptxas-options=-v`

Emitted by `build.bat` on every build and saved to `results/ptxas.txt`. No
kernel spills:

```
k_advect_manual  : 20 registers, 0 barriers, 0 bytes stack, 0 spill stores/loads
k_advect_tex     : 16 registers, 0 barriers, 0 bytes stack, 0 spill stores/loads
k_jacobi_naive   : 16 registers, 0 barriers, 0 bytes stack, 0 spill stores/loads
k_jacobi_shared  : 24 registers, 1 barrier,  0 bytes stack, 0 spill stores/loads
k_gs_redblack    : 15 registers, 0 barriers, 0 bytes stack, 0 spill stores/loads
```

### Fallback C — bandwidth from first principles

With counters unavailable, effective bandwidth is computed as (compulsory bytes)
/ (measured min time), counting each cell's loads and stores once:

| kernel (N=2048) | GB/s | % of 448 GB/s peak |
|---|---|---|
| jacobi: naive global | 390.2 | **87.1%** |
| advect: manual bilerp | 393.5 | 87.8% |
| jacobi: shared-mem halo | 357.3 | 79.8% |
| gauss-seidel: red-black | 193.7 | 43.2% |

**Conclusion on the Jacobi bottleneck: it is memory-bandwidth-bound.** The
evidence is convergent rather than counter-based — 87% of theoretical DRAM
bandwidth, 16 registers with no spills, 100% theoretical occupancy, a flat
block-size sweep at large N, arithmetic intensity of roughly 6 flops per 12
bytes (~0.5 flop/byte, far below the ~35 flop/byte ridge point of this GPU), and
a shared-memory optimisation that *hurt* because there was no headroom to
recover. On arithmetic intensity specifically: the Jacobi kernel does ~6 flops
per 12 bytes of compulsory traffic (~0.5 flop/byte), while this GPU's ridge
point is 10.07 TFLOP/s FP32 / 448 GB/s = **~22 flop/byte** — the kernel sits
roughly 45x to the memory-bound side of the roofline knee. All five point the
same way. This conclusion would nonetheless be
better supported by an `ncu` memory-workload analysis, which this machine cannot
provide.

---

## Reproducing

```bat
build.bat
build\fluid.exe poisson    --n 512 --mg      :: manufactured-solution correctness
build\fluid.exe verify     --n 512 --mg      :: divergence + mass conservation
build\fluid.exe stages     --n 2048 --mg     :: stage breakdown
build\fluid.exe sweep      --bench 30 --trials 4
build\fluid.exe kbench     --reps 300 --trials 6
build\fluid.exe blocksweep --n 2048
build\fluid.exe occupancy
build\fluid.exe sim        --n 512 --mg --frames 300 --every 1 --out frames_raw
```

Raw captured output for every table is in `results/`.
