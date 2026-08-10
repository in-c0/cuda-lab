# Real-time 2D fluid simulation in CUDA

A Stam-style stable-fluids solver running entirely on the GPU, written from
scratch in CUDA C++. Renders a buoyant dye plume at **512x512 in 4.2 ms/frame
(238 FPS)** and **2048x2048 in 17.1 ms/frame (59 FPS)** on an RTX 2080.

![Plume at frame 120](frames/frame_0120.png)
![Plume at frame 299](frames/frame_0299.png)

*Left: frame 120, the buoyant head rolling up into a counter-rotating vortex
pair. Right: frame 299, after the plume has reached the ceiling and begun to
interact with the walls. 512x512, dt = 1 cell, no-slip boundaries. Full 300-frame
sequence: `plume_512.mp4`.*

Everything runs headless — no window, no OpenGL, no CUDA-GL interop. The solver
writes a PPM sequence and a CSV diagnostic log, so correctness is checkable from
files on a machine with no display.

---

## What it simulates

Incompressible Navier-Stokes on a uniform 2D grid, with a passive dye tracer
that is advected by the flow and drives the flow through Boussinesq buoyancy.
The scenario is a classic rising plume: dye is injected at the bottom with a
wobbling source, buoyancy accelerates it upward, and shear rolls the head into a
vortex pair.

## The algorithm

Cell-centred collocated grid with a one-cell ghost border. Each timestep:

1. **Sources** — inject dye and momentum; apply Boussinesq buoyancy `v += k*dye`.
2. **Diffusion** — implicit (backward-Euler) viscosity, `(I - nu*dt*L)u = u_old`,
   solved with Jacobi sweeps.
3. **Advection** — semi-Lagrangian: backtrace each cell centre one timestep along
   the velocity field and interpolate the source field there. Because the
   backtrace point is clamped into the domain and the result is an interpolation
   (never an extrapolation), the scheme is **unconditionally stable — there is no
   CFL limit on dt**. The shipped run uses dt = 1 full cell, which an explicit
   scheme could not survive.
4. **Pressure projection** — Helmholtz-Hodge. Compute `div(u)`, solve
   `laplacian(p) = div(u)`, subtract `grad(p)`. This is what enforces
   incompressibility, and it is where essentially all the runtime goes.
5. **Boundaries** — **no-slip** walls. Velocity ghost cells mirror the interior
   with a sign flip on the wall-normal component, so the velocity at the wall
   face is zero for both components. Pressure and dye use a Neumann
   (zero-gradient) copy. Re-imposed after every relaxation sweep.

Velocity is stored **structure-of-arrays** (separate `u` and `v` planes) so every
stencil read is fully coalesced, with ping-pong buffers for the stages that
cannot be done in place.

## Proof that it is correct

Two independent checks, both measured, both in [`bench.md`](bench.md).

**1. Manufactured solution.** The field `u = grad(cos(k*pi*x)cos(k*pi*y))` is a
pure gradient, so an exact projection must reduce it to exactly zero — whatever
survives *is* the solver error. Multigrid removes **99.99%** of modes 1-16. The
remaining ~96% ceiling at k=64 is discretisation error (8 cells per wavelength),
not solver error: both solvers converge to the identical floor, which is the
signature of a discretisation limit rather than a bug.

**2. Divergence and mass in the live simulation.** 300 frames, injection stopped
at frame 60 so mass conservation is measured on a closed system:

| | max abs div before projection | max abs div after | RMS reduction | dye mass drift |
|---|---|---|---|---|
| Jacobi, 40 sweeps | 8.89e-01 | 6.14e-01 | 1.1x | -92.1% |
| **Multigrid, 4 V-cycles** | 6.54e-02 | **1.12e-03** | **38.4x** | **-16.0%** |

Per-frame numbers for the shipped run are in `results/sim_log.csv`.

## What was optimised, and by how much

Measured on an RTX 2080 (sm_75), CUDA 12.6, `-O3 -arch=sm_75`, CUDA events,
warm-up excluded, minimum of 3-6 trials of >= 30 frames. **Two of the four
micro-optimisations made it slower and are reported as such.**

| # | optimisation | result at N=2048 | verdict |
|---|---|---|---|
| 1 | `cudaTextureObject_t` + HW bilinear for the backtrace | 12.335 vs 11.980 ms | **2.9% SLOWER** |
| 2 | Shared-memory tiling with halo for the Jacobi stencil | 12.311 vs 11.980 ms | **2.7% SLOWER** |
| 3 | Red-black Gauss-Seidel instead of Jacobi | 2x convergence, 2x cost | **net wash** |
| 4 | **Geometric multigrid V-cycle** | 1.38x cost, **38x less divergence** | **the win** |
| 4b | Truncating the MG hierarchy 7 -> 6 levels | 8.097 -> 4.323 ms | **1.87x faster, free** |
| 5 | Block-size sweep (8x8 ... 128x8) | 1.17x at N=512, nothing at N=2048 | minor |

The three interesting findings:

- **Textures lost because of the staging copy.** Filtered texture reads on Turing
  need the data in a `cudaArray`, but simulation state lives in linear memory and
  changes every frame, so each advected field must be `cudaMemcpy2DToArray`'d
  first. Excluding that copy, hardware bilinear is a *dead tie* with four
  L1-cached loads and three FMAs (170.45 us vs 170.56 us at N=2048) — so there
  was never enough headroom to pay for the copy.

- **Shared memory lost because there was no headroom to recover.** The naive
  Jacobi kernel already sustains **390 GB/s of the 448 GB/s peak (87%)**. Tiling
  adds 33% extra halo traffic, a `__syncthreads()`, and 8 more registers, and
  drops throughput to 357 GB/s. Shared-memory tiling pays off when a stencil is
  *not* already bandwidth-saturated; this one is.

- **The only large win was algorithmic, not a memory trick.** Jacobi is a
  *smoother*, not a solver: it removes 76% of a k=64 error mode in 20 sweeps but
  0.6% of a k=1 mode in 320 sweeps. That is why the textbook 20-40 iteration
  Stam solver leaks 92% of its dye mass — the flow stays measurably
  compressible. Multigrid attacks exactly the modes Jacobi cannot see, and 640
  Jacobi sweeps (14.6x the cost) still cannot match 4 V-cycles.

## Performance

Whole-frame, minimum of 4 trials of 30 frames, block 32x8:

| N (interior) | Jacobi-20 | Jacobi-40 | Jacobi-80 | Multigrid (4 V-cycles) |
|---|---|---|---|---|
| 256  | 0.920 ms | 1.185 ms | 1.804 ms | 3.887 ms |
| 512  | 0.960 ms | 1.248 ms | 1.784 ms | 4.201 ms |
| 1024 | 2.620 ms | 3.371 ms | 4.881 ms | 5.848 ms |
| 2048 | 9.219 ms | 11.879 ms | 17.202 ms | 17.062 ms |

Stage breakdown at N=2048, Jacobi-40 (11.916 ms/frame):

| stage | ms | % |
|---|---|---|
| sources + buoyancy | 0.191 | 1.6% |
| diffusion (Jacobi) | 5.527 | 46.3% |
| advect velocity | 0.264 | 2.2% |
| **projection (Jacobi)** | **5.742** | **48.1%** |
| boundary | 0.027 | 0.2% |
| advect dye | 0.186 | 1.6% |

**The elliptic solves are 94% of the frame; both advections together are under
4%.** Anyone optimising this solver should ignore advection entirely — which is
what optimisation 4 does.

Note that N=256 and N=512 cost the same despite a 4x difference in cells: below
~1024 the solver is launch-overhead- and latency-bound, not bandwidth-bound. From
512 upward it scales with the cell count as a bandwidth-bound kernel should.

## Profiling

**Nsight Compute could not be used on this machine.** `ncu` 2024.3.0 is installed
and launches, but every profiling attempt — including the minimal
`--section LaunchStats` — fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0.
```

Lifting that requires a driver-level security setting change, which was out of
scope. **No achieved-occupancy or hardware-counter number is claimed anywhere in
this repository.** The failure is recorded verbatim in `results/ncu_error.txt`.

Three counter-free fallbacks were used instead (details in `bench.md`):

- **CUDA occupancy API** — every kernel reaches **100% theoretical occupancy** at
  every block size tested; 15-24 registers of 255, at most 5.2 KB of 64 KB shared.
  Occupancy is never the limiter.
- **`--ptxas-options=-v`** — no register spills in any kernel (`results/ptxas.txt`).
- **Bandwidth from first principles** — compulsory bytes / measured min time.

Conclusion on the Jacobi bottleneck: **memory-bandwidth-bound**, supported by five
convergent lines of evidence — 87% of theoretical DRAM bandwidth, no spills, 100%
theoretical occupancy, a flat block-size sweep at large N, and ~0.5 flop/byte
against this GPU's ~22 flop/byte roofline ridge point. It is a solid inference
but *not* the same thing as an `ncu` memory-workload analysis, which this machine
cannot provide.

## Honest limitations

- **2D only.** The algorithm extends to 3D directly, but nothing here has been
  run in 3D and the multigrid transfer operators would need rewriting.
- **Single GPU, single stream.** No multi-GPU, no stream overlap, no graphs.
  CUDA Graphs would likely be a real win given how launch-bound the small grids
  and the coarse multigrid levels are — measured but not implemented.
- **`float` precision throughout.** Adequate here; a long-running or
  tighter-tolerance solve would want to at least accumulate the pressure residual
  in double.
- **Semi-Lagrangian advection is not conservative.** This is the cause of the
  remaining **-16% dye mass drift over 240 frames** under multigrid, and it is
  *not* a pressure-solver problem — no amount of extra projection fixes it.
  MacCormack/BFECC or a conservative flux form would; neither is implemented.
- **Semi-Lagrangian advection is also strongly dissipative.** Small-scale vortices
  smear out over time. This is the classic Stam trade: unconditional stability
  bought with numerical viscosity.
- **No free surface, no multiphase, no obstacles.** Closed box with no-slip walls
  only. There is no level set, no solid-boundary rasterisation, no two-way
  coupling.
- **Multigrid uses prolongation by injection**, not bilinear interpolation. It is
  cheaper and the post-smoothing sweep cleans up the blockiness, but it breaks the
  variational property and costs some convergence rate. Bilinear prolongation was
  not measured.
- **The 6-level hierarchy default is tuned to this benchmark, not universal.**
  It is free for the plume, but it costs real accuracy on the domain-scale mode
  (4 V-cycles remove 99.99% of mode k=1 at full depth vs 78% truncated). A
  lid-driven cavity or a steady-state problem would need the full hierarchy.
- **Timings come from a GPU that also drives the desktop.** Background
  compositing intermittently steals SM time — an outlier of 35 ms was observed for
  a configuration whose clean time is 4.3 ms. Minimum-of-N-trials and a
  monotonicity sanity check mitigate this, but absolute numbers should be read as
  +/-5%. A/B ratios measured back-to-back in one process are tighter.
- **The pressure solve is Neumann on all four walls and therefore singular** (the
  solution is defined only up to a constant). This is harmless for the projection,
  since only `grad(p)` is used, but the constant mode is not explicitly projected
  out and would matter if `p` itself were ever consumed.

## Building and running

Requires the CUDA Toolkit and MSVC. `build.bat` locates `vcvars64.bat`
automatically if `cl.exe` is not already on PATH.

```bat
build.bat
```

```bat
build\fluid.exe sim --n 512 --mg --frames 300 --every 1 --out frames_raw
build\fluid.exe verify   --n 512 --mg      :: divergence + mass conservation
build\fluid.exe poisson  --n 512 --mg      :: manufactured-solution correctness
build\fluid.exe converge --n 512           :: solver convergence vs cost
build\fluid.exe stages   --n 2048 --mg     :: per-stage timing breakdown
build\fluid.exe sweep                      :: grid x iteration-count sweep
build\fluid.exe kbench                     :: kernel-level A/B of the variants
build\fluid.exe blocksweep --n 2048        :: block-size sweep
build\fluid.exe occupancy                  :: registers / smem / theoretical occupancy
```

Useful flags: `--n` (interior resolution), `--iters` (pressure sweeps), `--mg`
(multigrid), `--rb` (red-black GS), `--tex` (texture advection), `--shared`
(shared-memory Jacobi), `--cycles` / `--levels` / `--coarse` / `--smooth`
(multigrid tuning), `--bx` / `--by` (block size), `--trials`, `--warmup`.

## Files

| | |
|---|---|
| `fluid.cuh` | shared declarations |
| `fluid_kernels.cu` | all device code — advection, Jacobi/RBGS, projection, multigrid transfers, boundaries, diagnostics, colormap |
| `main.cu` | solver driver, benchmark harness, all CLI modes |
| `build.bat` | nvcc build, `-arch=sm_75` |
| `bench.md` | every measured table, with methodology |
| `frames/` | representative PNG frames |
| `plume_512.mp4` | full 300-frame sequence |
| `results/` | raw captured output for every table in `bench.md` |
