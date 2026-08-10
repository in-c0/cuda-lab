# cuda-lab

Two CUDA projects written to answer one question honestly: does GPU intuition built in GLSL transfer to CUDA?

Hardware for every number here: **RTX 2080 (Turing, sm_75, 46 SMs, 448 GB/s peak)**, driver 610.74, CUDA 12.6, MSVC 14.29. Every benchmark states its method; nothing is estimated. Nsight Compute could not run on this machine (`ERR_NVGPUCTRPERM` — counter access requires elevation), so **no achieved-occupancy or hardware-counter figure is claimed anywhere**; where a bottleneck is named, the reasoning is shown and labelled as inference.

## [fluid-sim](fluid-sim/) — stable fluids, and why the textbook version leaks

Semi-Lagrangian advection → implicit viscosity → pressure projection → no-slip walls, all on GPU.

The finding that shaped it: **Jacobi is a smoother, not a solver.** A manufactured-solution test (a curl-free field an exact projection must annihilate, so the survivor *is* the error) shows it removing 76% of grid-scale mode k=64 in 20 sweeps but **0.6% of domain-scale mode k=1 in 320**. That is why the standard implementation bleeds 92% of its dye mass. Adding a geometric multigrid V-cycle cuts max‖div‖ from 6.1e-01 to **1.1e-03** and mass drift to **-16%** — and 640 Jacobi sweeps, at 14.6× the cost, still cannot match 4 V-cycles.

Three optimisations that **failed**, reported because the reasons are the interesting part: texture memory with hardware bilinear filtering (2.9% slower — filtered reads need a `cudaArray`, so state must be staged each frame, and the texture unit is only a dead tie with 4 L1 loads + 3 FMAs), shared-memory tiling of the Jacobi stencil (2.7% slower — the naive kernel already sustains **390 of 448 GB/s, 87% of peak**, so tiling's 33% halo traffic has nothing to buy), and red-black Gauss-Seidel (2× convergence exactly cancelled by stride-2 access halving effective bandwidth).

## [cppn-port](cppn-port/) — the same renderer, GLSL → CUDA

A port of [GlassOS](https://github.com/in-c0/glassos-webgl)'s per-pixel CPPN evaluation: six organisms × six hidden nodes per fragment, genome-selected activations, analytic zero-contour extraction, chromatic dispersion, Fresnel refraction, fBm caustics.

Correctness first: **1 LSB** against an independently transcribed CPU reference (SSIM 1.000000), then **SSIM 0.999965** against the real shader, fed byte-identical genome data through a WebGL2 harness.

Then the negative result: `__constant__` memory — the textbook analogue of GLSL uniforms — is **5–10% slower** than plain global loads here. The early-out is the entire memory-side win, and the SASS says why: ~690 FFMA against ~34 loads, a 1536-byte working set that never leaves L1, and 9.6 GB/s of store traffic against 448 theoretical. It was compute-bound all along. Best variant 0.862 ms at 1920×1080 (1.91× naive), which is **a tie within noise** against the original shader timed with `EXT_disjoint_timer_query_webgl2` — no speedup is claimed.

## What the two have in common

Both were measured before they were believed, both found their own methodology bugs (GPU clock drift faking a 33% regression; a manufactured-solution test that measured nothing until its boundary conditions were fixed; desktop compositing producing 3× outliers), and in both the most useful result was the optimisation that did not work.
