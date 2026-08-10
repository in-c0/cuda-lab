// fluid.cuh -- shared declarations for the CUDA 2D stable-fluids solver.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

// Cell-centred, collocated grid of N x N cells INCLUDING a one-cell ghost
// border, so the interior is (N-2) x (N-2). Index (i,j) -> j*N + i.
#define IX(i, j, N) ((j) * (N) + (i))

// Which quantity a boundary kernel is fixing up.
//   BND_SCALAR : Neumann copy (dye, pressure)
//   BND_U      : x-velocity, negated at walls (no-slip)
//   BND_V      : y-velocity, negated at walls (no-slip)
enum BndKind { BND_SCALAR = 0, BND_U = 1, BND_V = 2 };

// ---- Advection variants (optimisation A) ----------------------------------
void launch_advect_manual(float* dst, const float* src, const float* u,
                          const float* v, int N, float dt, dim3 blk);
void launch_advect_tex(float* dst, cudaTextureObject_t srcTex, const float* u,
                       const float* v, int N, float dt, dim3 blk);

// ---- Jacobi variants (optimisation B) -------------------------------------
// Generic Jacobi sweep for  (x - alpha*Laplacian(x)) form used by both the
// diffusion solve and the pressure solve:  x_new = (b + alpha*sum_nb) * rBeta
void launch_jacobi_naive(float* xnew, const float* x, const float* b, int N,
                         float alpha, float rBeta, dim3 blk);
void launch_jacobi_shared(float* xnew, const float* x, const float* b, int N,
                          float alpha, float rBeta, dim3 blk);
// Red-black Gauss-Seidel: in-place, two half-sweeps per iteration. Same
// arithmetic as Jacobi but uses already-updated neighbours, so it converges
// roughly twice as fast per sweep. `parity` selects the colour.
void launch_gs_redblack(float* x, const float* b, int N, float alpha,
                        float rBeta, int parity, dim3 blk);

// ---- Remaining stages ------------------------------------------------------
void launch_divergence(float* div, float* p, const float* u, const float* v,
                       int N, dim3 blk);
void launch_subtract_gradient(float* u, float* v, const float* p, int N,
                              dim3 blk);
void launch_boundary(float* f, int N, int kind);
void launch_inject(float* dye, float* u, float* v, int N, float cx, float cy,
                   float radius, float dyeAmt, float fx, float fy, dim3 blk);
void launch_buoyancy(float* v, const float* dye, int N, float k, dim3 blk);

// ---- Diagnostics -----------------------------------------------------------
// Max |divergence| over the interior (single float in device memory).
void launch_max_divergence(const float* u, const float* v, int N,
                           unsigned int* d_maxbits, dim3 blk);
void print_occupancy_table(const int* bx, const int* by, int nb);

// ---- Geometric multigrid (optimisation D, algorithmic) --------------------
void launch_residual(float* r, const float* p, const float* b, int N, dim3 blk);
void launch_restrict(float* dst, const float* src, int Nc, int Nf, dim3 blk);
void launch_prolong_add(float* dst, const float* src, int Nf, int Nc, dim3 blk);

// Manufactured solution: fill (u,v) with a pure gradient field of mode k.
// An exact projection must reduce it to zero.
void launch_set_pure_gradient(float* u, float* v, int N, float k, dim3 blk);
// Sum of a^2 + b^2 over the interior.
void launch_sumsq_field2(const float* a, const float* b, int N, double* d_sum,
                         dim3 blk);
// Sum of divergence^2 over the interior, for an RMS norm.
void launch_sumsq_divergence(const float* u, const float* v, int N,
                             double* d_sum, dim3 blk);
// Sum of dye over the interior (double in device memory).
void launch_sum_field(const float* f, int N, double* d_sum, dim3 blk);
// Dye -> RGB8 with a black-body-ish colormap.
void launch_colormap(unsigned char* rgb, const float* dye, const float* u,
                     const float* v, int N, dim3 blk);
