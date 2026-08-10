// fluid_kernels.cu -- all device code for the 2D stable-fluids solver.
#include "fluid.cuh"

// ---------------------------------------------------------------------------
// Semi-Lagrangian advection.
//
// Backtrace the cell centre one timestep along the velocity field and sample
// the source field there. Unconditionally stable: the sample point is clamped
// into the domain, so no CFL restriction on dt.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float bilerp(const float* f, int N, float x,
                                        float y) {
    int i0 = (int)floorf(x), j0 = (int)floorf(y);
    float sx = x - i0, sy = y - j0;
    int i1 = i0 + 1, j1 = j0 + 1;
    float a = f[IX(i0, j0, N)], b = f[IX(i1, j0, N)];
    float c = f[IX(i0, j1, N)], d = f[IX(i1, j1, N)];
    return (a + sx * (b - a)) + sy * ((c + sx * (d - c)) - (a + sx * (b - a)));
}

// Variant A1: manual bilinear interpolation out of global memory.
__global__ void k_advect_manual(float* __restrict__ dst,
                                const float* __restrict__ src,
                                const float* __restrict__ u,
                                const float* __restrict__ v, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    float x = i - dt * u[idx];
    float y = j - dt * v[idx];
    x = fminf(fmaxf(x, 0.5f), N - 1.5f);
    y = fminf(fmaxf(y, 0.5f), N - 1.5f);
    dst[idx] = bilerp(src, N, x, y);
}

// Variant A2: the same backtrace, but the interpolation is done by the
// texture unit (cudaFilterModeLinear) instead of by ALU + 4 global loads.
__global__ void k_advect_tex(float* __restrict__ dst, cudaTextureObject_t src,
                             const float* __restrict__ u,
                             const float* __restrict__ v, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    float x = i - dt * u[idx];
    float y = j - dt * v[idx];
    x = fminf(fmaxf(x, 0.5f), N - 1.5f);
    y = fminf(fmaxf(y, 0.5f), N - 1.5f);
    // +0.5 puts us on the texel centre convention of tex2D.
    dst[idx] = tex2D<float>(src, x + 0.5f, y + 0.5f);
}

// ---------------------------------------------------------------------------
// Jacobi relaxation, shared by the diffusion solve and the pressure solve.
//   x_new(i,j) = ( b(i,j) + alpha * sum_of_4_neighbours ) * rBeta
// Diffusion : alpha = nu*dt/h^2,  rBeta = 1/(1+4*alpha),  b = x_prev_timestep
// Pressure  : alpha = 1,          rBeta = 1/4,            b = -div
// ---------------------------------------------------------------------------

// Variant B1: naive, five global loads per cell (relies purely on L1/L2).
__global__ void k_jacobi_naive(float* __restrict__ xnew,
                               const float* __restrict__ x,
                               const float* __restrict__ b, int N, float alpha,
                               float rBeta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    float s = x[idx - 1] + x[idx + 1] + x[idx - N] + x[idx + N];
    xnew[idx] = (b[idx] + alpha * s) * rBeta;
}

// Variant B2: shared-memory tile with a one-cell halo, so interior neighbours
// are read once from global memory instead of (up to) five times.
__global__ void k_jacobi_shared(float* __restrict__ xnew,
                                const float* __restrict__ x,
                                const float* __restrict__ b, int N, float alpha,
                                float rBeta) {
    extern __shared__ float tile[];
    const int tw = blockDim.x + 2;
    const int lx = threadIdx.x + 1, ly = threadIdx.y + 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Clamped global coordinates so halo loads never leave the allocation.
    int ic = min(i, N - 1), jc = min(j, N - 1);
    tile[ly * tw + lx] = x[IX(ic, jc, N)];
    if (threadIdx.x == 0) tile[ly * tw] = x[IX(max(i - 1, 0), jc, N)];
    if (threadIdx.x == blockDim.x - 1)
        tile[ly * tw + lx + 1] = x[IX(min(i + 1, N - 1), jc, N)];
    if (threadIdx.y == 0) tile[lx] = x[IX(ic, max(j - 1, 0), N)];
    if (threadIdx.y == blockDim.y - 1)
        tile[(ly + 1) * tw + lx] = x[IX(ic, min(j + 1, N - 1), N)];
    __syncthreads();

    if (i >= N - 1 || j >= N - 1) return;
    float s = tile[ly * tw + lx - 1] + tile[ly * tw + lx + 1] +
              tile[(ly - 1) * tw + lx] + tile[(ly + 1) * tw + lx];
    xnew[IX(i, j, N)] = (b[IX(i, j, N)] + alpha * s) * rBeta;
}

// Variant B3 (algorithmic, not a micro-optimisation): red-black Gauss-Seidel.
// Each half-sweep updates one colour in place, reading only the other colour,
// so it is race-free without a second buffer AND sees neighbours already
// updated this iteration. Threads are packed along x at stride 2 so that a
// warp still writes a contiguous 128-byte-per-two-rows pattern.
__global__ void k_gs_redblack(float* __restrict__ x, const float* __restrict__ b,
                              int N, float alpha, float rBeta, int parity) {
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N - 1) return;
    int i = 2 * tx + 1 + ((j + parity) & 1);
    if (i >= N - 1) return;
    int idx = IX(i, j, N);
    float s = x[idx - 1] + x[idx + 1] + x[idx - N] + x[idx + N];
    x[idx] = (b[idx] + alpha * s) * rBeta;
}

// ---------------------------------------------------------------------------
// Geometric multigrid support.
//
// The discrete operator (unscaled by h) is  A p = 4p - sum_of_4_neighbours,
// and the systems above are all of the form  A p = b. Jacobi / red-black GS
// only damp the HIGH-frequency error modes, so they stall on smooth error.
// Multigrid fixes that by moving smooth error to a coarser grid where it looks
// high-frequency again.
// ---------------------------------------------------------------------------

__global__ void k_residual(float* __restrict__ r, const float* __restrict__ p,
                           const float* __restrict__ b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    float s = p[idx - 1] + p[idx + 1] + p[idx - N] + p[idx + N];
    r[idx] = b[idx] - (4.0f * p[idx] - s);
}

// Cell-centred restriction. Each coarse cell covers a 2x2 block of fine cells.
// We SUM (rather than average) the four residuals, which is exactly the factor
// of 4 needed to rescale the unscaled operator for a grid spacing of 2h.
__global__ void k_restrict(float* __restrict__ dst,
                           const float* __restrict__ src, int Nc, int Nf) {
    int I = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int J = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (I >= Nc - 1 || J >= Nc - 1) return;
    int i = 2 * I - 1, j = 2 * J - 1;
    dst[IX(I, J, Nc)] = src[IX(i, j, Nf)] + src[IX(i + 1, j, Nf)] +
                        src[IX(i, j + 1, Nf)] + src[IX(i + 1, j + 1, Nf)];
}

// Prolongation by injection: each coarse correction is added to its four fine
// children. Cheap; the post-smoothing sweep removes the resulting blockiness.
__global__ void k_prolong_add(float* __restrict__ dst,
                              const float* __restrict__ src, int Nf, int Nc) {
    int I = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int J = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (I >= Nc - 1 || J >= Nc - 1) return;
    float c = src[IX(I, J, Nc)];
    int i = 2 * I - 1, j = 2 * J - 1;
    dst[IX(i, j, Nf)] += c;
    dst[IX(i + 1, j, Nf)] += c;
    dst[IX(i, j + 1, Nf)] += c;
    dst[IX(i + 1, j + 1, Nf)] += c;
}

// ---------------------------------------------------------------------------
// Pressure projection: Helmholtz-Hodge. Compute div(u), solve Laplacian(p) =
// div(u) with Jacobi, then subtract grad(p). Grid spacing h = 1 cell.
// ---------------------------------------------------------------------------

__global__ void k_divergence(float* __restrict__ div, float* __restrict__ p,
                             const float* __restrict__ u,
                             const float* __restrict__ v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    // b = -div(u), matching the (b + sum)/4 Jacobi form above.
    div[idx] = -0.5f * (u[idx + 1] - u[idx - 1] + v[idx + N] - v[idx - N]);
    p[idx] = 0.0f;
}

__global__ void k_subtract_gradient(float* __restrict__ u,
                                    float* __restrict__ v,
                                    const float* __restrict__ p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    u[idx] -= 0.5f * (p[idx + 1] - p[idx - 1]);
    v[idx] -= 0.5f * (p[idx + N] - p[idx - N]);
}

// ---------------------------------------------------------------------------
// Boundaries: NO-SLIP walls. Velocity ghost cells mirror the interior with a
// sign flip on the wall-normal component (so the wall-face value is zero for
// both components); pressure and dye use a Neumann (zero-gradient) copy.
// ---------------------------------------------------------------------------
__global__ void k_boundary(float* __restrict__ f, int N, int kind) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    if (t >= 1 && t <= N - 2) {
        float sx = (kind == BND_U) ? -1.0f : 1.0f;  // left/right walls
        float sy = (kind == BND_V) ? -1.0f : 1.0f;  // bottom/top walls
        f[IX(0, t, N)] = sx * f[IX(1, t, N)];
        f[IX(N - 1, t, N)] = sx * f[IX(N - 2, t, N)];
        f[IX(t, 0, N)] = sy * f[IX(t, 1, N)];
        f[IX(t, N - 1, N)] = sy * f[IX(t, N - 2, N)];
    }
    if (t == 0) {
        f[IX(0, 0, N)] = 0.5f * (f[IX(1, 0, N)] + f[IX(0, 1, N)]);
        f[IX(0, N - 1, N)] = 0.5f * (f[IX(1, N - 1, N)] + f[IX(0, N - 2, N)]);
        f[IX(N - 1, 0, N)] = 0.5f * (f[IX(N - 2, 0, N)] + f[IX(N - 1, 1, N)]);
        f[IX(N - 1, N - 1, N)] =
            0.5f * (f[IX(N - 2, N - 1, N)] + f[IX(N - 1, N - 2, N)]);
    }
}

// ---------------------------------------------------------------------------
// Sources.
// ---------------------------------------------------------------------------
__global__ void k_inject(float* __restrict__ dye, float* __restrict__ u,
                         float* __restrict__ v, int N, float cx, float cy,
                         float radius, float dyeAmt, float fx, float fy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    float dx = i - cx, dy = j - cy;
    float r2 = dx * dx + dy * dy;
    float g = __expf(-r2 / (radius * radius));
    if (g < 1e-4f) return;
    int idx = IX(i, j, N);
    dye[idx] = fminf(dye[idx] + dyeAmt * g, 1.5f);
    u[idx] += fx * g;
    v[idx] += fy * g;
}

// Boussinesq buoyancy: dye acts as a light tracer pushed upward.
__global__ void k_buoyancy(float* __restrict__ v, const float* __restrict__ dye,
                           int N, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    v[idx] += k * dye[idx];
}

// ---------------------------------------------------------------------------
// Diagnostics.
// ---------------------------------------------------------------------------
__global__ void k_max_divergence(const float* __restrict__ u,
                                 const float* __restrict__ v, int N,
                                 unsigned int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N - 1 || j >= N - 1) return;
    int idx = IX(i, j, N);
    float d = fabsf(0.5f * (u[idx + 1] - u[idx - 1] + v[idx + N] - v[idx - N]));
    // Non-negative floats compare identically to their bit patterns as uints.
    atomicMax(out, __float_as_uint(d));
}

// Manufactured-solution setup: u = grad(phi) with phi = cos(k*pi*x)*cos(k*pi*y)
// on the unit square, k a positive integer. This field is a PURE gradient
// (curl-free), so an exact Helmholtz-Hodge projection must annihilate it
// completely and whatever velocity survives IS the solver error.
//
// cos (not sin) matters: it makes u.n vanish on all four walls, so the field
// is compatible with the no-flux Neumann pressure boundary. With sin, part of
// the field is unremovable by construction and the test measures nothing.
__global__ void k_set_pure_gradient(float* __restrict__ u,
                                    float* __restrict__ v, int N, float k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float h = 1.0f / (N - 2);
    float x = (i - 0.5f) * h, y = (j - 0.5f) * h;
    float w = k * 3.14159265358979f;
    u[IX(i, j, N)] = -w * sinf(w * x) * cosf(w * y) * h;
    v[IX(i, j, N)] = -w * cosf(w * x) * sinf(w * y) * h;
}

__global__ void k_sumsq_field2(const float* __restrict__ a,
                               const float* __restrict__ b, int N,
                               double* out) {
    __shared__ double sm[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double val = 0.0;
    if (i < N - 1 && j < N - 1) {
        int idx = IX(i, j, N);
        val = (double)a[idx] * a[idx] + (double)b[idx] * b[idx];
    }
    sm[tid] = val;
    __syncthreads();
    for (int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sm[0]);
}

__global__ void k_sumsq_divergence(const float* __restrict__ u,
                                   const float* __restrict__ v, int N,
                                   double* out) {
    __shared__ double sm[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double val = 0.0;
    if (i < N - 1 && j < N - 1) {
        int idx = IX(i, j, N);
        double d = 0.5 * (u[idx + 1] - u[idx - 1] + v[idx + N] - v[idx - N]);
        val = d * d;
    }
    sm[tid] = val;
    __syncthreads();
    for (int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sm[0]);
}

__global__ void k_sum_field(const float* __restrict__ f, int N, double* out) {
    __shared__ double sm[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double val = (i < N - 1 && j < N - 1) ? (double)f[IX(i, j, N)] : 0.0;
    sm[tid] = val;
    __syncthreads();
    for (int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sm[0]);
}

__device__ __forceinline__ float sat(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__global__ void k_colormap(unsigned char* __restrict__ rgb,
                           const float* __restrict__ dye,
                           const float* __restrict__ u,
                           const float* __restrict__ v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    int idx = IX(i, j, N);
    float d = sat(dye[idx]);
    float spd = sqrtf(u[idx] * u[idx] + v[idx] * v[idx]);
    // Black-body ramp on dye, with a faint cool tint from speed so the
    // velocity field is visible in the otherwise empty regions.
    float r = sat(1.6f * d);
    float g = sat(1.6f * d - 0.55f);
    float b = sat(2.2f * d - 1.4f) + 0.35f * sat(spd * 0.35f);
    // Image row 0 is the TOP of the PPM, but j = 0 is the bottom wall.
    int o = 3 * IX(i, N - 1 - j, N);
    rgb[o + 0] = (unsigned char)(255.f * sat(r));
    rgb[o + 1] = (unsigned char)(255.f * sat(g));
    rgb[o + 2] = (unsigned char)(255.f * sat(b));
}

// ---------------------------------------------------------------------------
// Launchers.
// ---------------------------------------------------------------------------
static inline dim3 grid_for(int N, dim3 blk) {
    return dim3((N - 2 + blk.x - 1) / blk.x, (N - 2 + blk.y - 1) / blk.y);
}

void launch_advect_manual(float* dst, const float* src, const float* u,
                          const float* v, int N, float dt, dim3 blk) {
    k_advect_manual<<<grid_for(N, blk), blk>>>(dst, src, u, v, N, dt);
}
void launch_advect_tex(float* dst, cudaTextureObject_t srcTex, const float* u,
                       const float* v, int N, float dt, dim3 blk) {
    k_advect_tex<<<grid_for(N, blk), blk>>>(dst, srcTex, u, v, N, dt);
}
void launch_jacobi_naive(float* xnew, const float* x, const float* b, int N,
                         float alpha, float rBeta, dim3 blk) {
    k_jacobi_naive<<<grid_for(N, blk), blk>>>(xnew, x, b, N, alpha, rBeta);
}
void launch_jacobi_shared(float* xnew, const float* x, const float* b, int N,
                          float alpha, float rBeta, dim3 blk) {
    size_t sh = (size_t)(blk.x + 2) * (blk.y + 2) * sizeof(float);
    k_jacobi_shared<<<grid_for(N, blk), blk, sh>>>(xnew, x, b, N, alpha, rBeta);
}
void launch_gs_redblack(float* x, const float* b, int N, float alpha,
                        float rBeta, int parity, dim3 blk) {
    // Half the columns per half-sweep.
    dim3 g((((N - 2 + 1) / 2) + blk.x - 1) / blk.x,
           (N - 2 + blk.y - 1) / blk.y);
    k_gs_redblack<<<g, blk>>>(x, b, N, alpha, rBeta, parity);
}
// Nsight Compute cannot run on this machine without elevated GPU-counter
// permissions, so we report what the CUDA occupancy API can tell us without
// hardware counters: per-kernel register and shared-memory usage, and the
// THEORETICAL occupancy the launch configuration allows. This is a real
// measurement of the launch configuration, not of achieved occupancy.
void print_occupancy_table(const int* bx, const int* by, int nb) {
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, 0);
    int maxWarps = pr.maxThreadsPerMultiProcessor / 32;
    struct K { const char* name; const void* fn; bool dynSmem; };
    K ks[] = {
        {"k_advect_manual", (const void*)k_advect_manual, false},
        {"k_advect_tex", (const void*)k_advect_tex, false},
        {"k_jacobi_naive", (const void*)k_jacobi_naive, false},
        {"k_jacobi_shared", (const void*)k_jacobi_shared, true},
        {"k_gs_redblack", (const void*)k_gs_redblack, false},
        {"k_residual", (const void*)k_residual, false},
    };
    printf("%-18s %5s %8s %9s %8s %10s %11s\n", "kernel", "regs", "static",
           "block", "dyn smem", "blocks/SM", "occupancy");
    for (auto& k : ks) {
        cudaFuncAttributes at;
        cudaFuncGetAttributes(&at, k.fn);
        for (int i = 0; i < nb; ++i) {
            int threads = bx[i] * by[i];
            if (threads > at.maxThreadsPerBlock) continue;
            size_t dyn = k.dynSmem ? (size_t)(bx[i] + 2) * (by[i] + 2) * 4 : 0;
            int blocks = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.fn,
                                                          threads, dyn);
            double occ = (double)blocks * (threads / 32.0) / maxWarps;
            printf("%-18s %5d %8zu %5dx%-3d %8zu %10d %10.1f%%\n", k.name,
                   at.numRegs, at.sharedSizeBytes, bx[i], by[i], dyn, blocks,
                   100.0 * occ);
        }
    }
}

void launch_residual(float* r, const float* p, const float* b, int N,
                     dim3 blk) {
    k_residual<<<grid_for(N, blk), blk>>>(r, p, b, N);
}
void launch_restrict(float* dst, const float* src, int Nc, int Nf, dim3 blk) {
    k_restrict<<<grid_for(Nc, blk), blk>>>(dst, src, Nc, Nf);
}
void launch_prolong_add(float* dst, const float* src, int Nf, int Nc,
                        dim3 blk) {
    k_prolong_add<<<grid_for(Nc, blk), blk>>>(dst, src, Nf, Nc);
}
void launch_set_pure_gradient(float* u, float* v, int N, float k, dim3 blk) {
    dim3 g((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);
    k_set_pure_gradient<<<g, blk>>>(u, v, N, k);
}
void launch_sumsq_field2(const float* a, const float* b, int N, double* d_sum,
                         dim3 blk) {
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    k_sumsq_field2<<<grid_for(N, blk), blk>>>(a, b, N, d_sum);
}
void launch_sumsq_divergence(const float* u, const float* v, int N,
                             double* d_sum, dim3 blk) {
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    k_sumsq_divergence<<<grid_for(N, blk), blk>>>(u, v, N, d_sum);
}
void launch_divergence(float* div, float* p, const float* u, const float* v,
                       int N, dim3 blk) {
    k_divergence<<<grid_for(N, blk), blk>>>(div, p, u, v, N);
}
void launch_subtract_gradient(float* u, float* v, const float* p, int N,
                              dim3 blk) {
    k_subtract_gradient<<<grid_for(N, blk), blk>>>(u, v, p, N);
}
void launch_boundary(float* f, int N, int kind) {
    k_boundary<<<(N + 127) / 128, 128>>>(f, N, kind);
}
void launch_inject(float* dye, float* u, float* v, int N, float cx, float cy,
                   float radius, float dyeAmt, float fx, float fy, dim3 blk) {
    k_inject<<<grid_for(N, blk), blk>>>(dye, u, v, N, cx, cy, radius, dyeAmt,
                                        fx, fy);
}
void launch_buoyancy(float* v, const float* dye, int N, float k, dim3 blk) {
    k_buoyancy<<<grid_for(N, blk), blk>>>(v, dye, N, k);
}
void launch_max_divergence(const float* u, const float* v, int N,
                           unsigned int* d_maxbits, dim3 blk) {
    CUDA_CHECK(cudaMemset(d_maxbits, 0, sizeof(unsigned int)));
    k_max_divergence<<<grid_for(N, blk), blk>>>(u, v, N, d_maxbits);
}
void launch_sum_field(const float* f, int N, double* d_sum, dim3 blk) {
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
    k_sum_field<<<grid_for(N, blk), blk>>>(f, N, d_sum);
}
void launch_colormap(unsigned char* rgb, const float* dye, const float* u,
                     const float* v, int N, dim3 blk) {
    dim3 g((N + blk.x - 1) / blk.x, (N + blk.y - 1) / blk.y);
    k_colormap<<<g, blk>>>(rgb, dye, u, v, N);
}
