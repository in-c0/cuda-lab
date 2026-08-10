// main.cu -- driver for the CUDA 2D stable-fluids solver.
//
// Modes:
//   sim        run the plume scenario, write PPM frames + a CSV diagnostic log
//   verify     divergence + dye-mass conservation over a full run
//   poisson    manufactured-solution test of the projection operator alone
//   converge   pressure-solver convergence vs cost on one frozen field
//   stages     per-stage timing breakdown (CUDA events, warm-up excluded)
//   sweep      grid-size x pressure-iteration-count sweep of ms/frame
//   kbench     kernel-level A/B of the optimisation variants, in isolation
//   opt        whole-frame A/B of the optimisation variants
//   blocksweep block-size sweep
//   occupancy  registers / shared memory / theoretical occupancy per kernel
//
// All timings use CUDA events, exclude warm-up, average >= 30 frames, and keep
// the minimum over --trials repetitions (this GPU also drives the desktop, so
// a single run can be badly polluted by background compositing).
#include <math.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "fluid.cuh"

// ---------------------------------------------------------------------------
struct Config {
    int N = 512;           // total grid incl. ghost border
    int pressureIters = 40;
    int diffIters = 20;
    float dt = 1.0f;       // one cell-unit timestep; large on purpose
    float visc = 0.00005f;
    float buoy = 0.06f;
    bool useTexAdvect = false;
    bool useSharedJacobi = false;
    bool useRedBlack = false;  // red-black Gauss-Seidel instead of Jacobi
    bool useMultigrid = false; // geometric multigrid V-cycles for pressure
    int mgCycles = 4;          // V-cycles per projection
    int mgSmooth = 2;          // red-black GS sweeps pre and post per level
    int mgMaxLevels = 6;       // tuned: deeper levels are pure launch overhead
    int mgCoarseSweeps = 10;   // RBGS sweeps on the coarsest level
    dim3 blk = dim3(32, 8);
};

struct Sim {
    int N;
    size_t bytes;
    float *u, *v, *dye, *u2, *v2, *dye2, *p, *p2, *div, *tmp;
    unsigned char* d_rgb;
    unsigned int* d_maxbits;
    double* d_sum;
    // Texture path resources (one cudaArray per advected field).
    cudaArray_t arrU = nullptr, arrV = nullptr, arrD = nullptr;
    cudaTextureObject_t texU = 0, texV = 0, texD = 0;
    // Multigrid hierarchy: level 0 is the fine grid (aliases p / div).
    std::vector<int> mgN;
    std::vector<float*> mgP, mgB, mgR;
};

// Build the coarse-grid hierarchy. Level 0 borrows the existing p/div/tmp
// buffers; coarser levels get their own (each 1/4 the size, so the whole
// hierarchy costs ~1/3 of one extra fine-grid buffer per array).
static void mg_build(Sim& s) {
    int n = s.N - 2;  // interior size
    s.mgN.push_back(s.N);
    s.mgP.push_back(nullptr);  // filled in at solve time
    s.mgB.push_back(nullptr);
    float* r0 = nullptr;
    CUDA_CHECK(cudaMalloc(&r0, s.bytes));
    s.mgR.push_back(r0);
    while (n % 2 == 0 && n >= 8) {
        n /= 2;
        int N = n + 2;
        s.mgN.push_back(N);
        size_t bs = (size_t)N * N * sizeof(float);
        float *p, *b, *r;
        CUDA_CHECK(cudaMalloc(&p, bs));
        CUDA_CHECK(cudaMalloc(&b, bs));
        CUDA_CHECK(cudaMalloc(&r, bs));
        CUDA_CHECK(cudaMemset(p, 0, bs));
        CUDA_CHECK(cudaMemset(b, 0, bs));
        CUDA_CHECK(cudaMemset(r, 0, bs));
        s.mgP.push_back(p); s.mgB.push_back(b); s.mgR.push_back(r);
    }
}

static void make_tex(cudaArray_t& arr, cudaTextureObject_t& tex, int N) {
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&arr, &ch, N, N));
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;  // hardware bilinear
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
}

static void sim_alloc(Sim& s, int N, bool withTex) {
    s.N = N;
    s.bytes = (size_t)N * N * sizeof(float);
    float** fs[] = {&s.u, &s.v,  &s.dye, &s.u2, &s.v2,
                    &s.dye2, &s.p, &s.p2, &s.div, &s.tmp};
    for (auto* f : fs) {
        CUDA_CHECK(cudaMalloc(f, s.bytes));
        CUDA_CHECK(cudaMemset(*f, 0, s.bytes));
    }
    CUDA_CHECK(cudaMalloc(&s.d_rgb, (size_t)N * N * 3));
    CUDA_CHECK(cudaMalloc(&s.d_maxbits, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&s.d_sum, sizeof(double)));
    mg_build(s);
    if (withTex) {
        make_tex(s.arrU, s.texU, N);
        make_tex(s.arrV, s.texV, N);
        make_tex(s.arrD, s.texD, N);
    }
}

static void sim_free(Sim& s) {
    for (float* f : {s.u, s.v, s.dye, s.u2, s.v2, s.dye2, s.p, s.p2, s.div,
                     s.tmp})
        cudaFree(f);
    cudaFree(s.d_rgb);
    cudaFree(s.d_maxbits);
    cudaFree(s.d_sum);
    for (auto t : {s.texU, s.texV, s.texD})
        if (t) cudaDestroyTextureObject(t);
    for (auto a : {s.arrU, s.arrV, s.arrD})
        if (a) cudaFreeArray(a);
    for (size_t l = 0; l < s.mgN.size(); ++l) {
        if (l > 0) { cudaFree(s.mgP[l]); cudaFree(s.mgB[l]); }
        cudaFree(s.mgR[l]);
    }
    s.mgN.clear(); s.mgP.clear(); s.mgB.clear(); s.mgR.clear();
}

// Advect `src` into `dst` using the current velocity, via whichever variant is
// selected. The texture path must first stage `src` into a cudaArray -- that
// copy is real work and is deliberately included in the timing.
static void advect(Sim& s, const Config& c, float* dst, float* src,
                   cudaArray_t arr, cudaTextureObject_t tex) {
    if (c.useTexAdvect) {
        CUDA_CHECK(cudaMemcpy2DToArray(arr, 0, 0, src, s.N * sizeof(float),
                                       s.N * sizeof(float), s.N,
                                       cudaMemcpyDeviceToDevice));
        launch_advect_tex(dst, tex, s.u, s.v, s.N, c.dt, c.blk);
    } else {
        launch_advect_manual(dst, src, s.u, s.v, s.N, c.dt, c.blk);
    }
}

// Run `iters` relaxation sweeps of the linear system, re-imposing the boundary
// after each sweep. On return `a` points at the buffer holding the answer.
// One red-black iteration = two half-sweeps = the same number of cell updates
// as one Jacobi sweep, so the two are compared on equal work.
static void relax(const Config& c, float*& a, float*& scratch,
                  const float* rhs, int N, float alpha, float rBeta, int iters,
                  int bndKind) {
    if (c.useRedBlack) {
        for (int k = 0; k < iters; ++k) {
            launch_gs_redblack(a, rhs, N, alpha, rBeta, 0, c.blk);
            launch_gs_redblack(a, rhs, N, alpha, rBeta, 1, c.blk);
            launch_boundary(a, N, bndKind);
        }
        return;
    }
    for (int k = 0; k < iters; ++k) {
        if (c.useSharedJacobi)
            launch_jacobi_shared(scratch, a, rhs, N, alpha, rBeta, c.blk);
        else
            launch_jacobi_naive(scratch, a, rhs, N, alpha, rBeta, c.blk);
        launch_boundary(scratch, N, bndKind);
        std::swap(a, scratch);
    }
}

// Red-black Gauss-Seidel smoother used at every multigrid level.
static void mg_smooth(const Config& c, float* p, const float* b, int N,
                      int sweeps) {
    for (int k = 0; k < sweeps; ++k) {
        launch_gs_redblack(p, b, N, 1.0f, 0.25f, 0, c.blk);
        launch_gs_redblack(p, b, N, 1.0f, 0.25f, 1, c.blk);
        launch_boundary(p, N, BND_SCALAR);
    }
}

// One multigrid V-cycle at level l. p[l] is an initial guess and is improved
// in place; b[l] is the right-hand side.
static void mg_vcycle(Sim& s, const Config& c, int l) {
    int N = s.mgN[l];
    int last = min((int)s.mgN.size() - 1, c.mgMaxLevels - 1);
    if (l == last) {
        // Coarsest grid: smooth it hard, it is small.
        mg_smooth(c, s.mgP[l], s.mgB[l], N, c.mgCoarseSweeps);
        return;
    }
    mg_smooth(c, s.mgP[l], s.mgB[l], N, c.mgSmooth);
    launch_residual(s.mgR[l], s.mgP[l], s.mgB[l], N, c.blk);
    launch_boundary(s.mgR[l], N, BND_SCALAR);
    int Nc = s.mgN[l + 1];
    launch_restrict(s.mgB[l + 1], s.mgR[l], Nc, N, c.blk);
    launch_boundary(s.mgB[l + 1], Nc, BND_SCALAR);
    CUDA_CHECK(cudaMemset(s.mgP[l + 1], 0, (size_t)Nc * Nc * sizeof(float)));
    mg_vcycle(s, c, l + 1);
    launch_prolong_add(s.mgP[l], s.mgP[l + 1], N, Nc, c.blk);
    launch_boundary(s.mgP[l], N, BND_SCALAR);
    mg_smooth(c, s.mgP[l], s.mgB[l], N, c.mgSmooth);
}

// ---------------------------------------------------------------------------
// One full timestep, split into the five stages we time separately.
// ---------------------------------------------------------------------------
static void stage_sources(Sim& s, const Config& c, int frame, bool injecting) {
    int N = s.N;
    if (injecting) {
        // Wobbling plume source near the bottom, to break left/right symmetry.
        float wob = 0.12f * N * sinf(frame * 0.06f);
        launch_inject(s.dye, s.u, s.v, N, 0.5f * N + wob, 0.12f * N,
                      0.045f * N, 0.35f, 0.6f * sinf(frame * 0.11f), 1.2f,
                      c.blk);
    }
    launch_buoyancy(s.v, s.dye, N, c.buoy, c.blk);
}

static void stage_diffuse(Sim& s, const Config& c) {
    int N = s.N;
    if (c.visc <= 0.0f || c.diffIters == 0) return;
    float alpha = c.visc * c.dt * N * N;  // nu*dt/h^2 with h = 1/N
    float rBeta = 1.0f / (1.0f + 4.0f * alpha);
    // Implicit (backward-Euler) viscosity: solve (I - alpha*L) x = x_old.
    // Ping-pong between the field and its scratch buffer; the boundary is
    // re-imposed every sweep so the walls stay no-slip during the solve.
    struct { float** f; float** scratch; int kind; } fields[2] = {
        {&s.u, &s.u2, BND_U}, {&s.v, &s.v2, BND_V}};
    for (auto& fd : fields) {
        CUDA_CHECK(cudaMemcpy(s.tmp, *fd.f, s.bytes, cudaMemcpyDeviceToDevice));
        float* a = *fd.f;
        float* b = *fd.scratch;
        relax(c, a, b, s.tmp, N, alpha, rBeta, c.diffIters, fd.kind);
        *fd.f = a;
        *fd.scratch = b;
    }
}

static void stage_advect(Sim& s, const Config& c) {
    // Velocity self-advection must read the *pre-advection* velocity for both
    // components, so write into scratch and swap only afterwards.
    advect(s, c, s.u2, s.u, s.arrU, s.texU);
    advect(s, c, s.v2, s.v, s.arrV, s.texV);
    std::swap(s.u, s.u2);
    std::swap(s.v, s.v2);
}

static void stage_advect_dye(Sim& s, const Config& c) {
    advect(s, c, s.dye2, s.dye, s.arrD, s.texD);
    std::swap(s.dye, s.dye2);
}

static void stage_project(Sim& s, const Config& c) {
    int N = s.N;
    launch_divergence(s.div, s.p, s.u, s.v, N, c.blk);
    launch_boundary(s.div, N, BND_SCALAR);
    launch_boundary(s.p, N, BND_SCALAR);
    if (c.useMultigrid) {
        s.mgP[0] = s.p;
        s.mgB[0] = s.div;
        for (int k = 0; k < c.mgCycles; ++k) mg_vcycle(s, c, 0);
    } else {
        float *a = s.p, *b = s.p2;
        relax(c, a, b, s.div, N, 1.0f, 0.25f, c.pressureIters, BND_SCALAR);
        s.p = a;
        s.p2 = b;
    }
    launch_subtract_gradient(s.u, s.v, s.p, N, c.blk);
}

static void stage_boundary(Sim& s) {
    launch_boundary(s.u, s.N, BND_U);
    launch_boundary(s.v, s.N, BND_V);
    launch_boundary(s.dye, s.N, BND_SCALAR);
}

static void step(Sim& s, const Config& c, int frame, bool injecting) {
    stage_sources(s, c, frame, injecting);
    stage_boundary(s);
    stage_diffuse(s, c);
    stage_advect(s, c);
    stage_boundary(s);
    stage_project(s, c);
    stage_boundary(s);
    stage_advect_dye(s, c);
    stage_boundary(s);
}

// ---------------------------------------------------------------------------
static float read_maxdiv(Sim& s, const Config& c) {
    launch_max_divergence(s.u, s.v, s.N, s.d_maxbits, c.blk);
    unsigned int bits = 0;
    CUDA_CHECK(cudaMemcpy(&bits, s.d_maxbits, sizeof(bits),
                          cudaMemcpyDeviceToHost));
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static double read_rmsdiv(Sim& s, const Config& c) {
    launch_sumsq_divergence(s.u, s.v, s.N, s.d_sum, dim3(16, 16));
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, s.d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    double n = (double)(s.N - 2) * (s.N - 2);
    return sqrt(h / n);
}

static double read_mass(Sim& s) {
    launch_sum_field(s.dye, s.N, s.d_sum, dim3(16, 16));
    double h = 0;
    CUDA_CHECK(cudaMemcpy(&h, s.d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

static void write_ppm(const char* path, const unsigned char* rgb, int N) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", N, N);
    fwrite(rgb, 1, (size_t)N * N * 3, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// This GPU also drives the desktop, so another process can steal SM time at
// any moment. A single averaged run is therefore not trustworthy -- we repeat
// the whole measurement `trials` times and keep the MINIMUM, which is the run
// least polluted by contention. (Observed in practice: an unlucky run read
// 35 ms for a configuration whose clean time is 4.3 ms.)
static int g_trials = 3;
static int g_warmup = 10;

static double time_ms_once(Sim& s, const Config& c, int iters, int warmup) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) step(s, c, i, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) step(s, c, i, true);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

static double time_ms(Sim& s, const Config& c, int iters, int warmup) {
    double best = 1e30;
    for (int t = 0; t < g_trials; ++t)
        best = fmin(best, time_ms_once(s, c, iters, warmup));
    return best;
}

// Per-stage timing: events bracket each stage inside the loop. Each stage is
// timed independently and the MINIMUM over frames is kept, so one contended
// frame cannot inflate a stage. The per-stage sync also serialises stages that
// would otherwise overlap, so the stage total runs slightly above the clean
// whole-frame number -- both are reported.
struct StageTimes { double src=0, diff=0, adv=0, proj=0, bnd=0, dyeadv=0; };

static StageTimes time_stages_once(Sim& s, const Config& c, int iters,
                                   int warmup) {
    cudaEvent_t e[8];
    for (auto& x : e) CUDA_CHECK(cudaEventCreate(&x));
    for (int i = 0; i < warmup; ++i) step(s, c, i, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    StageTimes t;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e[0]);
        stage_sources(s, c, i, true);
        stage_boundary(s);
        cudaEventRecord(e[1]);
        stage_diffuse(s, c);
        cudaEventRecord(e[2]);
        stage_advect(s, c);
        cudaEventRecord(e[3]);
        stage_boundary(s);
        cudaEventRecord(e[4]);
        stage_project(s, c);
        cudaEventRecord(e[5]);
        stage_boundary(s);
        cudaEventRecord(e[6]);
        stage_advect_dye(s, c);
        stage_boundary(s);
        cudaEventRecord(e[7]);
        CUDA_CHECK(cudaEventSynchronize(e[7]));
        float m;
        cudaEventElapsedTime(&m, e[0], e[1]); t.src += m;
        cudaEventElapsedTime(&m, e[1], e[2]); t.diff += m;
        cudaEventElapsedTime(&m, e[2], e[3]); t.adv += m;
        cudaEventElapsedTime(&m, e[3], e[4]); t.bnd += m;
        cudaEventElapsedTime(&m, e[4], e[5]); t.proj += m;
        cudaEventElapsedTime(&m, e[5], e[6]); t.bnd += m;
        cudaEventElapsedTime(&m, e[6], e[7]); t.dyeadv += m;
    }
    for (auto& x : e) cudaEventDestroy(x);
    t.src/=iters; t.diff/=iters; t.adv/=iters; t.proj/=iters; t.bnd/=iters;
    t.dyeadv/=iters;
    return t;
}

static StageTimes time_stages(Sim& s, const Config& c, int iters, int warmup) {
    StageTimes best;
    for (int i = 0; i < g_trials; ++i) {
        StageTimes t = time_stages_once(s, c, iters, warmup);
        if (i == 0) { best = t; continue; }
        best.src = fmin(best.src, t.src);
        best.diff = fmin(best.diff, t.diff);
        best.adv = fmin(best.adv, t.adv);
        best.proj = fmin(best.proj, t.proj);
        best.bnd = fmin(best.bnd, t.bnd);
        best.dyeadv = fmin(best.dyeadv, t.dyeadv);
    }
    return best;
}

// ---------------------------------------------------------------------------
static void print_device() {
    cudaDeviceProp pr;
    CUDA_CHECK(cudaGetDeviceProperties(&pr, 0));
    int rt = 0, dv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&dv);
    printf("# device: %s  sm_%d%d  SMs=%d  memClk=%dMHz  busWidth=%d-bit\n",
           pr.name, pr.major, pr.minor, pr.multiProcessorCount,
           pr.memoryClockRate / 1000, pr.memoryBusWidth);
    printf("# peak theoretical bandwidth: %.1f GB/s\n",
           2.0 * pr.memoryClockRate * 1e3 * (pr.memoryBusWidth / 8) / 1e9);
    printf("# cuda runtime %d.%d, driver api %d.%d\n", rt / 1000, (rt % 1000) / 10,
           dv / 1000, (dv % 1000) / 10);
}

static int arg_int(int argc, char** argv, const char* k, int def) {
    for (int i = 1; i < argc - 1; ++i)
        if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
    return def;
}
static const char* arg_str(int argc, char** argv, const char* k,
                           const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static bool arg_flag(int argc, char** argv, const char* k) {
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], k)) return true;
    return false;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* mode = (argc > 1 && argv[1][0] != '-') ? argv[1] : "sim";
    Config c;
    // --n names the INTERIOR (simulated) resolution; the allocated grid adds a
    // one-cell ghost border. Powers of two keep the multigrid hierarchy clean.
    c.N = arg_int(argc, argv, "--n", 512) + 2;
    c.pressureIters = arg_int(argc, argv, "--iters", 40);
    c.diffIters = arg_int(argc, argv, "--diffiters", 20);
    c.useTexAdvect = arg_flag(argc, argv, "--tex");
    c.useSharedJacobi = arg_flag(argc, argv, "--shared");
    c.useRedBlack = arg_flag(argc, argv, "--rb");
    c.useMultigrid = arg_flag(argc, argv, "--mg");
    c.mgCycles = arg_int(argc, argv, "--cycles", 4);
    c.mgSmooth = arg_int(argc, argv, "--smooth", 2);
    c.mgMaxLevels = arg_int(argc, argv, "--levels", 6);
    c.mgCoarseSweeps = arg_int(argc, argv, "--coarse", 10);
    g_trials = arg_int(argc, argv, "--trials", 3);
    g_warmup = arg_int(argc, argv, "--warmup", 10);
    c.blk = dim3(arg_int(argc, argv, "--bx", 32), arg_int(argc, argv, "--by", 8));
    print_device();

    if (!strcmp(mode, "sim")) {
        int frames = arg_int(argc, argv, "--frames", 300);
        int stopInject = arg_int(argc, argv, "--stopinject", 60);
        int every = arg_int(argc, argv, "--every", 10);
        const char* out = arg_str(argc, argv, "--out", "frames");
        const char* logp = arg_str(argc, argv, "--log", "sim_log.csv");
        Sim s; sim_alloc(s, c.N, c.useTexAdvect);
        std::vector<unsigned char> host((size_t)c.N * c.N * 3);
        FILE* lg = fopen(logp, "w");
        fprintf(lg, "frame,dye_mass,maxdiv_before_projection,maxdiv_after_projection\n");
        printf("# scenario: buoyant dye plume, no-slip walls, dt=%.2f, "
               "N=%d (interior %d^2), pressure iters=%d\n",
               c.dt, c.N, c.N - 2, c.pressureIters);
        for (int f = 0; f < frames; ++f) {
            bool inj = (f < stopInject);
            // Measure divergence just before and just after the projection.
            stage_sources(s, c, f, inj);
            stage_boundary(s);
            stage_diffuse(s, c);
            stage_advect(s, c);
            stage_boundary(s);
            float dbefore = read_maxdiv(s, c);
            stage_project(s, c);
            stage_boundary(s);
            float dafter = read_maxdiv(s, c);
            stage_advect_dye(s, c);
            stage_boundary(s);
            double mass = read_mass(s);
            fprintf(lg, "%d,%.6f,%.6e,%.6e\n", f, mass, dbefore, dafter);
            if (f % every == 0 || f == frames - 1) {
                launch_colormap(s.d_rgb, s.dye, s.u, s.v, c.N, dim3(16, 16));
                CUDA_CHECK(cudaMemcpy(host.data(), s.d_rgb, host.size(),
                                      cudaMemcpyDeviceToHost));
                char path[512];
                snprintf(path, sizeof(path), "%s/frame_%04d.ppm", out, f);
                write_ppm(path, host.data(), c.N);
            }
            if (f % 50 == 0)
                printf("frame %3d  mass=%.4f  div_before=%.3e  div_after=%.3e\n",
                       f, mass, dbefore, dafter);
        }
        fclose(lg);
        printf("# wrote frames to %s/ and log to %s\n", out, logp);
        sim_free(s);
        return 0;
    }

    if (!strcmp(mode, "verify")) {
        int frames = arg_int(argc, argv, "--frames", 300);
        Sim s; sim_alloc(s, c.N, c.useTexAdvect);
        double m0 = 0, worstAfter = 0, worstBefore = 0;
        double rmsB = 0, rmsA = 0;
        int nr = 0;
        for (int f = 0; f < frames; ++f) {
            bool inj = (f < 60);
            stage_sources(s, c, f, inj); stage_boundary(s);
            stage_diffuse(s, c); stage_advect(s, c); stage_boundary(s);
            float db = read_maxdiv(s, c);
            double rb = read_rmsdiv(s, c);
            stage_project(s, c); stage_boundary(s);
            float da = read_maxdiv(s, c);
            double ra = read_rmsdiv(s, c);
            stage_advect_dye(s, c); stage_boundary(s);
            if (f >= 60) {
                worstBefore = fmax(worstBefore, db);
                worstAfter = fmax(worstAfter, da);
                rmsB += rb; rmsA += ra; nr++;
            }
            if (f == 60) m0 = read_mass(s);
        }
        double m1 = read_mass(s);
        printf("N=%d interior, frames=%d, solver=%s (injection off after 60)\n",
               c.N - 2, frames,
               c.useMultigrid ? "multigrid V-cycles"
                              : (c.useRedBlack ? "red-black GS" : "Jacobi"));
        if (c.useMultigrid)
            printf("  %d V-cycles/projection, %d RBGS sweeps per level\n",
                   c.mgCycles, c.mgSmooth);
        else
            printf("  %d sweeps/projection\n", c.pressureIters);
        printf("max |div| before projection (worst, frames 60..%d): %.4e\n",
               frames - 1, worstBefore);
        printf("max |div| after  projection (worst, frames 60..%d): %.4e\n",
               frames - 1, worstAfter);
        printf("mean RMS |div| before projection: %.4e\n", rmsB / nr);
        printf("mean RMS |div| after  projection: %.4e\n", rmsA / nr);
        printf("mean RMS divergence reduction:    %.1fx\n", rmsB / rmsA);
        printf("dye mass at frame 60: %.6f\n", m0);
        printf("dye mass at frame %d: %.6f\n", frames - 1, m1);
        printf("dye mass drift: %+.4f%%\n", 100.0 * (m1 - m0) / m0);
        sim_free(s);
        return 0;
    }

    // Manufactured-solution test of the projection operator alone.
    if (!strcmp(mode, "poisson")) {
        Sim s; sim_alloc(s, c.N, false);
        auto rms_vel = [&]() {
            launch_sumsq_field2(s.u, s.v, s.N, s.d_sum, dim3(16, 16));
            double h = 0;
            CUDA_CHECK(cudaMemcpy(&h, s.d_sum, sizeof(double),
                                  cudaMemcpyDeviceToHost));
            return sqrt(h / ((double)(s.N - 2) * (s.N - 2)));
        };
        printf("Manufactured solution: u = grad(cos(k*pi*x)*cos(k*pi*y)).\n");
        printf("This field is curl-free, so the exact projection result is "
               "u = 0; the surviving |u| is the solver error.\n");
        printf("N=%d, solver=%s\n", c.N, c.useRedBlack ? "red-black GS" : "Jacobi");
        printf("%-6s %14s %8s %14s %12s\n", "mode k", "RMS|u| initial",
               c.useMultigrid ? "Vcycles" : "sweeps", "RMS|u| after",
               "removed");
        for (float k : {1.0f, 4.0f, 16.0f, 64.0f}) {
            for (int it : {20, 80, 320}) {
                Config cc = c; cc.pressureIters = it;
                if (cc.useMultigrid) cc.mgCycles = it / 20;  // 1, 4, 16 cycles
                launch_set_pure_gradient(s.u, s.v, s.N, k, dim3(16, 16));
                double r0 = rms_vel();
                stage_project(s, cc);
                double r1 = rms_vel();
                printf("%-6.0f %14.4e %8d %14.4e %11.3f%%\n", k, r0,
                       cc.useMultigrid ? cc.mgCycles : it, r1,
                       100.0 * (1.0 - r1 / r0));
                fflush(stdout);
            }
        }
        sim_free(s);
        return 0;
    }

    // Convergence of the pressure solve on ONE fixed, fully developed velocity
    // field: how much divergence does each solver remove, and at what cost.
    if (!strcmp(mode, "converge")) {
        Sim s; sim_alloc(s, c.N, c.useTexAdvect);
        Config warm = c;
        for (int f = 0; f < 80; ++f) step(s, warm, f, f < 60);
        // Snapshot the post-advection, pre-projection velocity.
        float *su, *sv;
        CUDA_CHECK(cudaMalloc(&su, s.bytes));
        CUDA_CHECK(cudaMalloc(&sv, s.bytes));
        stage_sources(s, c, 80, false); stage_boundary(s);
        stage_advect(s, c); stage_boundary(s);
        CUDA_CHECK(cudaMemcpy(su, s.u, s.bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sv, s.v, s.bytes, cudaMemcpyDeviceToDevice));
        double rms0 = read_rmsdiv(s, c);
        float max0 = read_maxdiv(s, c);
        printf("N=%d  fixed turbulent field, RMS|div|=%.4e max|div|=%.4e\n",
               c.N, rms0, max0);
        printf("%-14s %8s %14s %12s %12s %10s\n", "solver", "sweeps",
               "RMS|div| after", "reduction", "max|div|", "ms");
        int its[] = {10, 20, 40, 80, 160, 320, 640};
        for (int rb = 0; rb < 2; ++rb) {
            for (int it : its) {
                Config cc = c; cc.pressureIters = it; cc.useRedBlack = (rb == 1);
                CUDA_CHECK(cudaMemcpy(s.u, su, s.bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(s.v, sv, s.bytes, cudaMemcpyDeviceToDevice));
                stage_project(s, cc);
                double rms = read_rmsdiv(s, cc);
                float mx = read_maxdiv(s, cc);
                // Time the projection alone, 50 reps, warm-up excluded.
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                for (int w = 0; w < 5; ++w) stage_project(s, cc);
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEventRecord(a);
                for (int r = 0; r < 50; ++r) stage_project(s, cc);
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b);
                cudaEventDestroy(a); cudaEventDestroy(b);
                printf("%-14s %8d %14.4e %11.1fx %12.4e %10.4f\n",
                       cc.useRedBlack ? "red-black GS" : "Jacobi", it, rms,
                       rms0 / rms, mx, ms / 50);
                fflush(stdout);
            }
        }
        cudaFree(su); cudaFree(sv);
        sim_free(s);
        return 0;
    }

    if (!strcmp(mode, "stages")) {
        int iters = arg_int(argc, argv, "--bench", 50);
        Sim s; sim_alloc(s, c.N, c.useTexAdvect);
        double whole = time_ms(s, c, iters, g_warmup);
        StageTimes t = time_stages(s, c, iters, g_warmup);
        double tot = t.src + t.diff + t.adv + t.proj + t.bnd + t.dyeadv;
        printf("N=%d iters=%d block=%dx%d tex=%d shared=%d avg over %d frames\n",
               c.N, c.pressureIters, c.blk.x, c.blk.y, (int)c.useTexAdvect,
               (int)c.useSharedJacobi, iters);
        printf("whole-frame (single event pair): %.4f ms\n", whole);
        printf("%-22s %10s %8s\n", "stage", "ms/frame", "%");
        auto row = [&](const char* n, double m) {
            printf("%-22s %10.4f %7.1f%%\n", n, m, 100.0 * m / tot);
        };
        row("sources+buoyancy", t.src);
        row("diffusion (jacobi)", t.diff);
        row("advect velocity", t.adv);
        row("projection (jacobi)", t.proj);
        row("boundary", t.bnd);
        row("advect dye", t.dyeadv);
        printf("%-22s %10.4f\n", "instrumented total", tot);
        sim_free(s);
        return 0;
    }

    // Kernel-level A/B. Whole-frame timing dilutes a single kernel's effect
    // with unrelated stages, so time the candidate kernels in isolation:
    // `reps` back-to-back launches inside one event pair, warm-up excluded,
    // repeated `trials` times so the spread is visible.
    if (!strcmp(mode, "kbench")) {
        int reps = arg_int(argc, argv, "--reps", 200);
        int trials = arg_int(argc, argv, "--trials", 5);
        printf("%-6s %-30s %11s %11s %11s %10s\n", "N", "kernel", "min us",
               "mean us", "max us", "GB/s");
        for (int n : {512, 1024, 2048}) {
            Sim s; sim_alloc(s, n + 2, true);
            int N = n + 2;
            // Give the fields non-trivial contents so nothing is optimised
            // away and the advection backtrace actually moves.
            launch_set_pure_gradient(s.u, s.v, N, 8.0f, dim3(16, 16));
            CUDA_CHECK(cudaDeviceSynchronize());

            struct Case { const char* name; int kind; double bytes; };
            // Bytes moved per launch, counting compulsory traffic only.
            double cells = (double)n * n;
            Case cases[] = {
                {"advect: manual bilerp", 0, cells * 4 * 4},
                {"advect: tex2D + array copy", 1, cells * 4 * 5},
                {"advect: tex2D (copy excluded)", 2, cells * 4 * 4},
                {"jacobi: naive global", 3, cells * 4 * 3},
                {"jacobi: shared-mem halo", 4, cells * 4 * 3},
                {"gauss-seidel: red-black", 5, cells * 4 * 3},
            };
            for (auto& cs : cases) {
                auto run = [&]() {
                    switch (cs.kind) {
                        case 0: launch_advect_manual(s.dye2, s.dye, s.u, s.v, N,
                                                     c.dt, c.blk); break;
                        case 1: CUDA_CHECK(cudaMemcpy2DToArray(
                                    s.arrD, 0, 0, s.dye, N * sizeof(float),
                                    N * sizeof(float), N,
                                    cudaMemcpyDeviceToDevice));
                                launch_advect_tex(s.dye2, s.texD, s.u, s.v, N,
                                                  c.dt, c.blk); break;
                        case 2: launch_advect_tex(s.dye2, s.texD, s.u, s.v, N,
                                                  c.dt, c.blk); break;
                        case 3: launch_jacobi_naive(s.p2, s.p, s.div, N, 1.f,
                                                    0.25f, c.blk); break;
                        case 4: launch_jacobi_shared(s.p2, s.p, s.div, N, 1.f,
                                                     0.25f, c.blk); break;
                        case 5: launch_gs_redblack(s.p, s.div, N, 1.f, 0.25f,
                                                   0, c.blk);
                                launch_gs_redblack(s.p, s.div, N, 1.f, 0.25f,
                                                   1, c.blk); break;
                    }
                };
                for (int w = 0; w < 20; ++w) run();
                CUDA_CHECK(cudaDeviceSynchronize());
                double mn = 1e30, mx = 0, sum = 0;
                for (int t = 0; t < trials; ++t) {
                    cudaEvent_t a, b;
                    cudaEventCreate(&a); cudaEventCreate(&b);
                    cudaEventRecord(a);
                    for (int r = 0; r < reps; ++r) run();
                    cudaEventRecord(b);
                    CUDA_CHECK(cudaEventSynchronize(b));
                    float ms; cudaEventElapsedTime(&ms, a, b);
                    double us = 1000.0 * ms / reps;
                    mn = fmin(mn, us); mx = fmax(mx, us); sum += us;
                    cudaEventDestroy(a); cudaEventDestroy(b);
                }
                printf("%-6d %-30s %11.2f %11.2f %11.2f %10.1f\n", n, cs.name,
                       mn, sum / trials, mx, cs.bytes / (mn * 1e3));
                fflush(stdout);
            }
            sim_free(s);
        }
        return 0;
    }

    if (!strcmp(mode, "occupancy")) {
        int bx[] = {8, 16, 16, 32, 32, 32, 64, 64, 128};
        int by[] = {8, 8, 16, 4, 8, 16, 4, 8, 8};
        print_occupancy_table(bx, by, 9);
        return 0;
    }

    if (!strcmp(mode, "sweep")) {
        int iters = arg_int(argc, argv, "--bench", 50);
        int Ns[] = {256, 512, 1024, 2048};
        int PIs[] = {20, 40, 80};
        printf("%-8s %-8s %12s %10s\n", "N", "p_iters", "ms/frame", "FPS");
        for (int n : Ns) {
            for (int pi : PIs) {
                Config cc = c; cc.N = n + 2; cc.pressureIters = pi;
                Sim s; sim_alloc(s, cc.N, cc.useTexAdvect);
                double ms = time_ms(s, cc, iters, g_warmup);
                printf("%-8d %-8d %12.4f %10.1f\n", n, pi, ms, 1000.0 / ms);
                fflush(stdout);
                sim_free(s);
            }
        }
        return 0;
    }

    if (!strcmp(mode, "opt")) {
        int iters = arg_int(argc, argv, "--bench", 50);
        int Ns[] = {512, 1024, 2048};
        printf("%-6s %-28s %12s %10s\n", "N", "variant", "ms/frame", "vs base");
        for (int n : Ns) {
            double base = 0;
            struct V { const char* name; bool tex; bool sh; } vs[] = {
                {"baseline (global bilerp,", false, false},
                {"+ texture advect", true, false},
                {"+ shared-mem jacobi", false, true},
                {"+ both", true, true},
            };
            for (auto& v : vs) {
                Config cc = c; cc.N = n + 2; cc.useTexAdvect = v.tex;
                cc.useSharedJacobi = v.sh;
                Sim s; sim_alloc(s, cc.N, cc.useTexAdvect);
                double ms = time_ms(s, cc, iters, g_warmup);
                if (base == 0) base = ms;
                printf("%-6d %-28s %12.4f %9.3fx\n", n, v.name, ms, base / ms);
                fflush(stdout);
                sim_free(s);
            }
        }
        return 0;
    }

    if (!strcmp(mode, "blocksweep")) {
        int iters = arg_int(argc, argv, "--bench", 50);
        int n = c.N;
        int bx[] = {8, 16, 16, 32, 32, 32, 64, 64, 128};
        int by[] = {8, 8, 16, 4, 8, 16, 4, 8, 8};
        printf("N=%d p_iters=%d shared=%d\n", n, c.pressureIters,
               (int)c.useSharedJacobi);
        printf("%-10s %12s\n", "block", "ms/frame");
        for (int k = 0; k < 9; ++k) {
            if (bx[k] * by[k] > 1024) continue;
            Config cc = c; cc.blk = dim3(bx[k], by[k]);
            Sim s; sim_alloc(s, cc.N, cc.useTexAdvect);
            double ms = time_ms(s, cc, iters, g_warmup);
            printf("%2dx%-7d %12.4f\n", bx[k], by[k], ms);
            fflush(stdout);
            sim_free(s);
        }
        return 0;
    }

    fprintf(stderr, "unknown mode '%s'\n", mode);
    return 1;
}
