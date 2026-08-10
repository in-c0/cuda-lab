// ============================================================================
// GlassOS CPPN renderer -- CUDA port
//
// Source of truth: D:\Projects\glassos-webgl\index.html, fragment shader at
// lines 694-836. One CUDA thread == one fragment shader invocation.
//
// Three kernel variants (all in this file so they share the same math and are
// compiled with identical flags):
//   A  naive     : genomes read from global memory, every organism evaluated
//                  unconditionally (no `if (lr > 1.6) continue;` skip)
//   B  constant  : genomes in __constant__ memory + the shader's early-out
//   C  fastmath  : B + hardware-approximate transcendentals + #pragma unroll
//
// Build: build.bat   (nvcc -O3 -arch=sm_75)
// ============================================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "genome.h"
#include "png.h"

#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Genomes in constant memory.
//
// In GLSL these 96 vec4 arrive as a uniform array: read-only, identical for
// every invocation, broadcast from a fast on-chip path. __constant__ memory is
// the exact CUDA analogue -- 64KB, cached, and a warp-uniform read costs one
// broadcast instead of 32 loads. 96*16 = 1536 bytes, comfortably inside the
// limit, and every access here is warp-uniform (the index depends only on the
// organism loop counter, never on threadIdx), which is the ideal case.
// ---------------------------------------------------------------------------
__constant__ float4 c_G[GSLOTS];

struct Uni {
    float resx, resy, time, vitality, restless, dream, beat;
    float px, py, pglow;
    int alive;
    float ditherAmp;   // 0.015 as in the shader; set 0 to isolate float-vs-double error
};

// ---------------------------- GLSL builtins --------------------------------
__device__ __forceinline__ float gfract(float x){ return x - floorf(x); }
__device__ __forceinline__ float gclamp(float x,float a,float b){ return fminf(fmaxf(x,a),b); }
__device__ __forceinline__ float gmix(float a,float b,float t){ return a + (b-a)*t; }
__device__ __forceinline__ float gstep(float e0,float e1,float x){
    float t = gclamp((x-e0)/(e1-e0), 0.f, 1.f);
    return t*t*(3.f-2.f*t);
}

// FAST=0 : IEEE-ish libdevice paths (matches GLSL highp closely)
// FAST=1 : Turing SFU intrinsics. __sinf/__expf are ~2 ulp-ish over the ranges
//          used here but degrade badly for large |x|; see bench.md accuracy note.
template<int FAST> __device__ __forceinline__ float F_sin(float x){ return FAST? __sinf(x) : sinf(x); }
template<int FAST> __device__ __forceinline__ float F_cos(float x){ return FAST? __cosf(x) : cosf(x); }
template<int FAST> __device__ __forceinline__ float F_exp(float x){ return FAST? __expf(x) : expf(x); }
template<int FAST> __device__ __forceinline__ float F_div(float a,float b){ return FAST? __fdividef(a,b) : a/b; }
template<int FAST> __device__ __forceinline__ float F_pow(float a,float b){
    return FAST ? __powf(a,b) : powf(a,b);
}

template<int FAST>
__device__ __forceinline__ float hash2(float x,float y){
    float px = gfract(x*123.34f), py = gfract(y*345.45f);
    float d  = px*(px+34.345f) + py*(py+34.345f);
    px += d; py += d;
    return gfract(px*py);
}
template<int FAST>
__device__ __forceinline__ float noise2(float x,float y){
    float ix = floorf(x), iy = floorf(y);
    float fx = x-ix, fy = y-iy;
    float a = hash2<FAST>(ix,     iy);
    float b = hash2<FAST>(ix+1.f, iy);
    float c = hash2<FAST>(ix,     iy+1.f);
    float d = hash2<FAST>(ix+1.f, iy+1.f);
    float ux = fx*fx*(3.f-2.f*fx), uy = fy*fy*(3.f-2.f*fy);
    return gmix(gmix(a,b,ux), gmix(c,d,ux), uy);
}
template<int FAST>
__device__ __forceinline__ float fbm2(float x,float y){
    float v=0.f, a=0.5f;
    #pragma unroll
    for(int i=0;i<5;i++){ v += a*noise2<FAST>(x,y); x = x*2.03f+7.1f; y = y*2.03f+7.1f; a *= 0.5f; }
    return v;
}
// actF from the shader: a threshold chain over four activations. Every lane in
// a warp may pick a different branch only if the *genome* differs, and the
// genome index is warp-uniform -- so this chain is actually divergence-free
// across a warp, which is why it survives the port unchanged.
template<int FAST>
__device__ __forceinline__ float actF(float a, float x){
    if (a < 0.5f) return F_sin<FAST>(x);
    if (a < 1.5f) return F_exp<FAST>(-x*x)*2.f - 1.f;
    if (a < 2.5f) return tanhf(x);
    return fabsf(gfract(x*0.25f+0.25f)-0.5f)*4.f - 1.f;
}

// CPPN evaluation. GS points either at __constant__ or at global memory.
template<int FAST>
__device__ __forceinline__ float cppn(const float4* __restrict__ GS, int B,
                                      float lpx, float lpy, float lr,
                                      float ica, float isa, float tt){
    float acc = GS[B+15].z;
    #pragma unroll
    for (int i = 0; i < GH; i++){
        float4 n0 = GS[B+2+2*i];
        float4 n1 = GS[B+3+2*i];
        float pre = n0.y + n0.z*tt + n0.w*lpx + n1.x*lpy + n1.y*lr + n1.z*ica + n1.w*isa;
        float h = actF<FAST>(n0.x, pre);
        float4 ow4 = (i < 4) ? GS[B+14] : GS[B+15];
        int j = (i < 4) ? i : i-4;
        float ow = (j==0)?ow4.x:(j==1)?ow4.y:(j==2)?ow4.z:ow4.w;
        acc += ow*h;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// The kernel. CONSTG: read genomes from __constant__. EARLY: honour the
// shader's `if (lr > 1.6) continue;`. FAST: use SFU intrinsics.
// ---------------------------------------------------------------------------
template<int CONSTG, int EARLY, int FAST>
__global__ void render(uchar4* __restrict__ out, int W, int H_, Uni u,
                       const float4* __restrict__ gG){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= W || iy >= H_) return;

    const float4* GS = CONSTG ? c_G : gG;

    float fragx = ix + 0.5f;
    float fragy = (H_-1-iy) + 0.5f;            // PNG row 0 == top == max gl_FragCoord.y
    float uvx = fragx/u.resx, uvy = fragy/u.resy;
    float px = (fragx - 0.5f*u.resx)/u.resy;
    float py = (fragy - 0.5f*u.resy)/u.resy;
    float r  = sqrtf(px*px+py*py);
    float ang = atan2f(py,px);

    // ---- mood -> palette
    float wR=1.00f,wG=0.80f,wB=0.45f;
    float vR=0.62f,vG=0.48f,vB=1.00f;
    float bR=0.30f,bG=0.50f,bB=1.00f;
    float tR = wR*(0.25f+0.75f*u.vitality) + vR*0.45f + bR*(0.25f+0.65f*u.restless) + 0.02f;
    float tG = wG*(0.25f+0.75f*u.vitality) + vG*0.45f + bG*(0.25f+0.65f*u.restless) + 0.02f;
    float tB = wB*(0.25f+0.75f*u.vitality) + vB*0.45f + bB*(0.25f+0.65f*u.restless) + 0.02f;
    float tl = sqrtf(tR*tR+tG*tG+tB*tB);
    tR/=tl; tG/=tl; tB/=tl;
    float tcR = gmix(tR,bR,0.5f*u.dream), tcG = gmix(tG,bG,0.5f*u.dream), tcB = gmix(tB,bB,0.5f*u.dream);

    float cR=0.f,cG=0.f,cB=0.f;

    // ---- background nebula
    float driftT = u.time*0.035f;
    float neb  = fbm2<FAST>(px*1.6f+driftT, py*1.6f-driftT*0.7f);
    neb = F_pow<FAST>(neb, 1.7f);
    float neb2 = fbm2<FAST>(px*3.4f-driftT*1.3f, py*3.4f-driftT);
    cR += gmix(0.005f, tcR*0.09f, neb) + tR*0.03f*neb2*(0.3f+0.5f*u.restless);
    cG += gmix(0.007f, tcG*0.09f, neb) + tG*0.03f*neb2*(0.3f+0.5f*u.restless);
    cB += gmix(0.017f, tcB*0.09f, neb) + tB*0.03f*neb2*(0.3f+0.5f*u.restless);

    // ---- refractive glass orb
    float R = 0.225f + 0.014f*u.beat*(0.5f+u.vitality) + 0.007f*F_sin<FAST>(u.time*0.6f);
    float z = sqrtf(fmaxf(R*R - r*r, 0.f)) / fmaxf(R, 1e-4f);
    float inside = gstep(R+0.004f, R-0.004f, r);
    float fres = F_pow<FAST>(1.f - gclamp(z,0.f,1.f), 2.5f);

    float spx = px*(5.5f+2.0f*z), spy = py*(5.5f+2.0f*z);
    float wx = F_cos<FAST>(ang*2.f+u.time*0.4f)*(0.35f*z);
    float wy = F_sin<FAST>(ang*2.f-u.time*0.3f)*(0.35f*z);
    float ca = 0.010f + 0.018f*fres + 0.006f*u.restless;
    float bx = spx+wx+u.time*0.10f, by = spy+wy;
    float cauR = fbm2<FAST>(bx+ca, by+ca);
    float cauG = fbm2<FAST>(bx,    by);
    float cauB = fbm2<FAST>(bx-ca, by-ca);
    cauR*=cauR; cauG*=cauG; cauB*=cauB;
    float zf = (0.30f+0.9f*z);
    float coR = tR*(0.10f+1.45f*cauR)*inside*zf;
    float coG = tG*(0.10f+1.45f*cauG)*inside*zf;
    float coB = tB*(0.10f+1.45f*cauB)*inside*zf;
    float k2 = inside*(0.05f+0.15f*u.vitality)*(0.4f+0.7f*z);
    coR += tR*k2; coG += tG*k2; coB += tB*k2;
    float k3 = u.beat*inside*F_exp<FAST>(-r*8.f)*(0.25f+0.40f*u.vitality);
    coR += tR*k3; coG += tG*k3; coB += tB*k3;
    float rimR = gmix(1.00f,tR,0.55f)*fres*ca*24.f + tR*fres*1.2f;
    float rimG = gmix(0.62f,tG,0.55f)*fres*ca*24.f + tG*fres*1.2f;
    float rimB = gmix(0.82f,tB,0.55f)*fres*ca*24.f + tB*fres*1.2f;
    coR += rimR*inside*0.8f; coG += rimG*inside*0.8f; coB += rimB*inside*0.8f;
    cR += coR; cG += coG; cB += coB;

    float halo = F_exp<FAST>(-fmaxf(r-R,0.f)*8.5f) * (0.10f+0.13f*u.vitality)*(0.7f+0.3f*u.beat);
    cR += tR*halo; cG += tG*halo; cB += tB*halo;

    // ---- the organisms
    float disp = 0.020f + 0.020f*u.restless;
    #pragma unroll 1
    for (int gI = 0; gI < MAXG; gI++){
        if (gI >= u.alive) break;
        int B = gI*16;
        float4 m0 = GS[B];
        float4 m1 = GS[B+1];
        float lpx = (px-m0.x)/m0.z, lpy = (py-m0.y)/m0.z;
        float lr = sqrtf(lpx*lpx+lpy*lpy);
        // Fragment-shader early-out. In CUDA this is a per-lane predicate, so
        // the warp only skips work when *all 32 lanes* are outside the disc --
        // true for the large screen regions between organisms, false along
        // their silhouettes. That is the warp-divergence story of this port.
        if (EARLY && lr > 1.6f) continue;
        float ica = lr > 1e-4f ? lpx/lr : 1.f;
        float isa = lr > 1e-4f ? lpy/lr : 0.f;
        float tt = u.time*m1.w;
        float fR = cppn<FAST>(GS, B, lpx, lpy, lr+disp, ica, isa, tt);
        float fB = cppn<FAST>(GS, B, lpx, lpy, lr-disp, ica, isa, tt);
        float fG = 0.5f*(fR+fB);
        float lw = GS[B+15].w;
        float lR = 1.f - gstep(0.f, lw, fabsf(fR));
        float lG = 1.f - gstep(0.f, lw, fabsf(fG));
        float lB = 1.f - gstep(0.f, lw, fabsf(fB));
        float env = F_exp<FAST>(-lr*lr*0.9f);
        float gcR = gmix(m1.x,0.55f,0.35f*u.dream);
        float gcG = gmix(m1.y,0.60f,0.35f*u.dream);
        float gcB = gmix(m1.z,1.00f,0.35f*u.dream);
        float amp = env*m0.w*(1.55f+0.85f*u.dream);
        float aura = 0.16f*F_exp<FAST>(-fabsf(fG)*2.f)*env*m0.w;
        // !EARLY variant still masks by the disc so the image is identical
        float mask = (EARLY || lr <= 1.6f) ? 1.f : 0.f;
        cR += (lR*gcR*amp + gcR*aura)*mask;
        cG += (lG*gcG*amp + gcG*aura)*mask;
        cB += (lB*gcB*amp + gcB*aura)*mask;
    }

    // ---- pointer bloom
    float pdx = px-u.px, pdy = py-u.py;
    float pd = sqrtf(pdx*pdx+pdy*pdy);
    float spark = u.pglow*F_exp<FAST>(-pd*pd*26.f);
    cR += (tR*0.6f + 0.5f*0.5f)*spark*1.3f;
    cG += (tG*0.6f + 0.7f*0.5f)*spark*1.3f;
    cB += (tB*0.6f + 1.0f*0.5f)*spark*1.3f;
    float ring = u.pglow*0.25f*(1.f - gstep(0.f,0.02f, fabsf(pd-0.06f-0.04f*F_sin<FAST>(u.time*3.f))));
    cR += tR*ring; cG += tG*ring; cB += tB*ring;

    // ---- finish
    float vig = 1.f - 0.55f*gstep(0.35f,1.35f, r);
    cR*=vig; cG*=vig; cB*=vig;
    cR = F_pow<FAST>(1.f-F_exp<FAST>(-cR*1.08f), 0.4545f);
    cG = F_pow<FAST>(1.f-F_exp<FAST>(-cG*1.08f), 0.4545f);
    cB = F_pow<FAST>(1.f-F_exp<FAST>(-cB*1.08f), 0.4545f);
    float dith = (hash2<FAST>(uvx*u.resx+u.time, uvy*u.resy+u.time)-0.5f)*u.ditherAmp;
    cR+=dith; cG+=dith; cB+=dith;

    out[iy*W+ix] = make_uchar4(
        (unsigned char)(gclamp(cR,0.f,1.f)*255.f+0.5f),
        (unsigned char)(gclamp(cG,0.f,1.f)*255.f+0.5f),
        (unsigned char)(gclamp(cB,0.f,1.f)*255.f+0.5f), 255);
}

// ---------------------------------------------------------------------------
// One timed round: `iters` launches, returns ms per launch.
static float timeOnce(int variant, uchar4* d_out, int W, int H_, const Uni& u,
                      const float4* d_G, dim3 blk, int iters){
    dim3 grd((W+blk.x-1)/blk.x, (H_+blk.y-1)/blk.y);
    #define LAUNCH1 do{ if(variant==0) render<0,0,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                        else if(variant==1) render<1,1,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                        else if(variant==2) render<1,1,1><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                        else if(variant==3) render<1,0,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                        else if(variant==4) render<0,1,0><<<grd,blk>>>(d_out,W,H_,u,d_G);                         else render<0,1,1><<<grd,blk>>>(d_out,W,H_,u,d_G); }while(0)
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i=0;i<iters;i++) LAUNCH1;
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms=0; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    CHECK(cudaGetLastError());
    return ms/iters;
    #undef LAUNCH1
}

// Interleaved comparison: round-robin over `n` configurations for `rounds`
// rounds, keeping each configuration's minimum. This is the fix for a real
// problem seen on this machine -- measuring configs in sequential blocks let
// GPU clock drift over the run masquerade as a 30% difference between two
// launches of the *same* kernel. Round-robin puts every config under the same
// thermal/clock conditions.
static void timeInterleaved(const int* variants, int n, float* out,
                            uchar4* d_out, int W, int H_, const Uni& u,
                            const float4* d_G, dim3 blk, int iters, int rounds){
    for (int k=0;k<n;k++){
        for (int i=0;i<20;i++) timeOnce(variants[k], d_out, W, H_, u, d_G, blk, 1);
        out[k]=1e30f;
    }
    CHECK(cudaDeviceSynchronize());
    for (int rd=0; rd<rounds; rd++)
        for (int k=0;k<n;k++){
            float ms = timeOnce(variants[k], d_out, W, H_, u, d_G, blk, iters);
            if (ms<out[k]) out[k]=ms;
        }
}

static float timeKernel(int variant, uchar4* d_out, int W, int H_, const Uni& u,
                        const float4* d_G, dim3 blk, int iters){
    dim3 grd((W+blk.x-1)/blk.x, (H_+blk.y-1)/blk.y);
    #define LAUNCH(V) do{ if(variant==0) render<0,0,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                         else if(variant==1) render<1,1,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                         else if(variant==2) render<1,1,1><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                         else if(variant==3) render<1,0,0><<<grd,blk>>>(d_out,W,H_,u,d_G); \
                         else if(variant==4) render<0,1,0><<<grd,blk>>>(d_out,W,H_,u,d_G);                         else render<0,1,1><<<grd,blk>>>(d_out,W,H_,u,d_G); }while(0)
    // This is a desktop GPU also driving the display, so individual timed runs
    // are contaminated by other work. We take 3 timed rounds of `iters`
    // launches each and return the MINIMUM mean -- the least-interfered
    // estimate. Round-to-round spread is printed by the caller's repeats.
    for (int i=0;i<20;i++) LAUNCH(variant);            // warm-up
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int round=0; round<3; round++){
        cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        for (int i=0;i<iters;i++) LAUNCH(variant);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms=0; CHECK(cudaEventElapsedTime(&ms,a,b));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        if (ms/iters < best) best = ms/iters;
    }
    CHECK(cudaGetLastError());
    return best;
    #undef LAUNCH
}

int main(int argc, char** argv){
    int W = 1920, H_ = 1080, iters = 100;
    if (argc>2){ W=atoi(argv[1]); H_=atoi(argv[2]); }
    if (argc>3) iters=atoi(argv[3]);

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    int drv=0, rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("== GlassOS CPPN -- CUDA port ==\n");
    printf("GPU        : %s (sm_%d%d), %d SMs, %.1f GB\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem/1073741824.0);
    printf("clocks     : SM %.0f MHz, mem %.0f MHz, bus %d-bit -> %.1f GB/s theoretical\n",
           prop.clockRate/1000.0, prop.memoryClockRate/1000.0, prop.memoryBusWidth,
           2.0*prop.memoryClockRate*1000.0*(prop.memoryBusWidth/8)/1e9);
    printf("driver API : %d   runtime API : %d\n", drv, rt);
    printf("resolution : %dx%d   iterations : %d (after 10 warm-up)\n\n", W, H_, iters);

    Scene sc = buildScene(20260811u, 7.5f, (float)W/(float)H_);
    Uni u; u.resx=(float)W; u.resy=(float)H_; u.time=sc.time; u.vitality=sc.vitality;
    u.restless=sc.restless; u.dream=sc.dream; u.beat=sc.beat;
    u.px=sc.pointerX; u.py=sc.pointerY; u.pglow=sc.pointerGlow; u.alive=sc.alive; u.ditherAmp=0.015f;

    CHECK(cudaMemcpyToSymbol(c_G, sc.G, sizeof(sc.G)));
    float4* d_G; CHECK(cudaMalloc(&d_G, sizeof(sc.G)));
    CHECK(cudaMemcpy(d_G, sc.G, sizeof(sc.G), cudaMemcpyHostToDevice));
    uchar4* d_out; CHECK(cudaMalloc(&d_out, (size_t)W*H_*sizeof(uchar4)));
    unsigned char* h_out = (unsigned char*)malloc((size_t)W*H_*4);

    // ---- static resource usage + theoretical occupancy.
    // Nsight Compute needs elevated privileges on this machine (ERR_NVGPUCTRPERM),
    // so achieved occupancy / warp efficiency are unavailable. What follows is
    // measured on this GPU through the CUDA occupancy API instead.
    {
        cudaFuncAttributes fa[3];
        CHECK(cudaFuncGetAttributes(&fa[0], (const void*)render<0,0,0>));
        CHECK(cudaFuncGetAttributes(&fa[1], (const void*)render<1,1,0>));
        CHECK(cudaFuncGetAttributes(&fa[2], (const void*)render<1,1,1>));
        const char* sn[3]={"A naive","B const","C fast"};
        printf("-- static resources (cudaFuncGetAttributes) --\n");
        printf("%-10s %6s %8s %8s %10s\n","variant","regs","localB","constB","maxTPB");
        for(int v=0;v<3;v++)
            printf("%-10s %6d %8d %8d %10d\n", sn[v], fa[v].numRegs,
                   (int)fa[v].localSizeBytes, (int)fa[v].constSizeBytes, fa[v].maxThreadsPerBlock);
        printf("\n-- theoretical occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor) --\n");
        printf("%-10s", "threads");
        for(int v=0;v<3;v++) printf(" %12s", sn[v]);
        printf("\n");
        const int tp[] = {64,128,256,512,1024};
        for(int i=0;i<5;i++){
            printf("%-10d", tp[i]);
            for(int v=0;v<3;v++){
                int nb=0;
                const void* fp = (v==0)?(const void*)render<0,0,0>
                               :(v==1)?(const void*)render<1,1,0>:(const void*)render<1,1,1>;
                CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fp, tp[i], 0));
                double occ = (double)nb*tp[i]/prop.maxThreadsPerMultiProcessor;
                printf(" %5d blk %4.0f%%", nb, occ*100.0);
            }
            printf("\n");
        }
        printf("\n");
    }

    // ---- block-size sweep on variant C
    const int NB = 10, ROUNDS = 5;
    printf("-- block-size sweep (variant C, %dx%d, min over %d interleaved rounds of %d) --\n",
           W, H_, ROUNDS, iters);
    printf("%-10s %10s %10s %10s\n","block","ms","Mpix/s","thr/blk");
    const int bx[] = {8,16,16,32,32,32,64,64,128,256};
    const int by[] = {8, 4, 8, 2, 4, 8, 2, 4,  2,  1};
    float bms[NB];
    for (int i=0;i<NB;i++){ bms[i]=1e30f; for(int k=0;k<10;k++) timeOnce(2,d_out,W,H_,u,d_G,dim3(bx[i],by[i]),1); }
    CHECK(cudaDeviceSynchronize());
    for (int rd=0; rd<ROUNDS; rd++)
        for (int i=0;i<NB;i++){
            float ms = timeOnce(2, d_out, W, H_, u, d_G, dim3(bx[i],by[i]), iters);
            if (ms<bms[i]) bms[i]=ms;
        }
    float best=1e30f; int bbx=16,bby=8;
    for (int i=0;i<NB;i++){
        char lbl[16]; sprintf(lbl,"%dx%d",bx[i],by[i]);
        printf("%-10s %10.3f %10.1f %10d\n", lbl, bms[i], (double)W*H_/(bms[i]*1e3), bx[i]*by[i]);
        if (bms[i]<best){ best=bms[i]; bbx=bx[i]; bby=by[i]; }
    }
    printf("best block : %dx%d (%.3f ms)\n\n", bbx, bby, best);

    // ---- three variants at the best block size
    dim3 blk(bbx,bby);
    printf("-- kernel configurations (%dx%d, block %dx%d, min over %d interleaved rounds of %d) --\n",
           W,H_,bbx,bby,ROUNDS,iters);
    printf("   all six configurations are timed round-robin in one loop, so GPU\n"
           "   clock drift over the run cannot masquerade as a difference between them.\n");
    printf("%-46s %9s %10s %8s %11s\n","configuration","ms","Mpix/s","vs A","wrBW GB/s");
    // 0=A(global,no early)  3=const,no early  4=global,early  1=B(const,early)
    // 2=C(B+fastmath)       5=D(global,early,fastmath)
    const int vlist[6]={0,3,4,1,2,5}; float vms[6];
    const char* cname[6]={
        "A  global   + no early-out            (naive)",
        "   constant + no early-out",
        "   global   + early-out",
        "B  constant + early-out",
        "C  constant + early-out + intrinsics",
        "D  global   + early-out + intrinsics  (best)"};
    timeInterleaved(vlist,6,vms,d_out,W,H_,u,d_G,blk,iters,ROUNDS);
    float t0=vms[0];
    for (int i=0;i<6;i++){
        double wbw = (double)W*H_*4.0/(vms[i]*1e-3)/1e9;
        printf("%-46s %9.3f %10.1f %7.2fx %11.2f\n", cname[i], vms[i],
               (double)W*H_/(vms[i]*1e3), t0/vms[i], wbw);
    }
    // save each headline variant's frame
    {
        dim3 grd((W+blk.x-1)/blk.x,(H_+blk.y-1)/blk.y);
        for (int v=0;v<3;v++){
            if(v==0) render<0,0,0><<<grd,blk>>>(d_out,W,H_,u,d_G);
            else if(v==1) render<1,1,0><<<grd,blk>>>(d_out,W,H_,u,d_G);
            else render<1,1,1><<<grd,blk>>>(d_out,W,H_,u,d_G);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out,d_out,(size_t)W*H_*4,cudaMemcpyDeviceToHost));
            char path[64]; sprintf(path,"out_variant_%c.png",'A'+v);
            write_png_rgba(path,h_out,W,H_);
            char raw[64]; sprintf(raw,"cuda_%c.raw",'A'+v);
            FILE*f=fopen(raw,"wb"); fwrite(h_out,1,(size_t)W*H_*4,f); fclose(f);
        }
    }

    // second pass with the hash dither disabled, so the CPU cross-check can
    // separate genuine float-vs-double drift from the deliberate +-0.015 noise
    {
        Uni u2 = u; u2.ditherAmp = 0.f;
        dim3 grd((W+blk.x-1)/blk.x,(H_+blk.y-1)/blk.y);
        const char* rn[2]={"cuda_Bnd.raw","cuda_Cnd.raw"};
        for(int v=1;v<3;v++){
            if(v==1) render<1,1,0><<<grd,blk>>>(d_out,W,H_,u2,d_G);
            else     render<1,1,1><<<grd,blk>>>(d_out,W,H_,u2,d_G);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out,d_out,(size_t)W*H_*4,cudaMemcpyDeviceToHost));
            FILE*f=fopen(rn[v-1],"wb"); fwrite(h_out,1,(size_t)W*H_*4,f); fclose(f);
        }
        printf("wrote cuda_Bnd.raw / cuda_Cnd.raw (dither disabled)\n");
    }

    printf("\nwrote out_variant_A/B/C.png and cuda_A/B/C.raw (%dx%d RGBA8)\n", W, H_);

    // dump the packed genome block so the CPU reference and the WebGL harness
    // can consume byte-identical uniforms
    FILE*gf=fopen("scene.json","w");
    fprintf(gf,"{\"width\":%d,\"height\":%d,\"time\":%.9g,\"vitality\":%.9g,\"restless\":%.9g,"
               "\"dream\":%.9g,\"beat\":%.9g,\"pointer\":[%.9g,%.9g],\"pointerGlow\":%.9g,\"alive\":%d,\"G\":[",
            W,H_,u.time,u.vitality,u.restless,u.dream,u.beat,u.px,u.py,u.pglow,u.alive);
    for(int i=0;i<GFLOATS;i++) fprintf(gf,"%s%.9g", i?",":"", sc.G[i]);
    fprintf(gf,"]}\n"); fclose(gf);
    printf("wrote scene.json (uniform dump)\n");

    free(h_out); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_G));
    return 0;
}
