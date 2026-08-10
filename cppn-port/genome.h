// Deterministic scene construction for the GlassOS CPPN port.
//
// The WebGL original seeds its population from Math.random() and evolves it
// over time. For a reproducible benchmark/correctness artifact we need the
// exact same *packing* but a fixed population, so this file re-implements
// the genome factory (randomGenome + viable + makeViable + makeOrganism +
// organismPos + packOrganisms from index.html) on top of a fixed-seed
// xorshift32 PRNG. Host-only code: the result is 6*16 float4 = 384 floats,
// which is exactly what the GLSL `uniform vec4 u_G[96]` receives.
#pragma once
#include <math.h>
#include <string.h>

#define GH    6    // hidden nodes per genome  (H in index.html)
#define MAXG  6    // organisms rendered at once
#define GSLOTS (MAXG*16)      // float4 slots
#define GFLOATS (GSLOTS*4)    // 384 floats

struct Rng {
    unsigned int s;
    explicit Rng(unsigned int seed) : s(seed ? seed : 1u) {}
    unsigned int next() { s ^= s<<13; s ^= s>>17; s ^= s<<5; return s; }
    // uniform [0,1) with 24 bits of mantissa, mirroring Math.random()'s role
    double uni() { return (next() >> 8) * (1.0/16777216.0); }
    double gauss(double sigma) { // Box-Muller, same as g0() in index.html
        double u = 0, v = 0;
        while (u == 0) u = uni();
        while (v == 0) v = uni();
        return sqrt(-2.0*log(u)) * cos(2.0*3.14159265358979323846*v) * sigma;
    }
};

struct Genome {
    double n[GH][8]; // act, bias, tW, xW, yW, rW, caW, saW
    double o[GH+1];  // output weights + bias
    double lw;
};

static const int ACTS[6] = {0,0,1,1,2,3}; // palette biased to sin/gauss

static double actFn(int a, double x) {
    switch (a) {
        case 0: return sin(x);
        case 1: return exp(-x*x)*2.0 - 1.0;
        case 2: return tanh(x);
        default: { double f = fmod(fmod(x*0.25+0.25, 1.0)+1.0, 1.0); return fabs(f-0.5)*4.0 - 1.0; }
    }
}

static Genome randomGenome(Rng& r) {
    Genome g;
    for (int i = 0; i < GH; i++) {
        g.n[i][0] = ACTS[(int)(r.uni()*6.0)];
        g.n[i][1] = r.gauss(0.7);
        g.n[i][2] = r.uni() < 0.7 ? r.gauss(0.5) : 0.0;
        g.n[i][3] = r.gauss(1.3);
        g.n[i][4] = r.gauss(1.3);
        g.n[i][5] = (r.uni()*2.0-1.0)*4.5;
        g.n[i][6] = r.gauss(1.1);
        g.n[i][7] = r.gauss(1.1);
    }
    for (int i = 0; i < GH; i++) g.o[i] = r.gauss(0.9);
    g.o[GH] = r.gauss(0.4);
    g.lw = 0.07 + r.uni()*0.07;
    return g;
}

static double evalGenome(const Genome& g, double x, double y, double t) {
    double rr = sqrt(x*x + y*y);
    double ca = rr > 1e-4 ? x/rr : 1.0, sa = rr > 1e-4 ? y/rr : 0.0;
    double acc = g.o[GH];
    for (int i = 0; i < GH; i++) {
        double pre = g.n[i][1] + g.n[i][2]*t + g.n[i][3]*x + g.n[i][4]*y
                   + g.n[i][5]*rr + g.n[i][6]*ca + g.n[i][7]*sa;
        acc += g.o[i]*actFn((int)g.n[i][0], pre);
    }
    return acc;
}

static int sgn(double v){ return v > 0 ? 1 : (v < 0 ? -1 : 0); }

static bool viable(const Genome& g) {
    int near = 0, total = 0, signChanges = 0;
    double maxAbs = 0, prevRow[7]; bool havePrev = false;
    for (int j = 0; j < 7; j++) {
        double row[7];
        for (int i = 0; i < 7; i++) {
            double x = -1.1 + 2.2*i/6.0, y = -1.1 + 2.2*j/6.0;
            double f = evalGenome(g, x, y, 0.0);
            row[i] = f; total++;
            if (fabs(f) < g.lw*2) near++;
            if (fabs(f) > maxAbs) maxAbs = fabs(f);
            if (i > 0 && sgn(row[i-1]) != sgn(f)) signChanges++;
            if (havePrev && sgn(prevRow[i]) != sgn(f)) signChanges++;
        }
        memcpy(prevRow, row, sizeof(row)); havePrev = true;
    }
    double frac = (double)near/total;
    return frac > 0.02 && frac < 0.5 && signChanges >= 4 && maxAbs > 0.35;
}

static Genome makeViable(Rng& r) {
    Genome g = randomGenome(r);
    for (int k = 0; k < 24 && !viable(g); k++) g = randomGenome(r);
    return g;
}

static void hslToRgb(double h, double s, double l, double* out) {
    for (int idx = 0; idx < 3; idx++) {
        double n = (idx==0)?0.0:((idx==1)?8.0:4.0);
        double k = fmod(n + h*12.0, 12.0);
        double m = k-3.0; if (9.0-k < m) m = 9.0-k; if (1.0 < m) m = 1.0; if (m < -1.0) m = -1.0;
        out[idx] = l - s*(l < 1.0-l ? l : 1.0-l)*m;
    }
}

// Scene uniforms, matching the GLSL uniform block.
struct Scene {
    float time, vitality, restless, dream, beat;
    float pointerX, pointerY, pointerGlow;
    int alive;
    float G[GFLOATS];
};

// Build a fixed, deterministic scene. seed selects the population.
static Scene buildScene(unsigned int seed, float time, float aspect) {
    Rng r(seed);
    Scene sc;
    sc.time = time;
    sc.vitality  = 0.62f;
    sc.restless  = 0.28f;
    sc.dream     = 0.0f;
    // beat = pow(|sin(simT*(0.9+0.5*vitality))|, 5)  -- as in stepAndRender()
    sc.beat = (float)pow(fabs(sin((double)time*(0.9+0.5*0.62))), 5.0);
    sc.pointerX = 0.21f; sc.pointerY = -0.13f; sc.pointerGlow = 0.55f;
    sc.alive = MAXG;
    memset(sc.G, 0, sizeof(sc.G));

    double ex = aspect*0.5 - 0.20; if (ex > 0.60) ex = 0.60;
    double ey = 0.335;
    double simT = time;

    for (int i = 0; i < MAXG; i++) {
        Genome g = makeViable(r);
        double hue = r.uni();
        double tint[3]; hslToRgb(hue, 0.75, 0.68, tint);
        double thetaJit = r.gauss(0.12);
        double radius   = 0.155 + r.uni()*0.045;
        double timeRate = 0.5 + r.uni()*0.8;
        double ph1 = r.uni()*6.28, ph2 = r.uni()*6.28;
        double energy = 1.0, feedGlow = 0.0, age = 2.5; // settled, healthy, not dying

        double th = ((double)i/MAXG)*2.0*3.14159265358979323846 - 3.14159265358979323846/2.0 + thetaJit;
        double px = cos(th)*ex + sin(simT*0.11 + ph1)*0.045;
        double py = sin(th)*ey + cos(simT*0.13 + ph2)*0.04;

        double fade = age/2.0; if (fade > 1) fade = 1;
        fade *= 0.65 + 0.35*(energy/1.4 > 1 ? 1 : energy/1.4);
        fade *= 1 + 0.4*feedGlow;

        int fb = i*64;
        sc.G[fb+0]=(float)px; sc.G[fb+1]=(float)py;
        sc.G[fb+2]=(float)(radius*(1+0.1*feedGlow));
        sc.G[fb+3]=(float)fade;
        sc.G[fb+4]=(float)tint[0]; sc.G[fb+5]=(float)tint[1]; sc.G[fb+6]=(float)tint[2];
        sc.G[fb+7]=(float)timeRate;
        for (int k = 0; k < GH; k++)
            for (int w = 0; w < 8; w++) sc.G[fb+8+k*8+w] = (float)g.n[k][w];
        for (int k = 0; k < 4; k++) sc.G[fb+56+k] = (float)g.o[k];
        sc.G[fb+60]=(float)g.o[4]; sc.G[fb+61]=(float)g.o[5];
        sc.G[fb+62]=(float)g.o[GH]; sc.G[fb+63]=(float)g.lw;
    }
    return sc;
}
