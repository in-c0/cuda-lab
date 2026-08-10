// ============================================================================
// Independent CPU reference for the GlassOS CPPN shader.
//
// Deliberately NOT sharing device code with cppn.cu: transcribed a second time,
// straight from the GLSL in index.html. Templated on the scalar type so the
// same reference can be evaluated in double (ground truth) and in float
// (matching what the GPU actually computes) -- the gap between those two tells
// you how much of any CUDA-vs-reference difference is precision rather than a
// porting bug. Single-threaded; clarity over speed.
//
// Usage: cpu_ref.exe [W H]
//   pass 0: double, dither on   -> out_cpu_ref.png,  compared to cuda_A/B/C.raw
//   pass 1: double, dither off  -> compared to cuda_Bnd/Cnd.raw
//   pass 2: float,  dither off  -> compared to cuda_Bnd/Cnd.raw
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "genome.h"
#include "png.h"

template<typename T> static T tfract(T x){ return x - (T)floor((double)x); }
template<typename T> static T tclamp(T x,T a,T b){ return x<a?a:(x>b?b:x); }
template<typename T> static T tmix(T a,T b,T t){ return a+(b-a)*t; }
template<typename T> static T tstep(T e0,T e1,T x){
    T t = tclamp<T>((x-e0)/(e1-e0),(T)0,(T)1); return t*t*((T)3-(T)2*t);
}
template<typename T> static T thash(T x,T y){
    T px = tfract<T>(x*(T)123.34), py = tfract<T>(y*(T)345.45);
    T d = px*(px+(T)34.345) + py*(py+(T)34.345);
    px += d; py += d;
    return tfract<T>(px*py);
}
template<typename T> static T tnoise(T x,T y){
    T ix=(T)floor((double)x), iy=(T)floor((double)y), fx=x-ix, fy=y-iy;
    T a=thash<T>(ix,iy), b=thash<T>(ix+(T)1,iy), c=thash<T>(ix,iy+(T)1), d=thash<T>(ix+(T)1,iy+(T)1);
    T ux=fx*fx*((T)3-(T)2*fx), uy=fy*fy*((T)3-(T)2*fy);
    return tmix<T>(tmix<T>(a,b,ux), tmix<T>(c,d,ux), uy);
}
template<typename T> static T tfbm(T x,T y){
    T v=(T)0,a=(T)0.5;
    for(int i=0;i<5;i++){ v+=a*tnoise<T>(x,y); x=x*(T)2.03+(T)7.1; y=y*(T)2.03+(T)7.1; a*=(T)0.5; }
    return v;
}
template<typename T> static T tactF(T a,T x){
    if(a<(T)0.5) return (T)sin((double)x);
    if(a<(T)1.5) return (T)(exp(-(double)x*(double)x)*2.0-1.0);
    if(a<(T)2.5) return (T)tanh((double)x);
    return (T)(fabs((double)tfract<T>(x*(T)0.25+(T)0.25)-0.5)*4.0-1.0);
}
template<typename T> static T tcppn(const float*G,int B,T lpx,T lpy,T lr,T ica,T isa,T tt){
    // B is a vec4 slot index; the float array index is 4*B.
    T acc = (T)G[4*(B+15)+2];
    for(int i=0;i<GH;i++){
        const float* n0 = G + 4*(B+2+2*i);
        const float* n1 = G + 4*(B+3+2*i);
        T pre = (T)n0[1] + (T)n0[2]*tt + (T)n0[3]*lpx + (T)n1[0]*lpy
              + (T)n1[1]*lr + (T)n1[2]*ica + (T)n1[3]*isa;
        T h = tactF<T>((T)n0[0], pre);
        T ow = (T)((i<4) ? G[4*(B+14)+i] : G[4*(B+15)+(i-4)]);
        acc += ow*h;
    }
    return acc;
}

template<typename T>
static void renderRef(const Scene& sc, int W, int H_, double ditherAmp, unsigned char* img){
    const float* G = sc.G;
    T resx=(T)W, resy=(T)H_, time=(T)sc.time, vit=(T)sc.vitality, res_=(T)sc.restless,
      dream=(T)sc.dream, beat=(T)sc.beat, pxu=(T)sc.pointerX, pyu=(T)sc.pointerY, pglow=(T)sc.pointerGlow;
    T dA = (T)ditherAmp;
    for(int iy=0; iy<H_; iy++){
        for(int ix=0; ix<W; ix++){
            T fragx=(T)ix+(T)0.5, fragy=(T)(H_-1-iy)+(T)0.5;
            T uvx=fragx/resx, uvy=fragy/resy;
            T p_x=(fragx-(T)0.5*resx)/resy, p_y=(fragy-(T)0.5*resy)/resy;
            T r=(T)sqrt((double)(p_x*p_x+p_y*p_y)), ang=(T)atan2((double)p_y,(double)p_x);

            T tR=(T)1.00*((T)0.25+(T)0.75*vit)+(T)0.62*(T)0.45+(T)0.30*((T)0.25+(T)0.65*res_)+(T)0.02;
            T tG=(T)0.80*((T)0.25+(T)0.75*vit)+(T)0.48*(T)0.45+(T)0.50*((T)0.25+(T)0.65*res_)+(T)0.02;
            T tB=(T)0.45*((T)0.25+(T)0.75*vit)+(T)1.00*(T)0.45+(T)1.00*((T)0.25+(T)0.65*res_)+(T)0.02;
            T tl=(T)sqrt((double)(tR*tR+tG*tG+tB*tB)); tR/=tl; tG/=tl; tB/=tl;
            T tcR=tmix<T>(tR,(T)0.30,(T)0.5*dream), tcG=tmix<T>(tG,(T)0.50,(T)0.5*dream), tcB=tmix<T>(tB,(T)1.00,(T)0.5*dream);

            T cR=(T)0,cG=(T)0,cB=(T)0;
            T driftT=time*(T)0.035;
            T neb=(T)pow((double)tfbm<T>(p_x*(T)1.6+driftT, p_y*(T)1.6-driftT*(T)0.7),1.7);
            T neb2=tfbm<T>(p_x*(T)3.4-driftT*(T)1.3, p_y*(T)3.4-driftT);
            cR+=tmix<T>((T)0.005,tcR*(T)0.09,neb)+tR*(T)0.03*neb2*((T)0.3+(T)0.5*res_);
            cG+=tmix<T>((T)0.007,tcG*(T)0.09,neb)+tG*(T)0.03*neb2*((T)0.3+(T)0.5*res_);
            cB+=tmix<T>((T)0.017,tcB*(T)0.09,neb)+tB*(T)0.03*neb2*((T)0.3+(T)0.5*res_);

            T R=(T)0.225+(T)0.014*beat*((T)0.5+vit)+(T)0.007*(T)sin((double)time*0.6);
            T z=(T)sqrt((double)(R*R-r*r>(T)0?R*R-r*r:(T)0))/(R>(T)1e-4?R:(T)1e-4);
            T inside=tstep<T>(R+(T)0.004,R-(T)0.004,r);
            T fres=(T)pow(1.0-(double)tclamp<T>(z,(T)0,(T)1),2.5);
            T spx=p_x*((T)5.5+(T)2.0*z), spy=p_y*((T)5.5+(T)2.0*z);
            T wx=(T)cos((double)(ang*(T)2+time*(T)0.4))*((T)0.35*z);
            T wy=(T)sin((double)(ang*(T)2-time*(T)0.3))*((T)0.35*z);
            T ca=(T)0.010+(T)0.018*fres+(T)0.006*res_;
            T bx=spx+wx+time*(T)0.10, by=spy+wy;
            T cauR=tfbm<T>(bx+ca,by+ca), cauG=tfbm<T>(bx,by), cauB=tfbm<T>(bx-ca,by-ca);
            cauR*=cauR; cauG*=cauG; cauB*=cauB;
            T zf=(T)0.30+(T)0.9*z;
            T coR=tR*((T)0.10+(T)1.45*cauR)*inside*zf;
            T coG=tG*((T)0.10+(T)1.45*cauG)*inside*zf;
            T coB=tB*((T)0.10+(T)1.45*cauB)*inside*zf;
            T k2=inside*((T)0.05+(T)0.15*vit)*((T)0.4+(T)0.7*z);
            coR+=tR*k2; coG+=tG*k2; coB+=tB*k2;
            T rr = r>(T)0?r:(T)0;
            T k3=beat*inside*(T)exp(-(double)rr*8.0)*((T)0.25+(T)0.40*vit);
            coR+=tR*k3; coG+=tG*k3; coB+=tB*k3;
            coR+=(tmix<T>((T)1.00,tR,(T)0.55)*fres*ca*(T)24.0+tR*fres*(T)1.2)*inside*(T)0.8;
            coG+=(tmix<T>((T)0.62,tG,(T)0.55)*fres*ca*(T)24.0+tG*fres*(T)1.2)*inside*(T)0.8;
            coB+=(tmix<T>((T)0.82,tB,(T)0.55)*fres*ca*(T)24.0+tB*fres*(T)1.2)*inside*(T)0.8;
            cR+=coR; cG+=coG; cB+=coB;
            T hd = r-R > (T)0 ? r-R : (T)0;
            T halo=(T)exp(-(double)hd*8.5)*((T)0.10+(T)0.13*vit)*((T)0.7+(T)0.3*beat);
            cR+=tR*halo; cG+=tG*halo; cB+=tB*halo;

            T disp=(T)0.020+(T)0.020*res_;
            for(int gI=0; gI<MAXG && gI<sc.alive; gI++){
                int B=gI*16;
                const float* m0=G+4*B; const float* m1=G+4*(B+1);
                T lpx=(p_x-(T)m0[0])/(T)m0[2], lpy=(p_y-(T)m0[1])/(T)m0[2];
                T lr=(T)sqrt((double)(lpx*lpx+lpy*lpy));
                if(lr>(T)1.6) continue;
                T ica= lr>(T)1e-4? lpx/lr : (T)1;
                T isa= lr>(T)1e-4? lpy/lr : (T)0;
                T tt=time*(T)m1[3];
                T fR=tcppn<T>(G,B,lpx,lpy,lr+disp,ica,isa,tt);
                T fB=tcppn<T>(G,B,lpx,lpy,lr-disp,ica,isa,tt);
                T fG=(T)0.5*(fR+fB);
                T lw=(T)G[4*(B+15)+3];
                T lR=(T)1-tstep<T>((T)0,lw,(T)fabs((double)fR));
                T lG=(T)1-tstep<T>((T)0,lw,(T)fabs((double)fG));
                T lB=(T)1-tstep<T>((T)0,lw,(T)fabs((double)fB));
                T env=(T)exp(-(double)(lr*lr)*0.9);
                T gcR=tmix<T>((T)m1[0],(T)0.55,(T)0.35*dream);
                T gcG=tmix<T>((T)m1[1],(T)0.60,(T)0.35*dream);
                T gcB=tmix<T>((T)m1[2],(T)1.00,(T)0.35*dream);
                T amp=env*(T)m0[3]*((T)1.55+(T)0.85*dream);
                T aura=(T)0.16*(T)exp(-fabs((double)fG)*2.0)*env*(T)m0[3];
                cR+=lR*gcR*amp+gcR*aura;
                cG+=lG*gcG*amp+gcG*aura;
                cB+=lB*gcB*amp+gcB*aura;
            }

            T pdx=p_x-pxu, pdy=p_y-pyu, pd=(T)sqrt((double)(pdx*pdx+pdy*pdy));
            T spark=pglow*(T)exp(-(double)(pd*pd)*26.0);
            cR+=(tR*(T)0.6+(T)0.25)*spark*(T)1.3;
            cG+=(tG*(T)0.6+(T)0.35)*spark*(T)1.3;
            cB+=(tB*(T)0.6+(T)0.50)*spark*(T)1.3;
            T ring=pglow*(T)0.25*((T)1-tstep<T>((T)0,(T)0.02,(T)fabs((double)(pd-(T)0.06-(T)0.04*(T)sin((double)time*3.0)))));
            cR+=tR*ring; cG+=tG*ring; cB+=tB*ring;

            T vig=(T)1-(T)0.55*tstep<T>((T)0.35,(T)1.35,r);
            cR*=vig; cG*=vig; cB*=vig;
            cR=(T)pow(1.0-exp(-(double)cR*1.08),0.4545);
            cG=(T)pow(1.0-exp(-(double)cG*1.08),0.4545);
            cB=(T)pow(1.0-exp(-(double)cB*1.08),0.4545);
            T dith=(thash<T>(uvx*resx+time, uvy*resy+time)-(T)0.5)*dA;
            cR+=dith; cG+=dith; cB+=dith;

            size_t o=((size_t)iy*W+ix)*4;
            img[o+0]=(unsigned char)(tclamp<T>(cR,(T)0,(T)1)*(T)255+(T)0.5);
            img[o+1]=(unsigned char)(tclamp<T>(cG,(T)0,(T)1)*(T)255+(T)0.5);
            img[o+2]=(unsigned char)(tclamp<T>(cB,(T)0,(T)1)*(T)255+(T)0.5);
            img[o+3]=255;
        }
    }
}

static void compare(const char* label, const unsigned char* img, const char* rawpath, int W, int H_){
    size_t n=(size_t)W*H_*4;
    FILE*f=fopen(rawpath,"rb");
    if(!f){ printf("  %-28s : %s not found (run cppn.exe first)\n",label,rawpath); return; }
    unsigned char* other=(unsigned char*)malloc(n);
    size_t got=fread(other,1,n,f); fclose(f);
    if(got!=n){ printf("  %-28s : size mismatch\n",label); free(other); return; }
    long long maxd=0, cnt=0, over1=0, over2=0; double sum=0, ma=0, mb=0;
    for(size_t i=0;i<n;i+=4){
        for(int c=0;c<3;c++){
            int d=abs((int)img[i+c]-(int)other[i+c]);
            if(d>maxd) maxd=d; sum+=d; cnt++;
            if(d>1) over1++; if(d>2) over2++;
        }
        ma += 0.2126*img[i]+0.7152*img[i+1]+0.0722*img[i+2];
        mb += 0.2126*other[i]+0.7152*other[i+1]+0.0722*other[i+2];
    }
    size_t npix=(size_t)W*H_; ma/=npix; mb/=npix;
    double va=0,vb=0,cov=0;
    for(size_t i=0;i<n;i+=4){
        double la=0.2126*img[i]+0.7152*img[i+1]+0.0722*img[i+2];
        double lb=0.2126*other[i]+0.7152*other[i+1]+0.0722*other[i+2];
        va+=(la-ma)*(la-ma); vb+=(lb-mb)*(lb-mb); cov+=(la-ma)*(lb-mb);
    }
    va/=npix-1; vb/=npix-1; cov/=npix-1;
    double C1=6.5025, C2=58.5225;
    double ssim=((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2));
    printf("  %-28s : max|d|=%2lld  mean|d|=%.4f  %%>1LSB=%7.4f  %%>2LSB=%7.4f  SSIM(luma)=%.6f\n",
           label, maxd, sum/cnt, 100.0*over1/cnt, 100.0*over2/cnt, ssim);
    free(other);
}

int main(int argc,char**argv){
    int W=1920,H_=1080;
    if(argc>2){ W=atoi(argv[1]); H_=atoi(argv[2]); }
    Scene sc = buildScene(20260811u, 7.5f, (float)W/(float)H_);
    unsigned char* img=(unsigned char*)malloc((size_t)W*H_*4);
    printf("CPU reference %dx%d  (all differences in 8-bit LSB per channel)\n\n", W, H_);

    printf("[1] double precision, dither ON  (full shader)\n");
    renderRef<double>(sc, W, H_, 0.015, img);
    write_png_rgba("out_cpu_ref.png", img, W, H_);
    compare("CUDA A (naive)",       img, "cuda_A.raw", W, H_);
    compare("CUDA B (constant)",    img, "cuda_B.raw", W, H_);
    compare("CUDA C (fast-math)",   img, "cuda_C.raw", W, H_);

    printf("[2] double precision, dither OFF\n");
    renderRef<double>(sc, W, H_, 0.0, img);
    compare("CUDA B (no dither)",   img, "cuda_Bnd.raw", W, H_);
    compare("CUDA C (no dither)",   img, "cuda_Cnd.raw", W, H_);

    printf("[3] float precision, dither OFF  (same arithmetic width as the GPU)\n");
    renderRef<float>(sc, W, H_, 0.0, img);
    write_png_rgba("out_cpu_ref_float.png", img, W, H_);
    compare("CUDA B (no dither)",   img, "cuda_Bnd.raw", W, H_);
    compare("CUDA C (no dither)",   img, "cuda_Cnd.raw", W, H_);

    free(img);
    return 0;
}
