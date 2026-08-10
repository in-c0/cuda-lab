// Minimal PNG writer: stored (uncompressed) deflate blocks.
// No external dependencies -- keeps the build to a single nvcc invocation.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long png_crc_table[256];
static int png_crc_ready = 0;
static void png_make_crc(){
    for (unsigned long n=0;n<256;n++){
        unsigned long c=n;
        for(int k=0;k<8;k++) c = (c&1)? 0xedb88320UL ^ (c>>1) : (c>>1);
        png_crc_table[n]=c;
    }
    png_crc_ready=1;
}
static unsigned long png_crc(const unsigned char*buf,size_t len,unsigned long c){
    if(!png_crc_ready) png_make_crc();
    for(size_t n=0;n<len;n++) c = png_crc_table[(c^buf[n])&0xff] ^ (c>>8);
    return c;
}
static void png_be32(unsigned char*p,unsigned long v){
    p[0]=(unsigned char)(v>>24);p[1]=(unsigned char)(v>>16);p[2]=(unsigned char)(v>>8);p[3]=(unsigned char)v;
}
static void png_chunk(FILE*f,const char*type,const unsigned char*data,size_t len){
    unsigned char hdr[4]; png_be32(hdr,(unsigned long)len); fwrite(hdr,1,4,f);
    fwrite(type,1,4,f);
    if(len) fwrite(data,1,len,f);
    unsigned long c = png_crc((const unsigned char*)type,4,0xffffffffUL);
    if(len) c = png_crc(data,len,c);
    c ^= 0xffffffffUL;
    unsigned char cb[4]; png_be32(cb,c); fwrite(cb,1,4,f);
}
// rgba: w*h*4 bytes
static int write_png_rgba(const char*path,const unsigned char*rgba,int w,int h){
    FILE*f=fopen(path,"wb"); if(!f) return 0;
    static const unsigned char sig[8]={137,80,78,71,13,10,26,10};
    fwrite(sig,1,8,f);
    unsigned char ihdr[13];
    png_be32(ihdr,(unsigned long)w); png_be32(ihdr+4,(unsigned long)h);
    ihdr[8]=8; ihdr[9]=6; ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;
    png_chunk(f,"IHDR",ihdr,13);

    // raw scanlines with filter byte 0
    size_t raw_len = (size_t)h*((size_t)w*4+1);
    unsigned char*raw=(unsigned char*)malloc(raw_len);
    for(int y=0;y<h;y++){
        unsigned char*d = raw + (size_t)y*((size_t)w*4+1);
        d[0]=0;
        memcpy(d+1, rgba+(size_t)y*w*4, (size_t)w*4);
    }
    // zlib stream: header + stored deflate blocks + adler32
    size_t nblocks = (raw_len + 65534)/65535;
    size_t z_len = 2 + nblocks*5 + raw_len + 4;
    unsigned char*z=(unsigned char*)malloc(z_len);
    size_t zi=0; z[zi++]=0x78; z[zi++]=0x01;
    size_t off=0;
    while(off<raw_len){
        size_t n = raw_len-off; if(n>65535) n=65535;
        int final = (off+n>=raw_len);
        z[zi++]=(unsigned char)final;
        z[zi++]=(unsigned char)(n&0xff); z[zi++]=(unsigned char)(n>>8);
        z[zi++]=(unsigned char)((~n)&0xff); z[zi++]=(unsigned char)(((~n)>>8)&0xff);
        memcpy(z+zi, raw+off, n); zi+=n; off+=n;
    }
    unsigned long a1=1,a2=0;
    for(size_t i=0;i<raw_len;i++){ a1=(a1+raw[i])%65521; a2=(a2+a1)%65521; }
    png_be32(z+zi,(a2<<16)|a1); zi+=4;
    png_chunk(f,"IDAT",z,zi);
    png_chunk(f,"IEND",NULL,0);
    fclose(f); free(raw); free(z);
    return 1;
}
