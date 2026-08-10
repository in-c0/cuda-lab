#!/usr/bin/env python
"""
Build a standalone WebGL2 page that runs the ORIGINAL GlassOS fragment shader,
verbatim, on the same uniforms the CUDA program used -- so the CUDA output can
be compared to the real GLSL apples-to-apples.

  python make_harness.py            # writes glsl_harness.html from scene.json
  python make_harness.py --compare  # diffs glsl_shot.png against cuda_*.raw

Between those two steps, screenshot the harness with headless Edge:

  & "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe" `
      --headless=new --use-angle=d3d11 --hide-scrollbars `
      --force-device-scale-factor=1 --window-size=1920,1080 `
      --virtual-time-budget=30000 `
      --screenshot="D:\\Projects\\cuda-lab\\cppn-port\\glsl_shot.png" `
      --user-data-dir="$env:TEMP\\edgeglsl" `
      "file:///D:/Projects/cuda-lab/cppn-port/glsl_harness.html"

Swap --screenshot for --dump-dom to read back the WebGL renderer string and the
EXT_disjoint_timer_query_webgl2 GPU timing that the page also collects.

NOTE: do not use the in-app browser pane on this machine; headless Edge only.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHADER_SRC = r"D:\Projects\glassos-webgl\index.html"

PAGE = """<!doctype html><html><head><meta charset=utf8>
<style>html,body{margin:0;padding:0;background:#000;overflow:hidden}canvas{display:block}</style>
</head><body><canvas id=c></canvas><script>
const S = %(scene)s;
const cv=document.getElementById('c');
cv.width=S.width; cv.height=S.height;
cv.style.width=S.width+'px'; cv.style.height=S.height+'px';
const gl=cv.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
if(!gl){ document.title='NOWEBGL'; throw new Error('no webgl2'); }
const vs=`#version 300 es
in vec4 a_position;
void main(){ gl_Position=a_position; }`;
const fs=%(fs)s;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);
 if(!gl.getShaderParameter(x,gl.COMPILE_STATUS)){document.title='ERR';throw new Error(gl.getShaderInfoLog(x));}return x;}
const pr=gl.createProgram();
gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs)); gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(pr); gl.useProgram(pr);
const buf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buf);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
const ap=gl.getAttribLocation(pr,'a_position'); gl.enableVertexAttribArray(ap);
gl.vertexAttribPointer(ap,2,gl.FLOAT,false,0,0);
const U=n=>gl.getUniformLocation(pr,n);
gl.viewport(0,0,S.width,S.height);
gl.uniform2f(U('u_res'),S.width,S.height);
gl.uniform1f(U('u_time'),S.time);
gl.uniform1f(U('u_vitality'),S.vitality);
gl.uniform1f(U('u_restless'),S.restless);
gl.uniform1f(U('u_dream'),S.dream);
gl.uniform1f(U('u_beat'),S.beat);
gl.uniform2f(U('u_pointer'),S.pointer[0],S.pointer[1]);
gl.uniform1f(U('u_pointerGlow'),S.pointerGlow);
gl.uniform1i(U('u_alive'),S.alive);
gl.uniform4fv(U('u_G'),new Float32Array(S.G));
gl.drawArrays(gl.TRIANGLES,0,6);
gl.finish();
const dbg=gl.getExtension('WEBGL_debug_renderer_info');
let info='RENDERER='+gl.getParameter(gl.RENDERER);
if(dbg) info+=' | UNMASKED='+gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
function report(s){ document.title=s; const d=document.createElement('pre'); d.id='info';
  d.textContent=s; d.style.cssText='position:fixed;left:-9999px;top:0'; document.body.appendChild(d); }
const ext=gl.getExtension('EXT_disjoint_timer_query_webgl2');
if(ext){
  const N=200, q=gl.createQuery();
  for(let i=0;i<20;i++) gl.drawArrays(gl.TRIANGLES,0,6);
  gl.finish();
  gl.beginQuery(ext.TIME_ELAPSED_EXT,q);
  for(let i=0;i<N;i++) gl.drawArrays(gl.TRIANGLES,0,6);
  gl.endQuery(ext.TIME_ELAPSED_EXT);
  gl.finish();
  const poll=()=>{
    if(gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE)){
      const ns=gl.getQueryParameter(q,gl.QUERY_RESULT);
      report(info+' | TIMER_ELAPSED_ms_per_draw='+(ns/1e6/N).toFixed(4)+' | N='+N);
    } else setTimeout(poll,50);
  };
  poll();
} else report(info+' | TIMER=unavailable');
</script></body></html>"""


def build():
    scene = json.load(open(os.path.join(HERE, "scene.json")))
    src = open(SHADER_SRC, encoding="utf8").read()
    m = re.search(r"const fsSource = `(.*?)`;", src, re.S)
    if not m:
        sys.exit("could not find fsSource in " + SHADER_SRC)
    # the only edits: resolve the two JS template substitutions. The GLSL itself
    # is byte-for-byte what the live page compiles.
    fs = (m.group(1).replace("${MAXG * 16}", "96")
                    .replace("${H}", "6")
                    .replace("${MAXG}", "6"))
    out = PAGE % {"scene": json.dumps(scene), "fs": json.dumps(fs)}
    path = os.path.join(HERE, "glsl_harness.html")
    open(path, "w", encoding="utf8").write(out)
    print("wrote", path, len(out), "bytes")


def compare():
    import numpy as np
    from PIL import Image
    g = np.asarray(Image.open(os.path.join(HERE, "glsl_shot.png")).convert("RGB")).astype(np.int32)
    H, W = g.shape[:2]
    lines = []
    for name in ("cuda_A.raw", "cuda_B.raw", "cuda_C.raw"):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            continue
        c = np.fromfile(p, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].astype(np.int32)
        d = np.abs(g - c)
        la = 0.2126*g[:, :, 0] + 0.7152*g[:, :, 1] + 0.0722*g[:, :, 2]
        lb = 0.2126*c[:, :, 0] + 0.7152*c[:, :, 1] + 0.0722*c[:, :, 2]
        ma, mb = la.mean(), lb.mean()
        va, vb = la.var(ddof=1), lb.var(ddof=1)
        cov = ((la-ma)*(lb-mb)).sum()/(la.size-1)
        C1, C2 = 6.5025, 58.5225
        ssim = ((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2))
        line = (f"{name} vs GLSL (WebGL2/ANGLE-D3D11, same RTX 2080): "
                f"max|d|={d.max()}  mean|d|={d.mean():.4f}  "
                f"%>1LSB={(d > 1).mean()*100:.4f}  %>2LSB={(d > 2).mean()*100:.4f}  "
                f"SSIM(luma)={ssim:.6f}")
        print(line)
        lines.append(line)
    open(os.path.join(HERE, "glsl_compare.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    compare() if "--compare" in sys.argv else build()
