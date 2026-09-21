"""v3: frozen (time-coherent) spatial error, eps tuned so TAWSS rL2 = 13.54 % (RHSIA),
split into components parallel / transverse to the local mean WSS direction."""
import numpy as np
rng = np.random.default_rng(1)
d = np.load("../results_v2/wss_data.npz"); key=[k for k in d.files if d[k].ndim==3][0]
tau = d[key][80:120].astype(np.float64)*1060.0
def biom(t):
    M=np.linalg.norm(t,axis=2).mean(0); S=np.linalg.norm(t.mean(0),axis=1)
    return M, 0.5*(1-S/M), 1/np.maximum(S,1e-30)
def rl2(a,b): return 100*np.linalg.norm(a-b)/np.linalg.norm(b)
M0,O0,R0 = biom(tau)
rms = np.sqrt((tau**2).sum(2).mean(0))
mdir = tau.mean(0); mdir /= np.linalg.norm(mdir,axis=1,keepdims=True)+1e-30
def run(name, eps, mode, seeds=3):
    out=[]
    for s in range(seeds):
        g = np.random.default_rng(s).standard_normal((tau.shape[1],3))/np.sqrt(3)
        if mode=="par":   g = (g*mdir).sum(1,keepdims=True)*mdir*np.sqrt(3)
        if mode=="trans": g = g-(g*mdir).sum(1,keepdims=True)*mdir; g*=np.sqrt(1.5)
        tp = tau + eps*rms[:,None]*g[None]
        M,O,Rr = biom(tp)
        out.append([rl2(tp,tau), rl2(M,M0), rl2(O,O0), rl2(Rr,R0)])
    f,t,o,r = np.mean(out,0)
    print("   %-9s eps %.3f  field %5.1f  TAWSS %5.2f  OSI %6.2f  RRT %5.2f   OSI/TAWSS %5.2f  RRT/TAWSS %5.2f"%(name,eps,f,t,o,r,o/t,r/t))
for eps in (0.16,0.18,0.20):
    run("iso", eps, "iso")
run("parallel", 0.18, "par")
run("transv.", 0.18, "trans")
print("   RHSIA:               TAWSS 13.54  OSI  68.35  RRT 17.31   OSI/TAWSS  5.05  RRT/TAWSS  1.28")
