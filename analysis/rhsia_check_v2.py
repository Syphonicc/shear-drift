"""
v2: which error *type* reproduces RHSIA's OSI/TAWSS = 5.05 ?
Perturb the real cerebral WSS (cycle 3) three ways at matched field-L2 eps,
then compute RHSIA-style rL2 for TAWSS/OSI/RRT.
  drift  : clock drift (structured, from data/fields.bin, done in v1)
  white  : Gaussian noise per face per snapshot, std = eps * rms|tau|(face)  (unstructured, time-white)
  frozen : Gaussian noise per face, constant in time (pure spatial bias)
  smooth : temporal smoothing (box, 3 samples)
"""
import numpy as np
rng = np.random.default_rng(0)
d = np.load("../results_v2/wss_data.npz")
key = [k for k in d.files if d[k].ndim == 3][0]
tau = d[key][80:120].astype(np.float64) * 1060.0          # (T,F,3) Pa
T = tau.shape[0]

def biom(t):
    M = np.linalg.norm(t, axis=2).mean(0)                    # TAWSS
    S = np.linalg.norm(t.mean(0), axis=1)
    R = S / M
    OSI = 0.5*(1-R)
    RRT = 1.0/np.maximum(S, 1e-30)
    return M, OSI, RRT
def rl2(a,b): return 100*np.linalg.norm(a-b)/np.linalg.norm(b)
def report(name, tp):
    eps = rl2(tp, tau)
    M0,O0,Rr0 = biom(tau); M,O,Rr = biom(tp)
    e = [rl2(M,M0), rl2(O,O0), rl2(Rr,Rr0)]
    print("   %-7s field %5.1f  TAWSS %5.2f  OSI %6.2f  RRT %5.2f   OSI/TAWSS %5.2f  RRT/TAWSS %5.2f"
          % (name, eps, *e, e[1]/e[0], e[2]/e[0]))

rms = np.sqrt((tau**2).sum(2).mean(0))[None,:,None]        # (1,F,1)
print("perturbation of real cerebral WSS, RHSIA-style rL2 (%)")
for eps in (0.05, 0.10, 0.135, 0.20):
    print(" eps target %.3f" % eps)
    n = rng.standard_normal(tau.shape)/np.sqrt(3)
    report("white", tau + eps*rms*n)
    nf = rng.standard_normal((1,)+tau.shape[1:])/np.sqrt(3)
    report("frozen", tau + eps*rms*nf)
k = 3
sm = np.stack([np.roll(tau, s, axis=0) for s in range(-(k//2), k//2+1)]).mean(0)
report("smooth3", sm)
print("   RHSIA:  field ~?     TAWSS 13.54  OSI  68.35  RRT 17.31   OSI/TAWSS  5.05  RRT/TAWSS  1.28")
