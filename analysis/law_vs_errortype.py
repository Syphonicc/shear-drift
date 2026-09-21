"""Does the (1/R-1)^p amplification hold for error types other than clock/window error?
Per-face excess RRT error (|dRRT/RRT| - |dTAWSS/TAWSS|) binned by X=1/R-1, cerebral wall,
for: window/clock error (detune, as in export_demo), frozen per-node bias, white noise.
Reply to W. Ding's point: a pure time-stretch integrated over its OWN period changes nothing."""
import numpy as np
d = np.load("../results_v2/wss_data.npz"); key=[k for k in d.files if d[k].ndim==3][0]
tau = d[key][80:120].astype(np.float64)*1060.0; N=tau.shape[0]
def biom(t):
    M=np.linalg.norm(t,axis=2).mean(0); S=np.linalg.norm(t.mean(0),axis=1); return M, S/M, 1/np.maximum(S,1e-30)
M0,R0,RRT0 = biom(tau); X = np.clip(1/R0-1,1e-9,None)
k = np.fft.fftfreq(N)*N
def detune(rho):   # surrogate runs (1+rho) fast, evaluated on the true-period grid (window mismatch)
    E = np.exp(2j*np.pi*np.outer(np.arange(N)*(1+rho),k)/N); return np.real(np.einsum('tk,kfc->tfc',E,np.fft.fft(tau,axis=0)))/N
def own_period(rho): # same stretched signal, but integrated over exactly one of ITS periods -> identical samples
    return tau.copy()
rms = np.sqrt((tau**2).sum(2).mean(0))
rng = np.random.default_rng(0)
cases = {
 "clock 5% (window mismatch)": detune(0.05),
 "clock 5%, own period": own_period(0.05),
 "frozen bias":  tau + 0.108*rms[None,:,None]*(rng.standard_normal((1,)+tau.shape[1:])/np.sqrt(3)),
 "white noise":  tau + 0.108*rms[None,:,None]*(rng.standard_normal(tau.shape)/np.sqrt(3)),
}
c,p = 0.194,0.641
bins=[0,0.02,0.1,0.3,1,10]
print("per-face median excess RRT error beyond TAWSS (%), by cancellation factor bin")
print("%-28s %6s |"%("case","eps")+"".join("%14s"%f"[{a},{b})" for a,b in zip(bins[:-1],bins[1:])))
for name,tp in cases.items():
    eps = np.linalg.norm(tp-tau)/np.linalg.norm(tau)
    M,R,RRT = biom(tp)
    ex = np.abs(RRT/RRT0-1) - np.abs(M/M0-1)
    row=[]
    for a,b in zip(bins[:-1],bins[1:]):
        m=(X>=a)&(X<b); row.append(100*np.median(ex[m]) if m.sum()>5 else np.nan)
    print("%-28s %5.1f%% |"%(name,100*eps)+"".join("%14.2f"%v for v in row))
row=[]
for a,b in zip(bins[:-1],bins[1:]):
    m=(X>=a)&(X<b); row.append(100*np.median(c*0.108*X[m]**p))
print("%-28s %6s |"%("law, eps=10.8%","")+"".join("%14.2f"%v for v in row))
print("faces per bin:", [int(((X>=a)&(X<b)).sum()) for a,b in zip(bins[:-1],bins[1:])])
