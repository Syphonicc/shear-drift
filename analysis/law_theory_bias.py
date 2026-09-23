"""General first-order formula, any perturbation d(t) of the WSS vector tau(t) at one face:
   dR/R = s_hat . <d> / <|tau|>R  -  <tau_hat . d> / <|tau|>      (<.> = time mean, s_hat = direction of <tau>)
Frozen bias d = b (constant in time):  dR/R = b . v,  v = s_hat/(R<|tau|>) - <tau_hat>/<|tau|>.
Isotropic random b of size |b|:  rms(dR/R) = |b| |v| / sqrt(3).
1D limit: v = (1/<|tau|>) [ 1/R - (f+ - f-) ]  =>  (dR/R)/eps_b = X + 2 f-   (X = 1/R-1, f- = fraction of time reversed).
Check on the cerebral wall against the frozen-bias experiment."""
import numpy as np
d = np.load("../results_v2/wss_data.npz"); key=[k for k in d.files if d[k].ndim==3][0]
tau = d[key][80:120].astype(np.float64)*1060.0
mag = np.linalg.norm(tau,axis=2); M = mag.mean(0); Sv = tau.mean(0); S = np.linalg.norm(Sv,axis=1); R = S/M
that = tau/np.maximum(mag,1e-30)[:,:,None]; mt = that.mean(0)
v = Sv/np.maximum(S,1e-30)[:,None]/np.maximum(S,1e-30)[:,None] - mt/M[:,None]     # s_hat/S - <tau_hat>/M
rms = np.sqrt((tau**2).sum(2).mean(0)); eps=0.108
pred = eps*rms*np.linalg.norm(v,axis=1)/np.sqrt(3)                                  # |b| = eps*rms
# measured, averaged over 8 random bias directions
rng=np.random.default_rng(0); meas=[]
for s in range(8):
    b = eps*rms[:,None]*rng.standard_normal((tau.shape[1],3))/np.sqrt(3)
    tp = tau + b[None]; Mp=np.linalg.norm(tp,axis=2).mean(0); Sp=np.linalg.norm(tp.mean(0),axis=1)
    meas.append((Sp/Mp)/R - 1)
meas = np.sqrt(np.mean(np.square(meas),0))
X = 1/R-1
for a,b_ in zip([0,0.02,0.1,0.3,1],[0.02,0.1,0.3,1,10]):
    m=(X>=a)&(X<b_)
    print("X in [%4.2f,%5.2f)  n=%5d   median dR/R  predicted %.4f   measured %.4f   ratio %.2f" % (a,b_,m.sum(),np.median(pred[m]),np.median(meas[m]),np.median(pred[m])/np.median(meas[m])))
print("correlation(log pred, log meas) over all faces: %.3f" % np.corrcoef(np.log(pred+1e-12),np.log(meas+1e-12))[0,1])
