"""Round-2 checks.
R2a: does F/X^(5/6) actually converge to a constant as the reversed lobe vanishes?
R2b: tab:etype compares |dRRT/RRT|-|dTAWSS/TAWSS| (a difference of magnitudes) against
     a prediction for |dR/R| (a magnitude). Recompute the table in terms of |dR/R|,
     which is what the theory predicts, and test both closed forms directly."""
import numpy as np
print("### R2a  single harmonic tau = 1 + b cos(theta), b -> 1+ : does F/X^(5/6) settle?")
th=np.linspace(0,2*np.pi,2000001)[:-1]
def F_and_X(tau):
    tp,tm=np.clip(tau,0,None),np.clip(-tau,0,None)
    mp,mm,qp,qm=tp.mean(),tm.mean(),(tp**2).mean(),(tm**2).mean()
    return 2*np.sqrt(mm**2*qp+mp**2*qm)/((mp+mm)*(mp-mm)), 2*mm/(mp-mm)
for b in (1.3,1.1,1.03,1.01,1.003,1.001,1.0003,1.0001):
    F,X=F_and_X(1+b*np.cos(th))
    print("   b=%9.4f  X=%10.3e  F=%10.3e  F/X^(5/6)=%.4f  F/X=%9.2f"%(b,X,F,F/X**(5/6),F/X))

print("\n### R2b  cerebral wall: median |dR/R| by bin, measured vs the two closed forms")
d=np.load("../results_v2/wss_data.npz"); k=[x for x in d.files if d[x].ndim==3][0]
tau=d[k][80:120].astype(np.float64)*1060.0; N=40
mag=np.linalg.norm(tau,axis=2); M=mag.mean(0); Sv=tau.mean(0); S=np.linalg.norm(Sv,axis=1); R=S/M
X=np.clip(1/R-1,1e-12,None)
kk=np.fft.fftfreq(N)*N
def detune(rho): return np.real(np.einsum('tk,kfc->tfc',np.exp(2j*np.pi*np.outer(np.arange(N)*(1+rho),kk)/N),np.fft.fft(tau,axis=0)))/N
rms=np.sqrt((tau**2).sum(2).mean(0)); rng=np.random.default_rng(0)
def dRR(tp):
    Mp=np.linalg.norm(tp,axis=2).mean(0); Sp=np.linalg.norm(tp.mean(0),axis=1); return np.abs((Sp/Mp)/R-1)
rho=0.05; eps_win=np.linalg.norm(detune(rho)-tau)/np.linalg.norm(tau)
meas={}
meas["window mismatch rho=5%"]=dRR(detune(rho))
meas["frozen bias"]=np.sqrt(np.mean([dRR(tau+0.108*rms[:,None][None]*np.random.default_rng(s).standard_normal((1,)+tau.shape[1:])/np.sqrt(3))**2 for s in range(6)],0))
meas["white noise"]=np.sqrt(np.mean([dRR(tau+0.108*rms[None,:,None]*np.random.default_rng(s).standard_normal(tau.shape)/np.sqrt(3))**2 for s in range(6)],0))
# closed forms, per face
tp_,tm_=np.clip(tau,0,None),np.clip(-tau,0,None)   # 3D: use the component along s_hat
shat=Sv/np.maximum(S,1e-30)[:,None]; tpar=np.einsum('tfc,fc->tf',tau,shat)
pp,pm=np.clip(tpar,0,None),np.clip(-tpar,0,None)
mp,mm,qp,qm=pp.mean(0),pm.mean(0),(pp**2).mean(0),(pm**2).mean(0)
F=2*np.sqrt(mm**2*qp+mp**2*qm)/np.maximum((mp+mm)*(mp-mm),1e-30)
pred_win=rho*F
that=tau/np.maximum(mag,1e-30)[:,:,None]; mt=that.mean(0)
v=shat/np.maximum(S,1e-30)[:,None]-mt/M[:,None]
pred_bias=0.108*rms*np.linalg.norm(v,axis=1)/np.sqrt(3)
bins=[(0,0.02),(0.02,0.1),(0.1,0.3),(0.3,1),(1,10)]
hdr="%-26s"%"quantity"+"".join("%13s"%f"[{a},{b})" for a,b in bins)
print("   eps(window)=%.1f%%, eps(bias)=eps(noise)=10.8%%"%(100*eps_win)); print("  ",hdr)
print("   %-26s"%"faces"+"".join("%13d"%((X>=a)&(X<b)).sum() for a,b in bins))
for nm,arr in list(meas.items()):
    print("   %-26s"%("measured "+nm)+"".join("%13.3f"%(100*np.median(arr[(X>=a)&(X<b)])) for a,b in bins))
    if "window" in nm: print("   %-26s"%"  predicted (eq.window)"+"".join("%13.3f"%(100*np.median(pred_win[(X>=a)&(X<b)])) for a,b in bins))
    if "bias" in nm:   print("   %-26s"%"  predicted (eq.bias)"+"".join("%13.3f"%(100*np.median(pred_bias[(X>=a)&(X<b)])) for a,b in bins))
