"""Checks demanded by self-review round 1: M2 band sensitivity, M3 alpha shift,
M4 validity envelope, M9 noise-vs-N."""
import numpy as np
d=np.load("../results_v2/wss_data.npz"); k=[x for x in d.files if d[x].ndim==3][0]
tau0=d[k][80:120].astype(np.float64)*1060.0
def biom(t):
    M=np.linalg.norm(t,axis=2).mean(0); S=np.linalg.norm(t.mean(0),axis=1); return M,S/M,1/np.maximum(S,1e-30)
def rl2(a,b): return 100*np.linalg.norm(a-b)/np.linalg.norm(b)

print("### M3  heart-rate augmentation: alpha shift and waveform validity")
T0=0.9; NU=3.302e-6; a_radius=1.5e-3   # MCA radius ~1.5 mm
for Ns in (34,40,46):
    T=T0*Ns/40; al=a_radius*np.sqrt(2*np.pi/T/NU)
    print("   %2d samples/beat -> T=%.3f s, HR=%3.0f bpm, alpha=%.2f (%+.0f%% vs baseline)"%(Ns,T,60/T,al,100*(al/ (a_radius*np.sqrt(2*np.pi/T0/NU))-1)))
print("   Womersley WSS phase lag ~ arctan-type; at alpha<2 flow is quasi-steady,")
print("   |tau| tracks Q with lag < %.1f%% of the cycle -> time-stretch is a controlled approximation."%(100*0.5*(a_radius*np.sqrt(2*np.pi/T0/NU))**2/8/np.pi))

print("\n### M9  white noise: does the RRT error scale as 1/sqrt(N)?")
rng=np.random.default_rng(0)
for up in (1,2,4):
    t=np.repeat(tau0,up,axis=0) if up>1 else tau0   # same waveform, N x up snapshots
    N=t.shape[0]; rms=np.sqrt((t**2).sum(2).mean(0))
    M0,R0,RT0=biom(t)
    e=[]
    for s in range(4):
        n=np.random.default_rng(s).standard_normal(t.shape)/np.sqrt(3)
        M,R,RT=biom(t+0.108*rms[None,:,None]*n); e.append(rl2(RT,RT0))
    print("   N=%3d  RRT rL2 = %.3f %%   (x sqrt(N/40) = %.3f)"%(N,np.mean(e),np.mean(e)*np.sqrt(N/40)))

print("\n### M4  validity envelope of the first-order bias formula")
mag=np.linalg.norm(tau0,axis=2); M=mag.mean(0); Sv=tau0.mean(0); S=np.linalg.norm(Sv,axis=1); R=S/M
that=tau0/np.maximum(mag,1e-30)[:,:,None]; mt=that.mean(0)
v=Sv/np.maximum(S,1e-30)[:,None]**2 - mt/M[:,None]
rms=np.sqrt((tau0**2).sum(2).mean(0)); X=1/R-1
bins=[(0.02,0.1),(0.1,0.3),(0.3,1),(1,10)]
print("   ratio predicted/measured, by eps and cancellation bin:")
print("   %6s"%"eps"+"".join("%16s"%f"X[{a},{b})" for a,b in bins))
for eps in (0.02,0.05,0.108,0.20):
    pred=eps*rms*np.linalg.norm(v,axis=1)/np.sqrt(3); meas=[]
    for s in range(4):
        b=eps*rms[:,None]*np.random.default_rng(s).standard_normal((tau0.shape[1],3))/np.sqrt(3)
        tp=tau0+b[None]; Mp=np.linalg.norm(tp,axis=2).mean(0); Sp=np.linalg.norm(tp.mean(0),axis=1)
        meas.append((Sp/Mp)/R-1)
    meas=np.sqrt(np.mean(np.square(meas),0))
    row=[np.median(pred[(X>=a)&(X<b_)])/np.median(meas[(X>=a)&(X<b_)]) for a,b_ in bins]
    print("   %5.1f%%"%(100*eps)+"".join("%16.2f"%r for r in row))

print("\n### M2  are the fingerprint bands robust to the OSI distribution?")
print("   reweight faces to shift the wall's OSI distribution, recompute OSI/TAWSS rL2 per error type")
def triple(tp,sel):
    M,R,RT=biom(tau0[:,sel]); Mp,Rp,RTp=biom(tp[:,sel])
    O=(1-R)/2; Op=(1-Rp)/2
    return rl2(Mp,M), rl2(Op,O), rl2(RTp,RT)
rmsf=np.sqrt((tau0**2).sum(2).mean(0))
cases={}
cases["drift"]=None
g=np.random.default_rng(1)
cases["white"]=tau0+0.108*rmsf[None,:,None]*g.standard_normal(tau0.shape)/np.sqrt(3)
cases["bias"]=tau0+0.108*rmsf[:,None][None]*g.standard_normal((1,)+tau0.shape[1:])/np.sqrt(3)
kk=np.fft.fftfreq(40)*40
cases["drift"]=np.real(np.einsum('tk,kfc->tfc',np.exp(2j*np.pi*np.outer(np.arange(40)*1.05,kk)/40),np.fft.fft(tau0,axis=0)))/40
print("   %-22s %10s %10s %10s"%("face subset (median OSI)","drift","white","bias"))
for name,sel in [("whole wall",np.ones(len(X),bool)),
                 ("OSI>0.002",  (1-1/(1+X))/2>0.002),
                 ("OSI>0.01",   (1-1/(1+X))/2>0.01),
                 ("OSI>0.03",   (1-1/(1+X))/2>0.03)]:
    O=(1-1/(1+X))/2
    out=[]
    for c in ("drift","white","bias"):
        t_,o_,r_=triple(cases[c],sel); out.append(o_/t_)
    print("   %-22s %10.2f %10.2f %10.2f   (n=%d, med OSI %.4f)"%(name,*out,sel.sum(),np.median(O[sel])))
