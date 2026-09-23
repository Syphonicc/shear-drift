"""M3 done properly: is a time-stretched beat the same as a true different-heart-rate beat?
Exact Womersley, MCA radius, same Q(t/T) shape, periods 0.765 / 0.900 / 1.035 s.
Compare TRUE tau at period T against the time-stretch of the T=0.9 tau."""
import numpy as np
from scipy.special import jv
MU,RHO=3.5e-3,1060.0; NU=MU/RHO; A=1.5e-3      # MCA radius 1.5 mm
Q=np.array([1.317576,1.317803,1.319834,1.337030,1.434652,1.774284,2.501917,3.553660,
4.726620,5.590000,5.571446,4.685276,3.609585,2.887715,2.596267,2.513271,2.417488,
2.248588,2.073265,1.947601,1.866825,1.807408,1.756493,1.710338,1.668062,1.629294,
1.593740,1.561133,1.531229,1.503805,1.478653,1.455587,1.434433,1.415033,1.397241,
1.380924,1.365960,1.352236,1.339650,1.328107])*1e-6   # the actual inlet table, 40 pts
Qh=np.fft.rfft(Q)/len(Q); NH=12
def tau_of(T,N=40):
    om=2*np.pi/T; al=A*np.sqrt(om/NU); t=np.arange(N)*T/N; tau=np.zeros(N)
    for n in range(NH+1):
        if n==0: kk=4*MU/(np.pi*A**3)
        else:
            L=1j**1.5*al*np.sqrt(n); g=jv(1,L)/jv(0,L)
            kk=-MU*(L/A)*g/(np.pi*A**2*(1-2*g/L))
        c=Qh[n]*(1 if n==0 else 2)
        tau+=np.real(kk*c*np.exp(1j*n*om*t))
    return tau,al
def biom(x):
    m=np.abs(x); M=m.mean(); S=abs(x.sum())/len(x); return M,(1-S/M)/2,1/S
base,al0=tau_of(0.90)
print("baseline T=0.900 s: alpha=%.2f  TAWSS=%.3f Pa  OSI=%.2e  RRT=%.4f /Pa"%((al0,)+biom(base)))
print("\n  T (s)  HR   alpha   |  TRUE vs TIME-STRETCHED beat: rel. difference")
print("                        |   field L2    TAWSS      OSI       RRT")
for T in (0.765,1.035):
    true,al=tau_of(T)
    stretch=base                      # time stretch = same samples, relabelled times
    l2=np.linalg.norm(true-stretch)/np.linalg.norm(true)
    bt,bs=np.array(biom(true)),np.array(biom(stretch))
    print("  %.3f %3.0f   %.2f   |  %7.2f%%  %7.2f%%  %7.2f%%  %7.2f%%"%(T,60/T,al,100*l2,*(100*abs(bs/bt-1))))
print("\n  -> time-stretch augmentation is exact for the biomarkers to within the numbers above;")
print("     it is the WSS *shape* error incurred by ignoring the alpha change over 58-78 bpm.")
