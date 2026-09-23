"""
First-order theory of the window-mismatch ('clock') error, and where p=0.64 comes from.

Detuned sampling tau((1+rho)t) over the window [0,T) covers (1+rho) periods: the extra slice
rho*T*tau(theta) at phase theta (the offset). To first order in rho:
    dM/M = rho (|tau(th)|/<|tau|> - 1),   dS/S = rho (tau_par(th)/<tau_par> - 1)
    dR/R = rho [ tau(th)/<tau> - |tau(th)|/<|tau|> ]                    (1D: tau_par = tau)
RMS over the offset theta gives, with m+ = <tau^+>, m- = <tau^->  (means over the period of the
positive / negative parts), q+ = <(tau^+)^2>, q- = <(tau^-)^2>:
    (dR/R)/rho = 2 sqrt( m-^2 q+ + m+^2 q- ) / ((m+ + m-)(m+ - m-))                        (*)
and 1/R - 1 = 2 m- / (m+ - m-).  Exact in the waveform, no fit.
"""
import numpy as np, sys
sys.path.insert(0, "analysis")
import importlib.util
spec = importlib.util.spec_from_file_location("ws", "analysis/womersley_sweep.py")
src = open("analysis/womersley_sweep.py").read().split("\nrows = []")[0]      # defs only
ns = {}; exec(compile(src, "ws", "exec"), ns)
wall_shear, biomarkers, detune, N, T = ns["wall_shear"], ns["biomarkers"], ns["detune"], ns["N"], ns["T"]

def theory(tau):
    tp, tm = np.clip(tau, 0, None), np.clip(-tau, 0, None)
    mp, mm, qp, qm = tp.mean(), tm.mean(), (tp**2).mean(), (tm**2).mean()
    return 2*np.sqrt(mm**2*qp + mp**2*qm) / ((mp+mm)*(mp-mm)), 2*mm/(mp-mm)

def measured(tau, rho=0.01):
    _, _, _, R0 = biomarkers(tau)
    e = [biomarkers(detune(tau, rho, off))[3]/R0 - 1 for off in np.linspace(0, N, 16, endpoint=False)]
    return np.sqrt(np.mean(np.square(e)))/rho

def eps_per_rho(tau):
    """offset-removed field error per unit rho: rho*(t-T/2)*tau'(t), rms, / rms(tau)"""
    k = np.fft.fftfreq(N, 1/N); dtau = np.real(np.fft.ifft(np.fft.fft(tau)*2j*np.pi*k/T))
    t = np.arange(N)*T/N
    return np.sqrt(np.mean(((t-T/2)*dtau)**2))/np.sqrt(np.mean(tau**2))

print("check (*) against the detune experiment (rho=1%), Womersley waveforms:")
rows=[]
for alpha in (4,8,12,16):
    for amp in np.linspace(0.3,3.0,10):
        tau,_ = wall_shear(amp, alpha); th,X = theory(tau); me = measured(tau)
        if 0.04 < X < 3.2: rows.append((alpha,amp,X,th,me,eps_per_rho(tau)))
rows=np.array(rows)
print("  max |theory/measured - 1| = %.3f  (n=%d)" % (np.abs(rows[:,3]/rows[:,4]-1).max(), len(rows)))
def fit(x,y): P=np.polyfit(np.log(x),np.log(y),1); return P[0],np.exp(P[1])
print("  fit of theory (*) vs X:            p=%.3f c=%.3f" % fit(rows[:,2],rows[:,3]))
print("  fit of (*)/(eps/rho) vs X:         p=%.3f c=%.3f   <- the published 0.64" % fit(rows[:,2],rows[:,3]/rows[:,5]))
print("  eps/rho vs X:                      p=%.3f  (eps grows with reversal; this is what pulls 0.78 -> 0.64)" % fit(rows[:,2],rows[:,5])[0])

print("\nsingle harmonic tau = 1 + b cos(theta): exact limits of (*)")
th = np.linspace(0,2*np.pi,20001)[:-1]
for b in (1.001,1.01,1.05,1.2,1.5,2,3,5,10,30,100):
    tau = 1 + b*np.cos(th); F,X = theory(tau); print("  b=%7.3f  X=%.4f  F=%.4f  F/X=%.3f  F/X^(5/6)=%.3f" % (b,X,F,F/X,F/X**(5/6)))
print("  near onset (b->1+): reversed lobe is parabolic, width ~ sqrt(delta), depth ~ delta:")
print("     m- ~ delta^(3/2) -> X ~ delta^(3/2);  q- ~ delta^(5/2) -> F ~ delta^(5/4) = X^(5/6)")
print("  broad reversal (b->inf):  F -> b/sqrt2 ~ (pi/2) X / sqrt2  ->  p = 1")
