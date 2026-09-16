#!/usr/bin/env python3
"""
womersley_sweep.py -- test the 1/R amplification law on exact analytical WSS.

Womersley flow in a rigid straight tube: for every flow harmonic Q_n the wall
shear stress is known exactly, so tau_w(t) has no mesh/solver error.
We sweep
  * reversal amplitude  -> R = |int tau dt| / int |tau| dt  from ~1 to ~0
  * Womersley number    -> alpha = 2 (cerebral) ... 16 (aorta)
and apply the same Fourier rate-detune used in mechanism_test.py.

Finding: excess R-error scales with the cancellation variable (1-R)/R = 1/R - 1,
          (dR/R)/eps ~ ((1-R)/R)^p ; RRT error ~ TAWSS error + dR/R.
          A pure 1/R power law does NOT fit (not a power law near R=1).
Control:                symmetric smoothing leaves RRT error = 0 exactly.

Outputs: womersley_sweep.csv, womersley_sweep.png
"""
import numpy as np
from scipy.special import jv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU, RHO, T = 3.5e-3, 1060.0, 0.9          # Pa s, kg/m^3, s
NU, OMEGA = MU / RHO, 2 * np.pi / T
N, NH = 40, 12                             # snapshots per cycle (as CFD), harmonics kept
RHOS = (0.01, 0.02, 0.05)                  # detuning levels
ALPHAS = (2, 4, 8, 12, 16)
AMPS = np.linspace(0.3, 3.0, 28)           # pulsatile amplitude / mean flow

def base_shape(t):
    """Zero-mean, unit-peak waveform: systolic peak, dicrotic shoulder, diastolic decay."""
    ph = t / T
    s = (np.exp(-((ph - 0.24) / 0.07) ** 2)
         + 0.25 * np.exp(-((ph - 0.40) / 0.05) ** 2)
         - 0.30 * np.exp(-((ph - 0.33) / 0.03) ** 2)     # early-diastolic backflow notch
         + 0.10 * np.exp(-3 * ph))
    return s - s.mean()

def wss_per_flow(n, alpha):
    """Exact complex tau_w / Q for harmonic n (n=0 -> Poiseuille)."""
    a = alpha / np.sqrt(OMEGA / NU)        # tube radius giving this alpha
    if n == 0:
        return 4 * MU / (np.pi * a ** 3), a
    L = 1j ** 1.5 * alpha * np.sqrt(n)
    g = jv(1, L) / jv(0, L)
    return -MU * (L / a) * g / (np.pi * a ** 2 * (1 - 2 * g / L)), a

def wall_shear(amp, alpha):
    tf = np.arange(512) * T / 512
    Q = 1.0 + amp * base_shape(tf) / np.abs(base_shape(tf)).max()
    Qh = np.fft.rfft(Q) / 512
    t = np.arange(N) * T / N
    tau = np.zeros(N)
    for n in range(NH + 1):
        k, _ = wss_per_flow(n, alpha)
        c = Qh[n] * (1 if n == 0 else 2)
        tau += np.real(k * c * np.exp(1j * n * OMEGA * t))
    qmin = Q.min()
    return tau, qmin

def biomarkers(tau):
    m = np.abs(tau)
    ta = m.mean()
    R = abs(tau.sum()) / m.sum()
    return ta, (1 - R) / 2, 1 / (R * ta), R

def detune(tau, rho, off=0.0):
    """Surrogate clock runs (1+rho) fast, starting at phase offset `off` (in samples)."""
    k = np.fft.fftfreq(N, 1 / N)
    j = np.arange(N)
    return np.real(np.exp(2j * np.pi * np.outer(off + j * (1 + rho), k) / N) @ np.fft.fft(tau)) / N

def shift(tau, off):
    k = np.fft.fftfreq(N, 1 / N)
    return np.real(np.fft.ifft(np.fft.fft(tau) * np.exp(2j * np.pi * k * off / N)))

def smooth(tau, w=3):
    ker = np.ones(w) / w
    return np.real(np.fft.ifft(np.fft.fft(tau) * np.fft.fft(np.roll(np.pad(ker, (0, N - w)), -(w // 2)))))

rows = []
OFFS = np.linspace(0, N, 40, endpoint=False)   # average over where in the cycle the rollout starts
for alpha in ALPHAS:
    for amp in AMPS:
        tau, qmin = wall_shear(amp, alpha)
        ta0, o0, r0, R0 = biomarkers(tau)
        if R0 < 0.03:            # RRT itself diverges; not meaningful
            continue
        _, _, rs, _ = biomarkers(smooth(tau))
        for rho in RHOS:
            eps, eT, eR, eRRT, eO = [], [], [], [], []
            for off in OFFS:
                ref = shift(tau, off)                  # truth, same start phase
                d = detune(tau, rho, off)
                ta, o, r, R = biomarkers(d)
                eps.append(np.linalg.norm(d - ref) / np.linalg.norm(ref))
                eT.append(ta / ta0 - 1); eR.append(R / R0 - 1); eRRT.append(r / r0 - 1)
                eO.append(o / o0 - 1 if o0 > 1e-6 else np.nan)
            rms = lambda v: np.sqrt(np.nanmean(np.square(v)))
            rows.append((alpha, amp, qmin, R0, o0, rho, rms(eps), rms(eT), rms(eR), rms(eO), rms(eRRT),
                         abs(rs / r0 - 1)))

A = np.array(rows)
hdr = "alpha,amp,Qmin_over_Qmean,R,OSI,rho,field_eps,err_TAWSS,err_R,err_OSI,err_RRT,smooth_err_RRT"
np.savetxt("womersley_sweep.csv", A, delimiter=",", header=hdr, comments="", fmt="%.6g")

# ---- summary (rms over 40 start phases) ----
print(f"{'alpha':>5} {'Qmin/Q':>7} {'R':>6} {'OSI':>6} | rho=0.05 rms: {'eps':>6} {'TAWSS':>7} {'dR/R':>7} {'OSI':>7} {'RRT':>7} | (dR/R)/eps  1/R")
sel = A[A[:, 5] == 0.05]
for alpha in ALPHAS:
    s_ = sel[sel[:, 0] == alpha]
    for i in np.linspace(0, len(s_) - 1, min(5, len(s_))).astype(int):
        a = s_[i]
        print(f"{a[0]:5.0f} {a[2]:7.2f} {a[3]:6.3f} {a[4]:6.3f} |    {a[6]*100:9.2f}% {a[7]*100:6.2f}% {a[8]*100:6.2f}% "
              f"{a[9]*100:6.2f}% {a[10]*100:6.2f}% | {a[8]/a[6]:9.3f} {1/a[3]:5.2f}")

osc = A[:, 3] < 0.95                         # need real oscillation for R to move at all
Xc = lambda R: (1 - R) / R                   # cancellation variable = 1/R - 1
x = np.log(Xc(A[osc, 3])); y = np.log(A[osc, 8] / A[osc, 6])
slope, icpt = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
print(f"\n(dR/R)/eps ~ ((1-R)/R)^{slope:.2f}   R^2={r2:.3f}  n={osc.sum()}   (first-order argument gives exponent 1)")
xb = np.log(1 / A[osc, 3]); pb = np.polyfit(xb, y, 1)
print(f"   for comparison, pure 1/R fit: exponent {pb[0]:.2f}, R^2={np.corrcoef(xb, y)[0,1]**2:.3f}")
for al in ALPHAS:
    m = osc & (A[:, 0] == al)
    if m.sum() > 3:
        sl = np.polyfit(np.log(Xc(A[m, 3])), np.log(A[m, 8] / A[m, 6]), 1)[0]
        print(f"   alpha={al:2d}: exponent {sl:5.2f}  (R from {A[m,3].min():.2f} to {A[m,3].max():.3f})")
print(f"smoothing control: max RRT error = {A[:,11].max():.1e}  (exactly 0 expected)")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for rho, mk in zip(RHOS, "o^s"):
    m = osc & (A[:, 5] == rho)
    sc = ax[0].scatter(Xc(A[m, 3]), A[m, 8] / A[m, 6], c=A[m, 0], cmap="viridis", marker=mk, s=22, label=f"ρ={rho}")
xx = np.logspace(np.log10(Xc(A[osc, 3]).min()), np.log10(Xc(A[osc, 3]).max()), 50)
ax[0].plot(xx, np.exp(icpt) * xx ** slope, "r-", lw=1, label=f"fit ∝ ((1-R)/R)^{slope:.2f}")
ax[0].set(xscale="log", yscale="log", xlabel="(1−R)/R  =  1/R − 1", ylabel="(δR/R) / field ε",
          title="cancellation amplification, exact Womersley WSS")
ax[0].legend(fontsize=8); fig.colorbar(sc, ax=ax[0], label="Womersley α")
s_ = A[A[:, 5] == 0.05]
for lab, col, c in (("TAWSS", 7, "C0"), ("OSI", 9, "C1"), ("RRT", 10, "C3")):
    ax[1].scatter(1 / s_[:, 3], s_[:, col] * 100, s=14, c=c, label=lab)
ax[1].set(xscale="log", yscale="log", xlabel="1/R", ylabel="rms error at ρ=0.05 (%)",
          title="biomarker error vs cancellation")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("womersley_sweep.png", dpi=150)
print("wrote womersley_sweep.csv and womersley_sweep.png")
