#!/usr/bin/env python3
"""
phase_correct.py -- re-time a drifting surrogate rollout using only its own period.

Idea (phase reduction, Taira & Nakao 2018; Yawata et al. 2024): an autonomous
rollout is a limit-cycle oscillator. If its orbit is right but its frequency is
wrong, the error lives in the phase direction and a clock correction removes it.

No ground-truth WSS is used for the correction:
  * rollout self-period  <- lag growth of the rollout against its OWN first cycle
  * true period          <- the heart rate the user asked for (N samples per beat)
Ground truth is used only afterwards, to score raw vs corrected.

Also reports the offset-removed field error eps_aligned, the right epsilon for the
cancellation law (a pure time shift leaves cycle-averaged biomarkers unchanged).

Run from hackathon/ (any env with numpy/scipy/matplotlib):
    python analysis/phase_correct.py                                 # uses *_hr results
    python analysis/phase_correct.py --tag _hr_memorize              # other runs
"""
import argparse, json, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("WSS", "../results_v2/wss_data.npz"))
ap.add_argument("--tag", default="_hr")
ap.add_argument("--skip", type=int, default=1, help="cycles to discard as rollout transient when timing")
ap.add_argument("--law", default="data/meta.json")
args = ap.parse_args()

RHO_BLOOD, N = 1060.0, 40
SAC_C, SAC_R = np.array([5.40, -4.00, 0.78]), 2.0
summ = json.load(open(f"data/pod_lstm_summary{args.tag}.json"))
res = np.load(f"data/pod_lstm_results{args.tag}.npz")
r = summ["r"]

d = np.load(args.data)
w3 = d["wss"][80:120].astype(np.float64) * RHO_BLOOD
A, cen = d["areas"], d["centres"] * 1e3
F = w3.shape[1]
Xs = w3.reshape(N, -1)
mean = Xs.mean(0)
_, _, Vt = np.linalg.svd(Xs - mean, full_matrices=False)
modes = Vt[:r]
coef = (Xs - mean) @ modes.T
assert np.allclose(coef, res["truth"], atol=1e-3 * np.abs(coef).max()), \
    "POD basis differs from the training run (check --data / r)"
decode = lambda a: (mean + a @ modes).reshape(len(a), F, 3)
roll = res["roll"].astype(np.float64)
T = len(roll); K = T // N

def biomarkers(x):
    mag = np.linalg.norm(x, axis=2); ta = mag.mean(0)
    R = np.clip(np.linalg.norm(x.sum(0), axis=1) / np.maximum(mag.sum(0), 1e-30), 1e-3, 1.0)
    return ta, (1 - R) / 2, 1 / (R * ta), R

rel = lambda a, b: np.divide(a - b, b, out=np.zeros_like(b), where=np.abs(b) > 1e-6)
aw = lambda v, s: float(np.sum(np.abs(v[s]) * A[s]) / np.sum(A[s])) if s.any() else 0.0
ta0, osi0, rrt0, R0 = biomarkers(w3)
X0 = 1 / R0 - 1
REG = {"wall": np.ones(F, bool), "sac": np.linalg.norm(cen - SAC_C, axis=1) < SAC_R, "osi>0.1": osi0 > 0.1}

def frac_lag(p, q):
    """L such that p(t) ~ q(t - L); FFT cross-correlation + parabolic refinement."""
    P, Q = np.fft.fft(p - p.mean(0), axis=0), np.fft.fft(q - q.mean(0), axis=0)
    cc = np.real(np.fft.ifft((P * np.conj(Q)).sum(1)))
    k = int(np.argmax(cc)); a, b, c = cc[k - 1], cc[k], cc[(k + 1) % len(cc)]
    den = a - 2 * b + c
    L = k + (0.5 * (a - c) / den if den != 0 else 0.0)
    return L - len(cc) if L > len(cc) / 2 else L

def self_rate(x):
    """clock error from the rollout alone: lag growth of each cycle vs the first kept cycle."""
    ref = x[args.skip * N:(args.skip + 1) * N]
    ks = np.arange(args.skip, K)
    lg = np.array([frac_lag(x[k * N:(k + 1) * N], ref) for k in ks])
    return -np.polyfit(ks * N, lg, 1)[0], lg

def retime(x, rho):
    """rollout phase at step t is t(1+rho); sample it at t/(1+rho) to restore the true clock."""
    t = np.arange(len(x))
    cs = CubicSpline(t, x, axis=0)
    tq = t / (1 + rho)
    return cs(tq), tq[-1] < t[-1]

def shift_truth(L):
    """ground-truth beat delayed by L samples (periodic, Fourier)."""
    k = np.fft.fftfreq(N, 1 / N)[:, None, None]
    return np.real(np.fft.ifft(np.fft.fft(w3, axis=0) * np.exp(-2j * np.pi * k * L / N), axis=0))

rho_self, lg_self = self_rate(roll)
corr, _ = retime(roll, rho_self)
rho_after, _ = self_rate(corr)
truth_lags = np.array([frac_lag(roll[k * N:(k + 1) * N], coef) for k in range(K)])
print(f"clock error from the rollout alone:  rho_self = {rho_self*100:+.2f}%   "
      f"(ground-truth lag growth said {summ['rho_lag']*100:+.2f}%)")
print(f"after re-timing, rollout self-rate  = {rho_after*100:+.3f}%   (target 0)")

law = json.load(open(args.law))["law"] if os.path.exists(args.law) else {"c": 0.194, "p": 0.641}
out, lawrows = [], []
print(f"\n{'cyc':>3} | {'L2 raw/cor':>13} {'L2 aligned raw/cor':>19} | "
      f"{'wall RRT raw/cor':>17} | {'sac RRT raw/cor':>16} | {'OSI>0.1 RRT raw/cor':>20}")
for k in range(K):
    row = {"cycle": k + 1}
    for name, x in (("raw", roll), ("cor", corr)):
        wp = decode(x[k * N:(k + 1) * N])
        tp, op, rp, _ = biomarkers(wp)
        L = frac_lag(x[k * N:(k + 1) * N], coef)
        nrm = np.linalg.norm(w3)
        row[name] = dict(
            l2=float(np.linalg.norm(wp - w3) / nrm),
            l2_aligned=float(np.linalg.norm(wp - shift_truth(L)) / nrm),
            lag=float(L),
            **{n: dict(tawss=aw(rel(tp, ta0), s), osi=aw(rel(op, osi0), s), rrt=aw(rel(rp, rrt0), s))
               for n, s in REG.items()})
        if name == "raw" and k == K - 1:
            eps = row["raw"]["l2_aligned"]
            pred = law["c"] * eps * X0 ** law["p"]
            meas = np.abs(rel(rp, rrt0)) - np.abs(rel(tp, ta0))
            for lo, up in ((0, .02), (.02, .1), (.1, .3), (.3, 1), (1, 10)):
                s = (X0 >= lo) & (X0 < up)
                if s.sum() >= 5:
                    lawrows.append((lo, up, int(s.sum()), float(np.median(pred[s])), float(np.median(meas[s]))))
    out.append(row)
    g = lambda n, reg: f"{row['raw'][reg]['rrt']*100:6.2f}/{row['cor'][reg]['rrt']*100:5.2f}%"
    print(f"{k+1:3d} | {row['raw']['l2']*100:5.1f}/{row['cor']['l2']*100:5.1f}% "
          f"{row['raw']['l2_aligned']*100:9.1f}/{row['cor']['l2_aligned']*100:5.1f}%  | "
          f"{g(0,'wall'):>17} | {g(0,'sac'):>16} | {g(0,'osi>0.1'):>20}")

print(f"\nlaw check, last raw cycle, with offset-removed eps = {out[-1]['raw']['l2_aligned']*100:.1f}%:")
print(f"  {'1/R-1 bin':>14} {'faces':>6} {'predicted':>10} {'measured':>10}")
for lo, up, n, p, m in lawrows:
    print(f"  [{lo:5.2f},{up:5.2f}) {n:6d} {p*100:9.2f}% {m*100:9.2f}%")

mean_over = lambda key, reg: np.mean([o[key][reg]["rrt"] for o in out[args.skip:]])
print("\nmean RRT error over rollout cycles after the transient:")
for reg in REG:
    print(f"  {reg:8s} raw {mean_over('raw', reg)*100:5.2f}%  ->  corrected {mean_over('cor', reg)*100:5.2f}%")

json.dump(dict(tag=args.tag, rho_self=float(rho_self), rho_after=float(rho_after),
               rho_truth=summ["rho_lag"], cycles=out,
               law_check=[dict(X=[lo, up], n=n, pred=p, meas=m) for lo, up, n, p, m in lawrows]),
          open(f"data/phase_correct{args.tag}.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
tt = np.arange(T) / N
ax[0].plot(tt, np.concatenate([coef] * K)[:, 0], c="grey", lw=2, alpha=.4, label="truth")
ax[0].plot(tt, roll[:, 0], c="C3", lw=1, label="rollout")
ax[0].plot(tt, corr[:, 0], c="C0", lw=1, ls="--", label="re-timed")
ax[0].set(xlabel="cycles", ylabel="POD mode 1", title="clock correction"); ax[0].legend(fontsize=8)
cyc = np.arange(1, K + 1)
for key, c in (("raw", "C3"), ("cor", "C0")):
    ax[1].plot(cyc, [o[key]["lag"] for o in out], c=c, marker="o", label=key)
ax[1].axhline(0, c="k", lw=.6); ax[1].set(xlabel="rollout cycle", ylabel="lag vs truth (samples)", title="phase lag")
ax[1].legend(fontsize=8)
for reg, m in (("wall", "o"), ("osi>0.1", "s")):
    for key, c in (("raw", "C3"), ("cor", "C0")):
        ax[2].plot(cyc, [o[key][reg]["rrt"] * 100 for o in out], c=c, marker=m, label=f"{key} {reg}")
ax[2].set(xlabel="rollout cycle", ylabel="RRT error (%)", title="RRT error, raw vs re-timed")
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"analysis/phase_correct{args.tag}.png", dpi=140)
print(f"wrote data/phase_correct{args.tag}.json, analysis/phase_correct{args.tag}.png")
