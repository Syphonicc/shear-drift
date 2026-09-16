#!/usr/bin/env python3
"""
pod_lstm.py -- train a POD+LSTM wall-shear surrogate, roll it out on its own,
and run the reliability check on its *real* (not injected) drift.

Pipeline
  1. POD of the wall WSS over one converged cycle (cycle 3)  -> r coefficients/step
  2. LSTM learns coefficient(t) -> coefficient(t+1) from a window of W steps
  3. Autonomous rollout for K cycles from a W-step seed (no ground truth fed back)
  4. For every rollout cycle: field error, TAWSS/OSI/RRT errors by region,
     compressor floor (POD reconstruction), drift rate estimate, rho diagnostic,
     and the cancellation law's per-face prediction vs what actually happened.

Run from hackathon/ inside the `drift` env:
    python analysis/pod_lstm.py                   # defaults
    python analysis/pod_lstm.py --r 8 --seed 1    # variants
Outputs: data/pod_lstm_results.npz, analysis/pod_lstm.png
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("WSS", "../results_v2/wss_data.npz"))
ap.add_argument("--r", type=int, default=10, help="POD modes")
ap.add_argument("--window", type=int, default=8)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--noise", type=float, default=0.02, help="training input noise (std, normalised units)")
ap.add_argument("--cycles", type=int, default=6, help="rollout length in cycles")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--law", default="data/meta.json")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_num_threads(max(1, os.cpu_count() // 2))

RHO_BLOOD, N = 1060.0, 40
SAC_C, SAC_R = np.array([5.40, -4.00, 0.78]), 2.0

# ---------------- data ----------------
d = np.load(args.data)
w3 = d["wss"][80:120].astype(np.float64) * RHO_BLOOD          # cycle 3, Pa, (N,F,3)
A, cen = d["areas"], d["centres"] * 1e3
F = w3.shape[1]
sac = np.linalg.norm(cen - SAC_C, axis=1) < SAC_R

def biomarkers(x):
    mag = np.linalg.norm(x, axis=2)
    ta = mag.mean(0)
    R = np.clip(np.linalg.norm(x.sum(0), axis=1) / np.maximum(mag.sum(0), 1e-30), 1e-3, 1.0)
    return ta, (1 - R) / 2, 1 / (R * ta), R

def rel(a, b):
    return np.divide(a - b, b, out=np.zeros_like(b), where=np.abs(b) > 1e-6)

def aw(v, s):
    return float(np.sum(np.abs(v[s]) * A[s]) / np.sum(A[s])) if s.any() else 0.0

ta0, osi0, rrt0, R0 = biomarkers(w3)
X0 = 1 / R0 - 1
hi = osi0 > 0.1
REG = {"wall": np.ones(F, bool), "sac": sac, "osi>0.1": hi}

# ---------------- 1. POD ----------------
Xs = w3.reshape(N, -1)
mean = Xs.mean(0)
U, S, Vt = np.linalg.svd(Xs - mean, full_matrices=False)
r = args.r
modes = Vt[:r]                                   # (r, 3F)
coef = (Xs - mean) @ modes.T                     # (N, r)
energy = (S[:r] ** 2).sum() / (S ** 2).sum()
decode = lambda a: (mean + a @ modes).reshape(len(a), F, 3)
print(f"POD: {F} faces, r={r} modes capture {energy*100:.3f}% of fluctuation energy")

# compressor floor: best possible biomarkers with r modes
wr = decode(coef)
tfl, ofl, rfl, _ = biomarkers(wr)
floor = {k: dict(tawss=aw(rel(tfl, ta0), s), osi=aw(rel(ofl, osi0), s), rrt=aw(rel(rfl, rrt0), s))
         for k, s in REG.items()}
floor_l2 = float(np.linalg.norm(wr - w3) / np.linalg.norm(w3))
print(f"compressor floor: field L2 {floor_l2*100:.3f}%  wall RRT {floor['wall']['rrt']*100:.3f}%")

# ---------------- 2. LSTM ----------------
mu, sd = coef.mean(0), coef.std(0) + 1e-12
z = (coef - mu) / sd
W = args.window
seq = np.concatenate([z] * 8)                     # the converged cycle repeats exactly
Xw = np.stack([seq[i:i + W] for i in range(len(seq) - W)])
Yw = seq[W:]
Xw, Yw = torch.tensor(Xw, dtype=torch.float32), torch.tensor(Yw, dtype=torch.float32)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.lstm = nn.LSTM(r, args.hidden, batch_first=True)
        s.out = nn.Linear(args.hidden, r)
    def forward(s, x):
        h, _ = s.lstm(x)
        return x[:, -1] + s.out(h[:, -1])         # predict the increment

net = Net()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
t0 = time.time()
for ep in range(args.epochs):
    xin = Xw + args.noise * torch.randn_like(Xw)
    loss = nn.functional.mse_loss(net(xin), Yw)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if ep % max(1, args.epochs // 6) == 0 or ep == args.epochs - 1:
        print(f"  epoch {ep:5d}  one-step MSE {loss.item():.2e}")
print(f"trained in {time.time()-t0:.0f} s")

# ---------------- 3. autonomous rollout ----------------
K = args.cycles
T = K * N
net.eval()
roll = np.zeros((T, r))
roll[:W] = z[:W]
with torch.no_grad():
    for t in range(W, T):
        x = torch.tensor(roll[t - W:t][None], dtype=torch.float32)
        roll[t] = net(x)[0].numpy()
truth_z = np.concatenate([z] * K)
one_step = float(np.sqrt(np.mean((net(Xw[:N]).detach().numpy() - Yw[:N].numpy()) ** 2)))
roll_c = roll * sd + mu

# drift rate: (a) sub-sample lag growth per cycle, (b) Hilbert phase slope of mode 1
def frac_lag(p, q):
    """lag (in samples) that best maps q onto p, via FFT cross-correlation + parabola."""
    P, Q = np.fft.fft(p - p.mean(0), axis=0), np.fft.fft(q - q.mean(0), axis=0)
    cc = np.real(np.fft.ifft((P * np.conj(Q)).sum(1)))
    k = int(np.argmax(cc)); a, b, c = cc[k - 1], cc[k], cc[(k + 1) % len(cc)]
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    lag = k + off
    return lag - len(cc) if lag > len(cc) / 2 else lag

lags = np.array([frac_lag(roll[k * N:(k + 1) * N], z) for k in range(K)])
rho_lag = -np.polyfit(np.arange(K) * N + N / 2, lags, 1)[0] if K > 1 else 0.0
ph = lambda s: np.unwrap(np.angle(hilbert(s - s.mean())))
tt = np.arange(W, T)
rho_hil = np.polyfit(tt, ph(roll[:, 0])[W:], 1)[0] / np.polyfit(tt, ph(truth_z[:, 0])[W:], 1)[0] - 1
amp = np.std(roll[-N:], 0).mean() / np.std(z, 0).mean()
print(f"\nrollout: {K} cycles. drift estimate  lag-growth rho = {rho_lag*100:+.2f}%   "
      f"Hilbert rho = {rho_hil*100:+.2f}%   final-cycle amplitude ratio {amp:.3f}")
print("  lag per cycle (samples):", np.round(lags, 2))

# ---------------- 4. the check, per rollout cycle ----------------
law = json.load(open(args.law))["law"] if os.path.exists(args.law) else {"c": 0.194, "p": 0.641}
bins = [0, 0.02, 0.1, 0.3, 1.0, 10.0]
rows, keep = [], {}
print(f"\n{'cyc':>3} {'L2':>6} {'lag':>6} | {'wall T/R':>13} | {'sac T/R':>13} | {'OSI>0.1 T/R':>13} | {'rho=roll/floor':>14}")
for k in range(K):
    wp = decode(roll_c[k * N:(k + 1) * N])
    tp, op, rp, _ = biomarkers(wp)
    eps = float(np.linalg.norm(wp - w3) / np.linalg.norm(w3))
    eT, eO, eR = rel(tp, ta0), rel(op, osi0), rel(rp, rrt0)
    reg = {n: dict(tawss=aw(eT, s), osi=aw(eO, s), rrt=aw(eR, s)) for n, s in REG.items()}
    ratio = reg["wall"]["rrt"] / max(floor["wall"]["rrt"], 1e-4)   # floor below 0.01% counts as 0.01%
    # law: predicted vs measured excess, binned by cancellation factor
    pred = law["c"] * eps * X0 ** law["p"]
    meas = np.abs(eR) - np.abs(eT)
    lawtab = []
    for lo, up in zip(bins[:-1], bins[1:]):
        s = (X0 >= lo) & (X0 < up)
        if s.sum() >= 5:
            lawtab.append(dict(X=[lo, up], n=int(s.sum()), pred=float(np.median(pred[s])),
                               meas=float(np.median(meas[s]))))
    rows.append(dict(cycle=k + 1, field_l2=eps, lag=float(lags[k]), regions=reg,
                     rho_diag=float(ratio), law=lawtab))
    f = lambda n: f"{reg[n]['tawss']*100:5.2f}/{reg[n]['rrt']*100:5.2f}%"
    print(f"{k+1:3d} {eps*100:5.1f}% {lags[k]:6.2f} | {f('wall'):>13} | {f('sac'):>13} | {f('osi>0.1'):>13} | {ratio:14.1f}")
    if k in (0, K - 1):
        keep[f"cycle{k+1}"] = np.stack([tp, op, rp]).astype(np.float32)

print("\nlaw check on the last cycle (median per-face excess RRT error beyond TAWSS):")
print(f"  {'1/R-1 bin':>14} {'faces':>6} {'predicted':>10} {'measured':>10}")
for b in rows[-1]["law"]:
    print(f"  [{b['X'][0]:5.2f},{b['X'][1]:5.2f}) {b['n']:6d} {b['pred']*100:9.2f}% {b['meas']*100:9.2f}%")
print("\nrho diagnostic: >1 means rollout drift, not the compressor, dominates the RRT error")

# ---------------- save ----------------
os.makedirs("data", exist_ok=True)
np.savez_compressed("data/pod_lstm_results.npz", roll=roll_c.astype(np.float32), truth=coef.astype(np.float32),
                    lags=lags, **keep)
json.dump(dict(args=vars(args), r=r, energy=float(energy), floor_l2=floor_l2, floor=floor,
               one_step_rmse=one_step, rho_lag=float(rho_lag), rho_hilbert=float(rho_hil),
               amplitude_ratio=float(amp), cycles=rows),
          open("data/pod_lstm_summary.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
tax = np.arange(T) / N
for i, c in zip(range(3), ("C0", "C1", "C2")):
    ax[0].plot(tax, truth_z[:, i], c=c, lw=1, alpha=.4)
    ax[0].plot(tax, roll[:, i], c=c, lw=1.2, ls="--", label=f"mode {i+1}")
ax[0].set(xlabel="cycles", ylabel="normalised coefficient", title="rollout (dashed) vs truth (faint)")
ax[0].legend(fontsize=8)
cyc = np.arange(1, K + 1)
ax[1].plot(cyc, [x["field_l2"] * 100 for x in rows], "k-o", label="field L2")
for n, c in (("wall", "C0"), ("osi>0.1", "C3")):
    ax[1].plot(cyc, [x["regions"][n]["tawss"] * 100 for x in rows], c=c, ls="-", marker=".", label=f"TAWSS {n}")
    ax[1].plot(cyc, [x["regions"][n]["rrt"] * 100 for x in rows], c=c, ls="--", marker=".", label=f"RRT {n}")
ax[1].axhline(floor["wall"]["rrt"] * 100, c="grey", lw=.8, ls=":", label="compressor floor (RRT)")
ax[1].set(xlabel="rollout cycle", ylabel="error (%)", yscale="log", title="error growth")
ax[1].legend(fontsize=7)
ax[2].plot(cyc, lags, "k-o")
ax[2].set(xlabel="rollout cycle", ylabel="phase lag (samples)",
          title=f"drift: rho ≈ {rho_lag*100:+.2f}% per step")
fig.tight_layout(); os.makedirs("analysis", exist_ok=True); fig.savefig("analysis/pod_lstm.png", dpi=140)
print("wrote data/pod_lstm_results.npz, data/pod_lstm_summary.json, analysis/pod_lstm.png")
