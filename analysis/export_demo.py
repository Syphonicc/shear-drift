#!/usr/bin/env python3
"""
export_demo.py -- package Run #2 wall data for the web explorer.

Reads   ../results_v2/wss_data.npz              (120, F, 3) kinematic WSS
        ../results_v2/VTK/results_v2_0/boundary/wall.vtp   (surface mesh)
        analysis/womersley_sweep.csv            (optional, for the law fit)
Writes  data/mesh.bin     float32 points (mm) + uint32 triangle indices
        data/fields.bin   float32 [levels, 3, F]  TAWSS(Pa), OSI, RRT(1/Pa)
        data/meta.json    levels, error stats, law constants, sweep points

Run from the hackathon/ directory:  python analysis/export_demo.py
"""
import json, os, sys
import numpy as np

RES = os.environ.get("RES", "../results_v2")
RHO_BLOOD = 1060.0
LEVELS = np.round(np.arange(0, 0.0501, 0.005), 4)       # 0 .. 5 % detuning
SAC_C, SAC_R = np.array([5.40, -4.00, 0.78]), 2.0       # mm, from Run #1 analysis
os.makedirs("data", exist_ok=True)

# ---------- load ----------
d = np.load(f"{RES}/wss_data.npz")
w = d["wss"][80:120].astype(np.float64) * RHO_BLOOD      # cycle 3, Pa
A, cen = d["areas"], d["centres"] * 1e3
N, F, _ = w.shape

try:
    import pyvista as pv
    m = pv.read(f"{RES}/VTK/results_v2_0/boundary/wall.vtp")
    pts = np.asarray(m.points, dtype=np.float64) * 1e3
    tri = m.faces.reshape(-1, 4)[:, 1:]
except ImportError:                                       # test mode
    pts, tri = d["_pts"] * 1e3, d["_tri"]
assert len(tri) == F, f"mesh has {len(tri)} faces, WSS has {F}"
gap = np.abs(pts[tri].mean(1) - cen).max()
assert gap < 1e-2, f"face order mismatch: centroid gap {gap:.3g} mm"
print(f"mesh {len(pts)} pts / {F} faces, face order verified (gap {gap:.1e} mm)")

# ---------- physics ----------
k = np.fft.fftfreq(N, 1 / N)
W = np.fft.fft(w, axis=0)

def detune(rho):
    E = np.exp(2j * np.pi * np.outer(np.arange(N) * (1 + rho), k) / N)
    return np.real(np.einsum("jk,kfi->jfi", E, W)) / N

def biomarkers(x):
    mag = np.linalg.norm(x, axis=2)
    ta = mag.mean(0)
    R = np.linalg.norm(x.sum(0), axis=1) / np.maximum(mag.sum(0), 1e-30)
    R = np.clip(R, 1e-3, 1.0)
    return ta, (1 - R) / 2, 1 / (R * ta), R

ta0, osi0, rrt0, R0 = biomarkers(w)
X0 = 1 / R0 - 1                                           # cancellation factor
sac = np.linalg.norm(cen - SAC_C, axis=1) < SAC_R
hi = osi0 > 0.1
def aw(v, s):                                             # area-weighted mean |v|, NaN-safe
    return float(np.sum(np.abs(v[s]) * A[s]) / np.sum(A[s])) if s.any() else 0.0

def rel(a, b):                                            # relative error, 0 where baseline ~0
    return np.divide(a - b, b, out=np.zeros_like(b), where=np.abs(b) > 1e-6)

fields = np.zeros((len(LEVELS), 3, F), np.float32)
stats = []
for i, rho in enumerate(LEVELS):
    x = w if rho == 0 else detune(rho)
    ta, osi, rrt, _ = biomarkers(x)
    fields[i] = ta, osi, rrt
    eps = float(np.linalg.norm(x - w) / np.linalg.norm(w))
    eT, eO, eR = rel(ta, ta0), rel(osi, osi0), rel(rrt, rrt0)
    row = dict(rho=float(rho), field_l2=eps)
    for tag, s in (("wall", np.ones(F, bool)), ("sac", sac), ("osi_gt_0.1", hi)):
        row[tag] = dict(tawss=aw(eT, s), osi=aw(eO, s), rrt=aw(eR, s),
                        rrt_p95=float(np.percentile(np.abs(eR[s]), 95)) if s.any() else 0.0)
    stats.append(row)
    print(f"rho={rho:.3f}  L2 {eps*100:5.2f}%  wall RRT {row['wall']['rrt']*100:5.2f}%  "
          f"OSI>0.1 RRT {row['osi_gt_0.1']['rrt']*100:5.2f}%")

# ---------- law constants from the Womersley sweep ----------
law, sweep_pts = None, []
csv = "analysis/womersley_sweep.csv"
if os.path.exists(csv):
    S = np.genfromtxt(csv, delimiter=",", names=True)
    ok = S["R"] < 0.95
    x, y = (1 - S["R"][ok]) / S["R"][ok], S["err_R"][ok] / S["field_eps"][ok]
    p, lc = np.polyfit(np.log(x), np.log(y), 1)
    law = dict(c=float(np.exp(lc)), p=float(p),
               r2=float(np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2), n=int(ok.sum()))
    sel = np.linspace(0, ok.sum() - 1, min(150, ok.sum())).astype(int)
    sweep_pts = [[round(float(a), 4), round(float(b), 4), int(al)]
                 for a, b, al in zip(x[sel], y[sel], S["alpha"][ok][sel])]
    print(f"law: (dR/R)/eps = {law['c']:.3f} * (1/R-1)^{law['p']:.3f}   R2={law['r2']:.3f}")
else:
    print("note: analysis/womersley_sweep.csv not found -- run womersley_sweep.py there first")

# ---------- write ----------
with open("data/mesh.bin", "wb") as f:
    f.write(pts.astype(np.float32).tobytes())
    f.write(tri.astype(np.uint32).tobytes())
fields.tofile("data/fields.bin")
extra = np.stack([X0, sac.astype(float)]).astype(np.float32)   # cancellation factor, sac mask
extra.tofile("data/extra.bin")

meta = dict(
    n_points=int(len(pts)), n_faces=int(F), levels=LEVELS.tolist(),
    fields=["TAWSS_Pa", "OSI", "RRT_perPa"], extra=["cancellation_factor", "sac_mask"],
    layout="mesh.bin: float32 xyz*n_points then uint32 abc*n_faces; "
           "fields.bin: float32 [levels,3,faces]; extra.bin: float32 [2,faces]",
    baseline=dict(tawss_mean=aw(ta0, np.ones(F, bool)), osi_mean=aw(osi0, np.ones(F, bool)),
                  osi_gt_0p1_area=float(A[hi].sum() / A.sum()), sac_faces=int(sac.sum())),
    stats=stats, law=law, sweep=sweep_pts,
    provenance="Run #2 CFD (pre-hackathon); detuning, biomarkers and export built during hackathon",
)
json.dump(meta, open("data/meta.json", "w"), indent=1, allow_nan=False)
sz = sum(os.path.getsize(f"data/{n}") for n in ("mesh.bin", "fields.bin", "extra.bin", "meta.json"))
print(f"wrote data/ ({sz/1e6:.1f} MB)")
