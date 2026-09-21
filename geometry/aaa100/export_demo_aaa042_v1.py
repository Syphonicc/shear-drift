#!/usr/bin/env python3
"""
export_demo_aaa042_v1.py -- package AAA042 (v6, periodic, cycle 17) wall data
for the web explorer, same schema as ../analysis/export_demo.py (cerebral case).

Reads   wss_data_aaa042_v6_3cyc.npz   (120, F, 3) kinematic WSS, cycles 15-17
        case_aaa042_v6_result/        OpenFOAM case (mesh + case.foam)
Writes  ../data/aaa042/mesh.bin     float32 points (mm) + uint32 triangle indices
        ../data/aaa042/fields.bin   float32 [levels, 3, F]  TAWSS(Pa), OSI, RRT(1/Pa)
        ../data/aaa042/extra.bin    float32 [2, F]  X (cancellation factor), sac (0/1)
        ../data/aaa042/meta.json    levels, error stats, law constants

Sac defined geometrically (independent of OSI): z in [1.554, 1.634] m, where the
local mean wall radius from the cross-section centroid exceeds ~25mm vs. the
~13-14mm baseline vessel radius near the inlet/outlets (checked separately).

Run from geometry/aaa100/:  python export_demo_aaa042_v1.py
"""
import json, os
import numpy as np
import pyvista as pv

RHO_BLOOD = 1060.0
T_CYCLE = 0.9
DT = 0.0225
LEVELS = np.round(np.arange(0, 0.0501, 0.005), 4)   # 0 .. 5% detuning, same as cerebral
SAC_Z = (1.554, 1.634)                               # m, geometric bulge band
OUT = "../../data/aaa042"
os.makedirs(OUT, exist_ok=True)

# ---------- load last-3-cycle WSS + mesh ----------
d = np.load("wss_data_aaa042_v6_3cyc.npz")
wss_all, t = d["wss"].astype(np.float64) * RHO_BLOOD, d["times"]   # Pa
areas = d["areas"]
N, F, _ = wss_all.shape
print(f"loaded {N} snapshots, {F} faces (quads/pentagons, pre-triangulation)")

reader = pv.POpenFOAMReader("case_aaa042_v6_result/case.foam")
reader.cell_to_point_creation = False
reader.set_active_time_value(15.3)
mesh = reader.read()
wall = mesh["boundary"]["wall"]
assert wall.n_cells == F, f"mesh has {wall.n_cells} cells, WSS has {F}"
centres = np.asarray(wall.cell_centers().points)

# ---------- biomarkers on the ORIGINAL (pre-triangulation) cells ----------
def biomarkers(w):
    """w: (nt, F, 3) WSS in Pa -> tawss, osi, rrt, R, netvec (all len-F, or netvec Fx3)."""
    mag = np.linalg.norm(w, axis=2)
    tawss = mag.sum(axis=0) * DT / T_CYCLE
    netvec = w.sum(axis=0) * DT / T_CYCLE
    netmag = np.linalg.norm(netvec, axis=1)
    R = np.clip(netmag / np.maximum(tawss, 1e-30), 0.0, 1.0)
    osi = 0.5 * (1.0 - R)
    rrt = 1.0 / np.maximum(R * tawss, 1e-30)
    return tawss, osi, rrt, R


def detune(w_full, rho, nt):
    """Clock-rate error without wraparound (same as mechanism_test.py)."""
    idx = np.arange(nt) * (1.0 + rho)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, w_full.shape[0] - 1)
    f = (idx - lo)[:, None, None]
    return w_full[lo] * (1.0 - f) + w_full[hi] * f


nt = 40
w2 = wss_all[-2 * nt:]                   # cycles 16-17, the window detune() resamples over
w0 = wss_all[-nt:]                       # cycle 17, the converged baseline
tawss0, osi0, rrt0, R0 = biomarkers(w0)
X = np.where(R0 > 1e-6, 1.0 / np.maximum(R0, 1e-6) - 1.0, 1e6)   # cancellation factor

sac = (centres[:, 2] >= SAC_Z[0]) & (centres[:, 2] <= SAC_Z[1])
print(f"baseline area-avg TAWSS {np.average(tawss0, weights=areas):.4g} Pa  "
      f"OSI {np.average(osi0, weights=areas):.4g}  sac faces {sac.sum()} ({100*areas[sac].sum()/areas.sum():.2f}% area)")

# ---------- per-level fields + stats (relative error vs baseline, mirrors export_demo.py) ----------
fields = np.empty((len(LEVELS), 3, F), dtype=np.float32)
fields[0] = [tawss0, osi0, rrt0]
hi = osi0 > 0.1
masks = {"wall": np.ones(F, bool), "sac": sac, "osi_gt_0.1": hi}


def aw(v, m):
    return float(np.sum(np.abs(v[m]) * areas[m]) / np.sum(areas[m])) if m.any() else 0.0


def rel(a, b):
    return np.divide(a - b, b, out=np.zeros_like(b), where=np.abs(b) > 1e-6)


stats = [{"field_l2": 0.0, **{k: {"tawss": 0.0, "osi": 0.0, "rrt": 0.0, "rrt_p95": 0.0} for k in masks}}]

for lvl in LEVELS[1:]:
    wd = detune(w2, lvl, nt)
    ta, os_, rr, _ = biomarkers(wd)
    fields[len(stats)] = [ta, os_, rr]
    l2 = float(np.linalg.norm(wd - w0) / np.linalg.norm(w0))
    eT, eO, eR = rel(ta, tawss0), rel(os_, osi0), rel(rr, rrt0)
    row = {"field_l2": l2}
    for k, m in masks.items():
        row[k] = {"tawss": aw(eT, m), "osi": aw(eO, m), "rrt": aw(eR, m),
                   "rrt_p95": float(np.percentile(np.abs(eR[m]), 95)) if m.any() else 0.0}
    stats.append(row)
    print(f"  level {lvl:.3f}  field_l2 {l2:.4f}  wall RRT err {row['wall']['rrt']*100:.2f}%")

# ---------- mesh: attach everything as cell data, triangulate ONCE ----------
wall["X"], wall["sac"] = X, sac.astype(np.uint8)
for i, lvl in enumerate(LEVELS):
    wall[f"ta_{i}"] = fields[i, 0]
    wall[f"osi_{i}"] = fields[i, 1]
    wall[f"rrt_{i}"] = fields[i, 2]

tri_mesh = wall.triangulate()
Ft0 = tri_mesh.n_cells
print(f"triangulated: {F} cells -> {Ft0} triangles, {tri_mesh.n_points} points")

# ---------- decimate for display (geometry only), remap fields by nearest original centroid ----------
orig_centres = np.asarray(tri_mesh.cell_centers().points)
orig_X = np.asarray(tri_mesh.cell_data["X"])
orig_sac = np.asarray(tri_mesh.cell_data["sac"])
orig_fields = np.stack([[tri_mesh.cell_data[f"ta_{i}"], tri_mesh.cell_data[f"osi_{i}"], tri_mesh.cell_data[f"rrt_{i}"]]
                         for i in range(len(LEVELS))])   # [levels, 3, Ft0]

TARGET_TRI = 32000
reduction = max(0.0, 1.0 - TARGET_TRI / Ft0)
dec = tri_mesh.decimate(reduction)
pts = np.asarray(dec.points) * 1e3   # m -> mm
tri = dec.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)
Ft = len(tri)
print(f"decimated: {Ft0} -> {Ft} triangles ({100*reduction:.0f}% target reduction)")

from scipy.spatial import cKDTree
dec_centres = np.asarray(dec.cell_centers().points)
_, nn = cKDTree(orig_centres).query(dec_centres)
tri_X = orig_X[nn]
tri_sac = orig_sac[nn]
fields_t = orig_fields[:, :, nn].astype(np.float32)   # [levels, 3, Ft]

# ---------- write ----------
with open(f"{OUT}/mesh.bin", "wb") as f:
    f.write(pts.astype("<f4").tobytes())
    f.write(tri.astype("<u4").tobytes())
fields_t.astype("<f4").tofile(f"{OUT}/fields.bin")
open(f"{OUT}/extra.bin", "wb").write(tri_X.astype("<f4").tobytes() + tri_sac.astype(np.uint8).tobytes())

meta = {
    "n_points": int(len(pts)), "n_faces": int(Ft),
    "levels": [float(x) for x in LEVELS],
    "baseline": {
        "tawss_mean": aw(tawss0, np.ones(F, bool)), "osi_mean": aw(osi0, np.ones(F, bool)),
        "osi_gt_0p1_area": float(areas[hi].sum() / areas.sum()), "sac_faces": int(sac.sum()),
    },
    "stats": stats,
    "law": {"c": 0.194, "p": 0.641, "r2": 0.956, "n": 279},
    "sweep": [],
    "provenance": "AAA042 (AAA-100 dataset), v6 restart, cycle 17 (17 cycles total, 2.66% periodicity floor)",
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1, allow_nan=False)
print(f"wrote {OUT}/mesh.bin fields.bin extra.bin meta.json  ({Ft} triangles)")
