"""TAWSS / OSI / RRT for AAA042, same cancellation-law definitions as
results_v2/biomarkers.py: M=int|tau|dt, S=|int tau dt|, R=S/M,
TAWSS=M/T, OSI=(1-R)/2, RRT=1/((1-2*OSI)*TAWSS).

Uses the LAST full cycle written (cycles 3-5 are kept per purgeWrite=120;
this defaults to cycle 5, i.e. t in (3.6, 4.5]).
"""
import numpy as np
from pathlib import Path

CASE = Path(__file__).parent / "case_aaa042_v1"
d = np.load(CASE / "wss_data_aaa042.npz")
wss, t, areas = d["wss"], d["times"], d["areas"]
rho = 1060.0
T = 0.9
dt = 0.0225

t_end = t.max()
t_start = t_end - T
m = (t > t_start + 1e-9) & (t <= t_end + 1e-9)
w = wss[m] * rho  # kinematic -> Pa

mag = np.linalg.norm(w, axis=2)
tawss = mag.sum(axis=0) * dt / T
netvec = w.sum(axis=0) * dt / T
netmag = np.linalg.norm(netvec, axis=1)

osi = 0.5 * (1.0 - netmag / np.maximum(tawss, 1e-30))
rrt = 1.0 / np.maximum((1.0 - 2.0 * osi) * tawss, 1e-30)

np.savez_compressed(CASE / "biomarkers_aaa042_lastcycle.npz", tawss=tawss, osi=osi, rrt=rrt, areas=areas)

aw = areas / areas.sum()
for n, v in [("TAWSS (Pa)", tawss), ("OSI", osi), ("RRT (1/Pa)", rrt)]:
    print(f"{n:12s} area-avg {np.sum(v*aw):10.4g}  min {v.min():10.4g}  max {v.max():10.4g}")
print("snapshots used:", m.sum(), f" (t={t_start:.3f} to {t_end:.3f})")
print("OSI>0.1 area frac: %.3f" % aw[osi > 0.1].sum())
