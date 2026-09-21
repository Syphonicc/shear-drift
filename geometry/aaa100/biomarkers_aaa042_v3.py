"""TAWSS / OSI / RRT for AAA042 v5 (fine-mesh sensitivity check), same cancellation-law
definitions as results_v2/biomarkers.py: M=int|tau|dt, S=|int tau dt|, R=S/M,
TAWSS=M/T, OSI=(1-R)/2, RRT=1/((1-2*OSI)*TAWSS). Last cycle (t=3.6-4.5).
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
d = np.load(ROOT / "wss_data_aaa042_v5.npz")
wss, t, areas = d["wss"], d["times"], d["areas"]
rho = 1060.0
T = 0.9
dt = 0.0225

w = wss * rho  # kinematic -> Pa

mag = np.linalg.norm(w, axis=2)
tawss = mag.sum(axis=0) * dt / T
netvec = w.sum(axis=0) * dt / T
netmag = np.linalg.norm(netvec, axis=1)

osi = 0.5 * (1.0 - netmag / np.maximum(tawss, 1e-30))
rrt = 1.0 / np.maximum((1.0 - 2.0 * osi) * tawss, 1e-30)

np.savez_compressed(ROOT / "biomarkers_aaa042_v5_lastcycle.npz", tawss=tawss, osi=osi, rrt=rrt, areas=areas)

aw = areas / areas.sum()
for n, v in [("TAWSS (Pa)", tawss), ("OSI", osi), ("RRT (1/Pa)", rrt)]:
    print(f"{n:12s} area-avg {np.sum(v*aw):10.4g}  min {v.min():10.4g}  max {v.max():10.4g}")
print("snapshots used:", len(t), f" (t={t[0]:.4f} to {t[-1]:.4f})")
print("OSI>0.1 area frac: %.4f" % aw[osi > 0.1].sum())
print("OSI>0.2 area frac: %.4f" % aw[osi > 0.2].sum())
