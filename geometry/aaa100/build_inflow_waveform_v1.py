"""Infrarenal aortic inflow waveform for AAA042, built from published landmark
values (NOT a digitized patient trace): triphasic shape from Mills et al. /
Olufsen et al., the standard reference used for AAA CFD inlets --
systolic peak ~45 cm/s, early-diastolic reversal trough ~-15 cm/s,
small second (reflected-wave) peak ~2.7 cm/s, settling near-zero through
diastole. Landmark (phase, velocity) points are connected with a monotone
cubic (PCHIP) interpolant to avoid ringing, then resampled onto this
project's existing cycle convention: T=0.9s, 40 samples/cycle (dt=0.0225s,
matching controlDict writeInterval in results_v2).

Flow rate Q = V * inlet_area, inlet_area from AAA042_openings.json
(aorta_inlet diameter = 28.1mm -> area = 6.201e-4 m^2).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
import json
from pathlib import Path

ROOT = Path(__file__).parent
T_CYCLE = 0.9  # s, matches project convention (40 samples x 0.0225s)
N_SAMPLES = 40

diam_mm = json.load(open(ROOT / "snappy_geo_v3" / "patch_meta.json"))["aorta_inlet"]["diameter_mm"]
area_m2 = np.pi * (diam_mm / 1000 / 2) ** 2

# (phase 0-1, velocity m/s) landmarks -- triphasic infrarenal shape
landmarks = [
    (0.00, 0.020),
    (0.05, 0.150),
    (0.12, 0.450),   # systolic peak ~45 cm/s
    (0.20, 0.250),
    (0.30, 0.020),
    (0.37, -0.150),  # reversal trough ~-15 cm/s
    (0.45, -0.020),
    (0.50, 0.027),   # second peak ~2.7 cm/s
    (0.60, 0.015),
    (0.75, 0.010),
    (1.00, 0.020),   # periodic: matches phase 0
]
phases = np.array([p for p, v in landmarks])
vels = np.array([v for p, v in landmarks])
interp = PchipInterpolator(phases, vels)

t = np.linspace(0, T_CYCLE, N_SAMPLES, endpoint=False)
phase = t / T_CYCLE
v = interp(phase)
q = v * area_m2

mean_v = np.trapz(np.append(v, v[0]), np.append(phase, 1.0))
print(f"inlet area: {area_m2:.4e} m^2 (diam {diam_mm:.1f}mm)")
print(f"peak Q: {q.max():.4e} m^3/s, trough Q: {q.min():.4e} m^3/s, mean V: {mean_v:.4f} m/s")

table_lines = "\n".join(f"    ({t[i]:.4f} {q[i]:.6e})" for i in range(N_SAMPLES))
out = ROOT / "aaa042_inflow_table_v1.txt"
out.write_text(table_lines + "\n")
print(f"wrote {out}")
