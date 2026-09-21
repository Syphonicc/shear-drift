"""Extract wallShearStress time series from the AAA042 case, same pattern as
results_v2/extract_wss.py: foamToVTK writes VTK/case_aaa042_v1_<n>/boundary/wall.vtp
per timestep, this reads wallShearStress off each and stacks into one array.

Run `foamToVTK -latestTime` or `foamToVTK` (all times) from the case dir first.
"""
import numpy as np, pyvista as pv, glob, re
from pathlib import Path

CASE = Path(__file__).parent / "case_aaa042_v1"
files = sorted(
    glob.glob(str(CASE / "VTK" / "case_aaa042_v1_*" / "boundary" / "wall.vtp")),
    key=lambda f: int(re.search(r"case_aaa042_v1_(\d+)", f).group(1)),
)
print(f"found {len(files)} files")
if not files:
    raise SystemExit("no VTK wall.vtp files found -- run foamToVTK in case_aaa042_v1 first")

m0 = pv.read(files[0])
nc = m0.n_cells
sized = m0.compute_cell_sizes(length=False, area=True, volume=False)
areas = np.asarray(sized.cell_data["Area"])
centres = np.asarray(m0.cell_centers().points)

wss = np.empty((len(files), nc, 3), dtype=np.float64)
for i, f in enumerate(files):
    wss[i] = pv.read(f).cell_data["wallShearStress"]

# actual write times come from the VTK series index * writeInterval (0.0225s)
times = np.arange(1, len(files) + 1) * 0.0225
np.savez_compressed(str(CASE / "wss_data_aaa042.npz"), wss=wss, areas=areas, centres=centres, times=times)
print("wss", wss.shape, "| area total %.4e m^2" % areas.sum())
print("t:", times[0], "->", times[-1])
