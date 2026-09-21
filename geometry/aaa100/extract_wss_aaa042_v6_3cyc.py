import numpy as np
import pyvista as pv
from pathlib import Path

CASE = Path(__file__).parent / "case_aaa042_v6_result"
FOAM_FILE = CASE / "case.foam"

reader = pv.POpenFOAMReader(str(FOAM_FILE))
reader.cell_to_point_creation = False
times = [t for t in reader.time_values if t > 12.6 + 1e-9 and t <= 15.3 + 1e-9]
print(f"using {len(times)} timesteps (t={times[0]} to {times[-1]})")

wss = None
areas = None
for i, t in enumerate(times):
    reader.set_active_time_value(t)
    mesh = reader.read()
    wall = mesh["boundary"]["wall"]
    if wss is None:
        sized = wall.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sized.cell_data["Area"])
        wss = np.empty((len(times), wall.n_cells, 3), dtype=np.float64)
    wss[i] = np.asarray(wall.cell_data["wallShearStress"])

times_arr = np.array(times)
out = CASE.parent / "wss_data_aaa042_v6_3cyc.npz"
np.savez_compressed(str(out), wss=wss, areas=areas, times=times_arr)
print(f"wss shape {wss.shape}")
print(f"wrote {out}")
