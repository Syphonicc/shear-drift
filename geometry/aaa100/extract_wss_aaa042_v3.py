"""Extract wallShearStress time series from the AAA042 v4 result, reading
directly via pyvista's native OpenFOAM reader (no foamToVTK round trip needed
-- CloudHPC only gave us reconstructed raw time directories).

Reads the last cycle (t=3.6 to 4.5, 40 samples) from case_aaa042_v5_result/.
"""
import numpy as np
import pyvista as pv
from pathlib import Path

CASE = Path(__file__).parent / "case_aaa042_v5_result"
FOAM_FILE = CASE / "case.foam"

reader = pv.POpenFOAMReader(str(FOAM_FILE))
reader.cell_to_point_creation = False
print("available time values:", len(reader.time_values))

times = [t for t in reader.time_values if t > 3.6 + 1e-9 and t <= 4.5 + 1e-9]
print(f"using {len(times)} timesteps in last cycle (t={times[0]} to {times[-1]})")

wss = None
areas = None
centres = None

for i, t in enumerate(times):
    reader.set_active_time_value(t)
    mesh = reader.read()
    wall = mesh["boundary"]["wall"]

    if wss is None:
        sized = wall.compute_cell_sizes(length=False, area=True, volume=False)
        areas = np.asarray(sized.cell_data["Area"])
        centres = np.asarray(wall.cell_centers().points)
        wss = np.empty((len(times), wall.n_cells, 3), dtype=np.float64)

    wss[i] = np.asarray(wall.cell_data["wallShearStress"])

times_arr = np.array(times)
out = CASE.parent / "wss_data_aaa042_v5.npz"
np.savez_compressed(str(out), wss=wss, areas=areas, centres=centres, times=times_arr)
print(f"wss shape {wss.shape} | area total {areas.sum():.4e} m^2")
print(f"wrote {out}")
