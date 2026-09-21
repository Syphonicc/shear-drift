"""Off-screen render of top AAA screening candidates for visual check."""
import pyvista as pv
from pathlib import Path
import sys

pv.OFF_SCREEN = True

ROOT = Path(__file__).parent
MESH_DIR = ROOT / "meshes" / "meshes"
OUT_DIR = ROOT / "renders"
OUT_DIR.mkdir(exist_ok=True)

cases = sys.argv[1:] if len(sys.argv) > 1 else ["AAA029", "AAA042", "AAA032", "AAA099", "AAA010"]

for case in cases:
    stl = MESH_DIR / f"{case}.stl"
    if not stl.exists():
        print(f"missing {case}")
        continue
    mesh = pv.read(str(stl))
    p = pv.Plotter(off_screen=True, window_size=(900, 900))
    p.add_mesh(mesh, color="lightcoral", opacity=1.0, smooth_shading=True)
    p.camera_position = "xz"
    p.camera.azimuth = 20
    p.camera.elevation = 10
    out = OUT_DIR / f"{case}.png"
    p.screenshot(str(out))
    p.close()
    print(f"wrote {out}")
