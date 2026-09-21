"""Cut the flow-boundary caps off the closed AAA042 STL, producing:
  - wall_open.stl  : the vessel wall, open at aorta inlet + 2 iliac outlets,
                      still closed (wall) at the 2 renal stumps.
  - <patch>.stl     : a flat triangulated disk plugging each of the 3 holes,
                      used as the matching inlet/outlet patch surface.

No CFD here — pure geometry via pyvista/vtk, for snappyHexMesh input.
"""
import pyvista as pv
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "snappy_geo_v2"
OUT.mkdir(exist_ok=True)

mesh = pv.read(str(ROOT / "meshes" / "meshes" / "AAA042.stl"))
openings = json.load(open(ROOT / "AAA042_openings.json"))

FLOW_PATCHES = ["aorta_inlet", "iliac_left_outlet", "iliac_right_outlet"]
BOX_MARGIN = 1.3  # local clip box half-width = diameter * this margin

patch_meta = {}
wall = mesh

for name in FLOW_PATCHES:
    info = openings[name]
    center = np.array(info["center"])
    normal = np.array(info["normal"])
    normal = normal / np.linalg.norm(normal)
    diam = info["diameter_mm"]
    R = diam * BOX_MARGIN

    # bounded local clip (NOT an infinite plane) -- an infinite plane clip
    # was slicing through the far side of this compact, bent geometry too.
    box = pv.Cube(center=center, x_length=2 * R, y_length=2 * R, z_length=2 * R)
    wall = wall.clip_surface(box, invert=False)  # keep the outside-the-box (main body) piece

    edges = wall.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False, manifold_edges=False, feature_edges=False
    )
    conn = edges.connectivity()
    region_ids = np.unique(conn["RegionId"])
    # pick the loop closest to our expected center (should be the only new one,
    # but guard against stray artifacts from earlier clips)
    best_rid, best_d = None, np.inf
    for rid in region_ids:
        pts = conn.points[conn["RegionId"] == rid]
        c = pts.mean(axis=0)
        d = np.linalg.norm(c - center)
        if d < best_d:
            best_d, best_rid = d, rid
    loop_pts = conn.points[conn["RegionId"] == best_rid]

    disk = pv.PolyData(loop_pts).delaunay_2d()
    area = disk.compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum()
    diam = 2 * np.sqrt(area / np.pi)

    disk_path = OUT / f"{name}.stl"
    disk.save(str(disk_path), binary=True)

    patch_meta[name] = dict(
        center=loop_pts.mean(axis=0).tolist(),
        normal=normal.tolist(),
        diameter_mm=float(diam),
        n_boundary_pts=int(len(loop_pts)),
    )
    print(f"{name}: diam={diam:.1f}mm boundary_pts={len(loop_pts)} -> {disk_path.name}")

wall_path = OUT / "wall_open.stl"
wall.save(str(wall_path), binary=True)
print(f"wall: n_points={wall.n_points} n_cells={wall.n_cells} -> {wall_path.name}")

# sanity: wall should still be closed except for the 3 holes we just cut
remaining_edges = wall.extract_feature_edges(
    boundary_edges=True, non_manifold_edges=False, manifold_edges=False, feature_edges=False
)
remaining_conn = remaining_edges.connectivity()
n_remaining_loops = len(np.unique(remaining_conn["RegionId"])) if remaining_edges.n_points else 0
print(f"remaining open boundary loops on wall: {n_remaining_loops} (expect 3)")

json.dump(patch_meta, open(OUT / "patch_meta.json", "w"), indent=2)
print("wrote patch_meta.json")
