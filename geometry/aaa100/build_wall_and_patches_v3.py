"""Cut the flow-boundary caps off the closed AAA042 STL, producing:
  - wall_open.stl   : vessel wall, open at aorta inlet + 2 iliac outlets,
                       still closed (wall) at the 2 renal stumps.
  - <patch>.stl      : a flat triangulated disk plugging each of the 3 holes.

Removal is done by a direct cell-centroid mask (sphere-bounded half-space),
not vtk's clip/clip_surface filters -- those produced holes tens of mm from
the intended location on this mesh (vtkImplicitPolyDataDistance sign issue
with a small local box on a large, bent, closed surface). No CFD here, pure
geometry via pyvista/vtk, for snappyHexMesh input.
"""
import pyvista as pv
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "snappy_geo_v3"
OUT.mkdir(exist_ok=True)

RADIUS_MARGIN = 2.0   # local sphere radius = diameter * this
PLANE_INSET = 2.0     # mm, cut plane pushed inward from the raw cap centroid

mesh = pv.read(str(ROOT / "meshes" / "meshes" / "AAA042.stl"))
openings = json.load(open(ROOT / "AAA042_openings.json"))
FLOW_PATCHES = ["aorta_inlet", "iliac_left_outlet", "iliac_right_outlet"]


def remove_local_cap(surf, center, normal, diam):
    normal = normal / np.linalg.norm(normal)
    surf = surf.extract_surface(algorithm="dataset_surface").triangulate()
    cent = surf.cell_centers().points
    origin_in = center - normal * PLANE_INSET
    beyond_plane = np.dot(cent - origin_in, normal) > 0
    within_sphere = np.linalg.norm(cent - center, axis=1) < diam * RADIUS_MARGIN
    remove_mask = beyond_plane & within_sphere
    keep_ids = np.where(~remove_mask)[0]
    kept = surf.extract_cells(keep_ids).extract_surface(algorithm="dataset_surface")
    return kept


patch_meta = {}
wall = mesh
for name in FLOW_PATCHES:
    info = openings[name]
    center = np.array(info["center"])
    normal = np.array(info["normal"])
    diam = info["diameter_mm"]

    wall = remove_local_cap(wall, center, normal, diam)

    edges = wall.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False, manifold_edges=False, feature_edges=False
    )
    conn = edges.connectivity()
    region_ids = np.unique(conn["RegionId"])
    best_rid, best_d = None, np.inf
    for rid in region_ids:
        pts = conn.points[conn["RegionId"] == rid]
        d = np.linalg.norm(pts.mean(axis=0) - center)
        if d < best_d:
            best_d, best_rid = d, rid
    loop_pts = conn.points[conn["RegionId"] == best_rid]
    assert best_d < diam, f"{name}: nearest loop is {best_d:.1f}mm from expected center, something's wrong"

    disk = pv.PolyData(loop_pts).delaunay_2d()
    area = disk.compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum()
    out_diam = 2 * np.sqrt(area / np.pi)

    disk_path = OUT / f"{name}.stl"
    disk.save(str(disk_path), binary=True)

    patch_meta[name] = dict(
        center=loop_pts.mean(axis=0).tolist(),
        normal=normal.tolist(),
        diameter_mm=float(out_diam),
        n_boundary_pts=int(len(loop_pts)),
    )
    print(f"{name}: diam={out_diam:.1f}mm boundary_pts={len(loop_pts)} dist_from_expected={best_d:.1f}mm -> {disk_path.name}")

wall_path = OUT / "wall_open.stl"
wall.save(str(wall_path), binary=True)
print(f"wall: n_points={wall.n_points} n_cells={wall.n_cells} -> {wall_path.name}")

remaining_edges = wall.extract_feature_edges(
    boundary_edges=True, non_manifold_edges=False, manifold_edges=False, feature_edges=False
)
remaining_conn = remaining_edges.connectivity()
n_remaining_loops = len(np.unique(remaining_conn["RegionId"])) if remaining_edges.n_points else 0
print(f"remaining open boundary loops on wall: {n_remaining_loops} (expect 3)")

json.dump(patch_meta, open(OUT / "patch_meta.json", "w"), indent=2)
print("wrote patch_meta.json")
