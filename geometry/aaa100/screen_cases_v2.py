"""Screen AAA-100 cases for a strong-flow-reversal candidate.

For each case: slice the aorta mesh along its centerline to get a diameter
profile, then score by (a) sac/neck diameter ratio, (b) neck angulation,
(c) iliac bifurcation tortuosity/angle. No CFD, no OpenFOAM/SU2 — pure
geometry via pyvista/vtk.
"""
import numpy as np
import pyvista as pv
from pathlib import Path
import csv

ROOT = Path(__file__).parent
MESH_DIR = ROOT / "meshes" / "meshes"
CL_DIR = ROOT / "centerlines" / "centerlines"


def tangents(points):
    t = np.gradient(points, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1
    return t / n


def diameter_profile(mesh, cl_points, tangs):
    diams = []
    for p, t in zip(cl_points, tangs):
        try:
            sliced = mesh.slice(normal=t, origin=p)
            if sliced.n_points < 3:
                diams.append(np.nan)
                continue
            bodies = sliced.connectivity(extraction_mode="closest", closest_point=p)
            filled = bodies.delaunay_2d()
            area = filled.compute_cell_sizes(length=False, area=True, volume=False)["Area"].sum()
            diams.append(2 * np.sqrt(area / np.pi))
        except Exception:
            diams.append(np.nan)
    return np.array(diams)


def tortuosity(points):
    arc = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    chord = np.linalg.norm(points[-1] - points[0])
    return arc / chord if chord > 0 else np.nan


def bend_angle(points):
    t = tangents(points)
    dots = np.clip(np.sum(t[:-1] * t[1:], axis=1), -1, 1)
    angles = np.degrees(np.arccos(dots))
    return np.nanmax(angles) if len(angles) else np.nan


def score_case(case_id):
    stl_path = MESH_DIR / f"{case_id}.stl"
    aorta_vtp = CL_DIR / case_id / "abdominal_aorta.vtp"
    il_l = CL_DIR / case_id / "iliac_left.vtp"
    il_r = CL_DIR / case_id / "iliac_right.vtp"
    if not (stl_path.exists() and aorta_vtp.exists()):
        return None

    mesh = pv.read(str(stl_path))
    cl = pv.read(str(aorta_vtp))
    pts = cl.points
    order = np.argsort(pts[:, 2])[::-1]  # top (thoracic, high Z) to bottom
    pts = pts[order]
    tangs = tangents(pts)

    diams = diameter_profile(mesh, pts, tangs)
    if np.all(np.isnan(diams)):
        return None

    n = len(diams)
    neck_region = diams[: max(3, n // 5)]
    neck_d = np.nanmedian(neck_region)
    max_d = np.nanmax(diams)
    diam_ratio = max_d / neck_d if neck_d else np.nan

    neck_angle = bend_angle(pts[: max(4, n // 3)])

    il_tort = np.nan
    bif_angle = np.nan
    if il_l.exists() and il_r.exists():
        try:
            # native point order is already sequential (uniform spacing);
            # do NOT re-sort by Z, that scrambles non-monotonic iliac paths.
            pl = pv.read(str(il_l)).points
            pr = pv.read(str(il_r)).points
            bif_ref = pts[-1]  # distal end of the (already proximal->distal) aorta centerline

            def orient_from_bifurcation(p):
                # flip so index 0 is the end nearest the aortic bifurcation
                if np.linalg.norm(p[0] - bif_ref) > np.linalg.norm(p[-1] - bif_ref):
                    return p[::-1]
                return p

            pl = orient_from_bifurcation(pl)
            pr = orient_from_bifurcation(pr)
            il_tort = np.nanmean([tortuosity(pl), tortuosity(pr)])
            tl = tangents(pl)[0]
            tr = tangents(pr)[0]
            cosang = np.clip(np.dot(tl, tr) / (np.linalg.norm(tl) * np.linalg.norm(tr)), -1, 1)
            bif_angle = np.degrees(np.arccos(cosang))
        except Exception:
            pass

    return dict(
        case=case_id,
        neck_d_mm=neck_d,
        max_d_mm=max_d,
        diam_ratio=diam_ratio,
        neck_angle_deg=neck_angle,
        iliac_tortuosity=il_tort,
        bifurcation_angle_deg=bif_angle,
    )


def main():
    cases = sorted(p.stem for p in MESH_DIR.glob("AAA*.stl"))
    rows = []
    for i, c in enumerate(cases):
        r = score_case(c)
        if r:
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(cases)} done")

    # normalize + combine
    def norm(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        return (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    n_ratio = norm("diam_ratio")
    n_neck = norm("neck_angle_deg")
    n_tort = norm("iliac_tortuosity")
    for r, a, b, c_ in zip(rows, n_ratio, n_neck, n_tort):
        r["score"] = float(np.nansum([a, b, c_]))

    rows.sort(key=lambda r: r["score"], reverse=True)

    out_csv = ROOT / "screen_results_v2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {out_csv}")
    print("\nTop 10 candidates:")
    print(f"{'case':8s} {'neck_d':>8s} {'max_d':>8s} {'ratio':>7s} {'neck_ang':>9s} {'il_tort':>8s} {'bif_ang':>8s} {'score':>7s}")
    for r in rows[:10]:
        print(f"{r['case']:8s} {r['neck_d_mm']:8.1f} {r['max_d_mm']:8.1f} {r['diam_ratio']:7.2f} "
              f"{r['neck_angle_deg']:9.1f} {r['iliac_tortuosity']:8.2f} {r['bifurcation_angle_deg']:8.1f} {r['score']:7.2f}")


if __name__ == "__main__":
    main()
