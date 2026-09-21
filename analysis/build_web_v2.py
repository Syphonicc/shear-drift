#!/usr/bin/env python3
"""build_web_v2.py -- pack data/ (cerebral) + data/aaa042/ into
web/explorer_template_v2.html -> web/shear_drift_explorer_v2.html.
Two geometries, one shared law/sweep (from the cerebral meta -- the law is a
geometry-independent physical claim, tested via Womersley + injected drift on
both walls). Run from hackathon/:  python analysis/build_web_v2.py"""
import base64, gzip, json
import numpy as np


def pack_geometry(root):
    m = json.load(open(f"{root}/meta.json"))
    P, F = m["n_points"], m["n_faces"]
    raw = np.fromfile(f"{root}/mesh.bin", np.uint8)
    pts = np.frombuffer(raw[:P * 12].tobytes(), np.float32)
    tri = np.frombuffer(raw[P * 12:].tobytes(), np.uint32).astype(np.uint32)
    L = len(m["levels"])
    fl = np.fromfile(f"{root}/fields.bin", np.float32).reshape(L, 3, F)
    # extra.bin layout differs by export script: export_demo.py (cerebral) writes
    # float32[F] X + float32[F] sac (2*F*4 bytes); export_demo_aaa042_v1.py writes
    # float32[F] X + uint8[F] sac (F*4+F bytes). Detect by file size.
    exraw = np.fromfile(f"{root}/extra.bin", np.uint8)
    X = np.frombuffer(exraw[:F * 4].tobytes(), np.float32)
    if len(exraw) == 2 * F * 4:
        sac = (np.frombuffer(exraw[F * 4:].tobytes(), np.float32) > 0.5).astype(np.uint8)
    elif len(exraw) == F * 4 + F:
        sac = np.frombuffer(exraw[F * 4:F * 4 + F].tobytes(), np.uint8)
    else:
        raise ValueError(f"{root}/extra.bin: unrecognized size {len(exraw)} for F={F}")
    base = fl[0]
    den = np.where(np.abs(base) > 1e-9, base, 1)
    rel = np.clip(np.round((fl[1:] - base) / den / 1e-5), -32767, 32767).astype("<i2")
    idx_dtype = "<u2" if P < 65536 else "<u4"
    blob = b"".join(a.tobytes() for a in (
        pts.astype("<f4"), tri.astype(idx_dtype), base.astype("<f4"),
        rel, X.astype("<f4"), sac.astype(np.uint8)))
    b64 = base64.b64encode(gzip.compress(blob, 9, mtime=0)).decode()
    meta = {k: m[k] for k in ("n_points", "n_faces", "levels", "baseline", "stats")}
    meta["idx32"] = P >= 65536
    return b64, meta, m


geo = {}
b64_cer, meta_cer, m_cer = pack_geometry("data")
b64_aaa, meta_aaa, m_aaa = pack_geometry("data/aaa042")
law = m_cer["law"]
sweep = m_cer["sweep"]

threejs = open("web/vendor/three.min.js").read()
html = open("web/explorer_template_v2.html").read()
html = html.replace("__THREEJS__", threejs)
html = html.replace("__BLOB_CEREBRAL__", b64_cer).replace("__META_CEREBRAL__", json.dumps(meta_cer, separators=(",", ":")))
html = html.replace("__BLOB_AAA042__", b64_aaa).replace("__META_AAA042__", json.dumps(meta_aaa, separators=(",", ":")))
html = html.replace("__LAW__", json.dumps(law, separators=(",", ":")))
html = html.replace("__SWEEP__", json.dumps(sweep, separators=(",", ":")))
open("web/shear_drift_explorer_v2.html", "w").write(html)
print(f"web/shear_drift_explorer_v2.html  {len(html)/1e6:.1f} MB")
