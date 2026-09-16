#!/usr/bin/env python3
"""build_web.py -- pack data/ into web/explorer_template.html -> web/shear_drift_explorer.html
Run from hackathon/:  python analysis/build_web.py"""
import base64, gzip, json
import numpy as np

m = json.load(open("data/meta.json")); P, F = m["n_points"], m["n_faces"]
raw = np.fromfile("data/mesh.bin", np.uint8)
pts = np.frombuffer(raw[:P * 12].tobytes(), np.float32)
tri = np.frombuffer(raw[P * 12:].tobytes(), np.uint32).astype(np.uint16)   # 16749 pts < 65536
L = len(m["levels"])
fl = np.fromfile("data/fields.bin", np.float32).reshape(L, 3, F)
ex = np.fromfile("data/extra.bin", np.float32).reshape(2, F)
base = fl[0]
den = np.where(np.abs(base) > 1e-9, base, 1)
rel = np.clip(np.round((fl[1:] - base) / den / 1e-5), -32767, 32767).astype("<i2")  # 1e-5 steps
blob = b"".join(a.tobytes() for a in (pts.astype("<f4"), tri.astype("<u2"), base.astype("<f4"),
                                      rel, ex[0].astype("<f4"), ex[1].astype(np.uint8)))
b64 = base64.b64encode(gzip.compress(blob, 9, mtime=0)).decode()
meta = {k: m[k] for k in ("n_points", "n_faces", "levels", "baseline", "stats", "law", "sweep")}
html = open("web/explorer_template.html").read()
html = html.replace("__BLOB__", b64).replace("__META__", json.dumps(meta, separators=(",", ":")))
open("web/shear_drift_explorer.html", "w").write(html)
print(f"web/shear_drift_explorer.html  {len(html)/1e6:.1f} MB")
