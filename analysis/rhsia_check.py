"""
External check of the cancellation law against RHSIA (arXiv 2601.19876v2, Table V).
RHSIA reports mean-based rL2 (%) over IA surfaces:  TAWSS 13.54, OSI 68.35, RRT 17.31
i.e. ratios  OSI/TAWSS = 5.05,  RRT/TAWSS = 1.28.

Two independent predictions, both using RHSIA's own metric definition
    rL2(X) = ||X_pred - X_true||_2 / ||X_true||_2   (over faces)
(A) Law-based: per-face relative errors from c*(1/R-1)^p with the fitted
    Womersley constants, applied to the cerebral-wall OSI distribution as a
    proxy for an IA surface, with eps chosen so that rL2(TAWSS) = 13.54 %.
(B) Direct: drift-injected fields on the real cerebral CFD (data/fields.bin),
    rL2 computed exactly as RHSIA does, at each injected rho.
"""
import json, numpy as np

meta = json.load(open("data/meta.json"))
lv = meta["levels"]; F = meta["n_faces"]
fields = np.fromfile("data/fields.bin", np.float32).reshape(len(lv), 3, F).astype(np.float64)
TAWSS0, OSI0, RRT0 = fields[0]
R0 = 1 - 2*OSI0
c, p = 0.194, 0.641
RH = dict(tawss=13.54, osi=68.35, rrt=17.31)

def rl2(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)

print("cerebral wall OSI distribution: median %.2e  mean %.2e  p95 %.3f  max %.3f  frac<0.01 %.3f"
      % (np.median(OSI0), OSI0.mean(), np.percentile(OSI0,95), OSI0.max(), (OSI0<0.01).mean()))

# ---------- (A) law-based ----------
x = np.clip(1/R0 - 1, 1e-12, None)           # = 2*OSI/(1-2*OSI)
dR_over_R = c * x**p                          # per unit eps
dOSI_over_OSI = c * x**(p-1)                  # per unit eps (corollary)
for eps in (0.1354,):
    tawss_rel = eps                           # TAWSS error ~ eps (no cancellation)
    rrt_rel   = np.sqrt(tawss_rel**2 + (eps*dR_over_R)**2)   # independent components
    rrt_rel_lin = tawss_rel + eps*dR_over_R                   # worst case, additive
    osi_rel   = eps*dOSI_over_OSI
    pred = dict(
        tawss = 100*rl2(TAWSS0*(1+tawss_rel), TAWSS0),
        osi   = 100*rl2(OSI0*(1+osi_rel), OSI0),
        rrt   = 100*rl2(RRT0*(1+rrt_rel), RRT0),
        rrt_add = 100*rl2(RRT0*(1+rrt_rel_lin), RRT0),
    )
    print("\n(A) law prediction, eps=%.3f" % eps)
    for k in ("tawss","osi","rrt","rrt_add"):
        print("   rL2 %-8s %6.2f %%" % (k, pred[k]))
    print("   ratio OSI/TAWSS  pred %.2f   RHSIA %.2f" % (pred["osi"]/pred["tawss"], RH["osi"]/RH["tawss"]))
    print("   ratio RRT/TAWSS  pred %.2f   RHSIA %.2f" % (pred["rrt"]/pred["tawss"], RH["rrt"]/RH["tawss"]))
    # first-order theory (p=1) for contrast
    osi_p1 = 100*rl2(OSI0*(1+eps), OSI0)
    print("   [p=1 first-order theory would give OSI/TAWSS = %.2f]" % (osi_p1/pred["tawss"]))

# ---------- (B) direct drift injection on real CFD ----------
print("\n(B) drift-injected cerebral CFD, RHSIA-style rL2 (%)")
print("   rho    field  TAWSS    OSI    RRT   OSI/TAWSS RRT/TAWSS")
for i, rho in enumerate(lv[1:], 1):
    T, O, Rr = fields[i]
    fl2 = 100*meta["stats"][i]["field_l2"]
    e = [100*rl2(T,TAWSS0), 100*rl2(O,OSI0), 100*rl2(Rr,RRT0)]
    print("   %.3f  %5.1f  %5.2f  %6.2f  %5.2f   %5.2f     %5.2f" % (rho, fl2, *e, e[1]/e[0], e[2]/e[0]))
print("   RHSIA:        13.54  68.35  17.31    5.05      1.28")
