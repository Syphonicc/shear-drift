#!/usr/bin/env python3
"""build_web_v5.py -- v4 + OSI-distribution-aware fingerprint bands (law_vs_errortype.py) + researcher quotes (POD+LSTM rollout, phase check,
error-type fingerprint vs RHSIA) -> web/shear_drift_explorer_v5.html.
Run from hackathon/:  python analysis/build_web_v3.py"""
import base64, gzip, json
import numpy as np
import importlib.util, sys

# reuse pack_geometry from v2 without executing its module body
src = open("analysis/build_web_v2.py").read().split("\ngeo = {}")[0]
ns = {}; exec(compile(src, "build_web_v2.py", "exec"), ns)
pack_geometry = ns["pack_geometry"]

b64_cer, meta_cer, m_cer = pack_geometry("data")
b64_aaa, meta_aaa, m_aaa = pack_geometry("data/aaa042")
law, sweep = m_cer["law"], m_cer["sweep"]

# ---------- AI panel ----------
S6 = json.load(open("data/pod_lstm_summary_hr.json"))
P6 = json.load(open("data/phase_correct_hr.json"))
S40 = json.load(open("runs/long40/data/pod_lstm_summary_hr.json"))
P40 = json.load(open("runs/long40/data/phase_correct_hr.json"))
R6 = np.load("data/pod_lstm_results_hr.npz")
roll, truth = R6["roll"], R6["truth"]            # (240,10), (40,10)
Tb = truth.shape[0]; nb = roll.shape[0] // Tb
mode = 0
sc = float(np.abs(truth[:, mode]).max())
def r3(a): return [round(float(v), 3) for v in a]
ai = {
    "T": Tb, "beats": nb,
    "one_step_rmse": S6["one_step_rmse"],
    "energy": S6["energy"], "r": S6["r"],
    "roll_m1": r3(roll[:, mode] / sc),
    "truth_m1": r3(np.tile(truth[:, mode], nb) / sc),
    "rho_self6": P6["rho_self"], "rho_truth6": P6["rho_truth"], "rho_after6": P6["rho_after"],
    "rho_self40": P40["rho_self"], "rho_after40": P40["rho_after"],
    "beats40": len(P40["cycles"]),
    "cycles": [{
        "l2": c["raw"]["l2"], "l2a": c["raw"]["l2_aligned"], "lag": c["raw"]["lag"],
        "lag_cor": c["cor"]["lag"],
        "wall": [c["raw"]["wall"]["tawss"], c["raw"]["wall"]["rrt"], c["cor"]["wall"]["rrt"]],
        "sac": [c["raw"]["sac"]["tawss"], c["raw"]["sac"]["rrt"], c["cor"]["sac"]["rrt"]],
        "osc": [c["raw"]["osi>0.1"]["tawss"], c["raw"]["osi>0.1"]["rrt"], c["cor"]["osi>0.1"]["rrt"]],
    } for c in P6["cycles"]],
    # error-type fingerprint on the cerebral wall, RHSIA-style mean rL2 (%),
    # from analysis/rhsia_check.py / _v2 / _v3 (seed-averaged where applicable)
    "fingerprint": [
        {"type": "Clock drift (rollout runs fast)", "eps": 10.8, "tawss": 2.5, "osi": 3.1, "rrt": 3.5, "fix": "re-time"},
        {"type": "Time-white noise (jitter per frame)", "eps": 13.5, "tawss": 2.3, "osi": 68.7, "rrt": 2.5, "fix": "smooth"},
        {"type": "Frozen per-node bias (wrong everywhere, every frame)", "eps": 14.5, "tawss": 11.0, "osi": 66.0, "rrt": 16.5, "fix": "better model"},
    ],
    "rhsia": {"tawss": 13.54, "osi": 68.35, "rrt": 17.31, "ref": "arXiv 2601.19876v2, Table V"},
    # analysis/law_vs_errortype.py, cerebral wall, top cancellation bin [1,10), 9 faces; median excess RRT error (%)
    "lawtype": [
        {"type": "Clock 5% faster, integrated over the surrogate's own period", "eps": 0.0, "excess": 0.00, "law": None, "note": "nothing changes: all three biomarkers are invariant to a pure time stretch"},
        {"type": "Clock 5% faster, period unknown, integrated over the patient's cycle (this page's slider)", "eps": 10.8, "excess": 2.37, "law": 2.46, "note": "window mismatch; removable once the period is estimated from the rollout"},
        {"type": "Persistent spatial bias, same field error", "eps": 10.8, "excess": 12.08, "law": 2.46, "note": "cancellation punishes bias ~5× harder than the fitted law says; this is the error real surrogates have", "hi": True},
        {"type": "Frame-to-frame noise, same field error", "eps": 10.8, "excess": 1.93, "law": 2.46, "note": "noise largely cancels in the net shear; OSI takes the hit instead"},
    ],
    # analysis/referee_checks.py: OSI/TAWSS error ratio by error type, on nested
    # subsets of the cerebral wall of increasing median OSI. The bands are NOT universal.
    "bands": [
        {"name": "whole wall",   "n": 33406, "osi": 0.0007, "window": 1.22, "noise": 27.4, "bias": 4.78},
        {"name": "OSI > 0.002",  "n": 7234,  "osi": 0.0048, "window": 1.13, "noise": 16.5, "bias": 3.94},
        {"name": "OSI > 0.01",   "n": 2146,  "osi": 0.0212, "window": 1.14, "noise": 10.5, "bias": 3.56},
        {"name": "OSI > 0.03",   "n": 635,   "osi": 0.0503, "window": 1.27, "noise": 5.00, "bias": 2.25},
    ],
    "quotes": [
        {"text": "You were right about the RRT error of my surrogate model, they present spatial error and usually persistent.", "who": "Wenhao Ding, Imperial College London", "on": "the fingerprint verdict for RHSIA"},
        {"text": "Yes, the heart beat might be faster than the actual patient, but what's the error if you just use that faster cardiac cycle? And from the AI model the cardiac cycle is usually something the user can tell and control.", "who": "Wenhao Ding, Imperial College London", "on": "the clock-error slider — he was right; the table above and the slider caption were corrected in response"},
        {"text": "From my experience of working with physicians, they'd like to see TAWSS and RRT, but the former is easier for them to accept. Still, there isn't a conclusion on which biomarker matters the most.", "who": "Wenhao Ding, Imperial College London", "on": "which biomarker clinicians use"},
    ],
}

threejs = open("web/vendor/three.min.js").read()
html = open("web/explorer_template_v5.html").read()
html = html.replace("__THREEJS__", threejs)
html = html.replace("__BLOB_CEREBRAL__", b64_cer).replace("__META_CEREBRAL__", json.dumps(meta_cer, separators=(",", ":")))
html = html.replace("__BLOB_AAA042__", b64_aaa).replace("__META_AAA042__", json.dumps(meta_aaa, separators=(",", ":")))
html = html.replace("__LAW__", json.dumps(law, separators=(",", ":")))
html = html.replace("__SWEEP__", json.dumps(sweep, separators=(",", ":")))
html = html.replace("__AI__", json.dumps(ai, separators=(",", ":")))
open("web/shear_drift_explorer_v5.html", "w").write(html)
print(f"web/shear_drift_explorer_v5.html  {len(html)/1e6:.1f} MB")
