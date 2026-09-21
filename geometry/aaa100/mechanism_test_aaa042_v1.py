#!/usr/bin/env python3
"""
mechanism_test.py  --  does cancellation ill-conditioning explain biomarker fragility?

Run from inside the OpenFOAM case dir (needs wss_data.npz from extract_wss.py):
    python mechanism_test.py

WHAT THIS TESTS
---------------
Let  I_vec = (1/T) integral tau dt      (vector, near-cancelling)
     I_mag = (1/T) integral |tau| dt    (scalar, no cancellation  = TAWSS)
     R     = |I_vec| / I_mag
     OSI   = (1-R)/2          ->  R = 1 - 2*OSI
     RRT   = 1/(R * TAWSS)

Cancellation theory says the relative error in R is amplified by ~1/R, giving

     dOSI/OSI  ~  eps / (2*OSI)          <-- NOT discriminating (see below)
     dRRT/RRT  ~  eps / (1 - 2*OSI)      <-- DISCRIMINATING

The OSI prediction is degenerate with the trivial "small denominator" explanation,
because 1-R IS 2*OSI. So OSI cannot settle the question.

RRT can:
     cancellation theory  -> RRT amplification grows as OSI -> 0.5   (HIGH OSI)
     small-denominator    -> RRT amplification grows as OSI -> 0     (LOW  OSI)
Opposite regimes. That is the test.

Also runs the detune-vs-smooth contrast: a systematic clock-rate error and a
symmetric temporal smoothing, matched on field-level perturbation size.
"""

import numpy as np

RHO_BLOOD = 1060.0
T_CYCLE   = 0.9
EPS       = 1e-30


# ----------------------------------------------------------------- biomarkers
def biomarkers(w, dt, T=T_CYCLE):
    """w: (nt, nc, 3) WSS in Pa. Returns TAWSS, OSI, RRT, R."""
    mag    = np.linalg.norm(w, axis=2)
    tawss  = mag.sum(axis=0) * dt / T
    ivec   = w.sum(axis=0) * dt / T
    imag   = np.linalg.norm(ivec, axis=1)
    R      = imag / np.maximum(tawss, EPS)
    R      = np.clip(R, 0.0, 1.0)
    osi    = 0.5 * (1.0 - R)
    rrt    = 1.0 / np.maximum(R * tawss, EPS)
    return tawss, osi, rrt, R


# --------------------------------------------------------------- perturbations
def detune(w_full, rho, nt):
    """Clock-rate error WITHOUT wraparound: resample the multi-cycle series at
    rate (1+rho) and take a fixed-duration window of nt samples. The window then
    spans a non-integer number of predicted cycles -> partial-cycle net integral."""
    N = w_full.shape[0]
    idx = np.arange(nt) * (1.0 + rho)
    if idx[-1] >= N - 1:
        raise ValueError('need more cycles for rho=%g' % rho)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, N - 1)
    f  = (idx - lo)[:, None, None]
    return w_full[lo] * (1.0 - f) + w_full[hi] * f


def smooth(w, sigma_steps):
    """Symmetric temporal smoothing (circular Gaussian) - the 'resolution' style
    perturbation. Same class as downsampling: symmetric, non-accumulating."""
    nt = w.shape[0]
    k  = np.arange(nt)
    k  = np.minimum(k, nt - k)
    g  = np.exp(-0.5 * (k / max(sigma_steps, 1e-9)) ** 2)
    g /= g.sum()
    G  = np.fft.rfft(g)
    W  = np.fft.rfft(w, axis=0)
    return np.fft.irfft(W * G[:, None, None], n=nt, axis=0)


def field_eps(w0, w1):
    """Relative field-level perturbation: the metric an ML paper would report."""
    return np.linalg.norm(w1 - w0) / max(np.linalg.norm(w0), EPS)


# ---------------------------------------------------------------------- driver
def main():
    d     = np.load('wss_data_aaa042_v4_2cyc.npz')
    wss   = d['wss'] * RHO_BLOOD          # kinematic -> Pa
    t     = d['times']
    areas = d['areas']
    dt    = float(np.round(t[1] - t[0], 12))

    m = (t > t[-1] - T_CYCLE + 1e-9)       # final cycle
    w0 = wss[m]
    nt = w0.shape[0]
    print(f'final cycle: {nt} snapshots, dt={dt}, T={T_CYCLE}')

    tawss0, osi0, rrt0, R0 = biomarkers(w0, dt)
    print(f'baseline  area-avg TAWSS {np.average(tawss0, weights=areas):.4g} Pa  '
          f'OSI {np.average(osi0, weights=areas):.4g}  '
          f'RRT {np.average(rrt0, weights=areas):.4g}')
    print(f'R range {R0.min():.4f} - {R0.max():.4f} ; '
          f'OSI range {osi0.min():.3g} - {osi0.max():.3g}')

    # ---- 1. detune vs smooth, matched on field-level eps -------------------
    print('\n--- perturbation comparison (matched field eps) ---')
    rows = []
    for rho in (0.01, 0.02, 0.03, 0.05):
        wd  = detune(wss[-2*nt:], rho, nt)
        e   = field_eps(w0, wd)
        # find smoothing sigma giving the same field eps
        lo, hi = 1e-3, float(nt)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if field_eps(w0, smooth(w0, mid)) < e:
                lo = mid
            else:
                hi = mid
        ws = smooth(w0, 0.5 * (lo + hi))

        out = {}
        for tag, wp in (('detune', wd), ('smooth', ws)):
            ta, os_, rr, _ = biomarkers(wp, dt)
            out[tag] = [
                100 * np.average(np.abs(ta - tawss0) / np.maximum(tawss0, EPS), weights=areas),
                100 * np.average(np.abs(os_ - osi0) / np.maximum(osi0, EPS),   weights=areas),
                100 * np.average(np.abs(rr - rrt0) / np.maximum(rrt0, EPS),    weights=areas),
            ]
        rows.append((rho, e, out))
        print(f'rho={rho:.3f}  field eps={100*e:5.2f}%   '
              f'DETUNE TAWSS {out["detune"][0]:6.2f}% OSI {out["detune"][1]:8.2f}% RRT {out["detune"][2]:8.2f}%  |  '
              f'SMOOTH TAWSS {out["smooth"][0]:6.2f}% OSI {out["smooth"][1]:8.2f}% RRT {out["smooth"][2]:8.2f}%')

    # ---- 2. THE MECHANISM TEST --------------------------------------------
    print('\n--- mechanism test: RRT amplification vs local OSI ---')
    rho = 0.02
    wd  = detune(wss[-2*nt:], rho, nt)
    e   = field_eps(w0, wd)
    ta, os_, rr, R1 = biomarkers(wd, dt)

    amp_rrt = (np.abs(rr - rrt0) / np.maximum(rrt0, EPS)) / e
    amp_osi = (np.abs(os_ - osi0) / np.maximum(osi0, EPS)) / e

    # bin by baseline OSI, area-weighted median amplification per bin
    good = np.isfinite(amp_rrt) & np.isfinite(osi0) & (osi0 > 0)
    edges = np.quantile(osi0[good], np.linspace(0, 1, 13))
    edges = np.unique(edges)
    print(f'{"OSI bin":>22} {"n":>6} {"med RRT amp":>12} {"pred 1/(1-2OSI)":>16} {"pred 1/OSI(small-den)":>22}')
    cx, cy = [], []
    for i in range(len(edges) - 1):
        sel = good & (osi0 >= edges[i]) & (osi0 < edges[i + 1])
        if sel.sum() < 20:
            continue
        oc = np.median(osi0[sel])
        a  = np.median(amp_rrt[sel])
        cx.append(oc); cy.append(a)
        print(f'[{edges[i]:.5f},{edges[i+1]:.5f}) {sel.sum():6d} {a:12.3f} '
              f'{1.0/max(1-2*oc,1e-6):16.3f} {1.0/max(oc,1e-6):22.3f}')

    cx, cy = np.array(cx), np.array(cy)
    if len(cx) > 3:
        # does RRT amplification INCREASE with OSI (cancellation) or DECREASE (small-denom)?
        s = np.polyfit(np.log(cx), np.log(cy), 1)[0]
        print(f'\nlog-log slope of RRT amplification vs OSI: {s:+.3f}')
        print('  cancellation theory predicts slope  > 0  (worse at HIGH OSI)')
        print('  small-denominator artefact predicts slope < 0  (worse at LOW OSI)')
        print(f'  VERDICT: {"CANCELLATION supported" if s > 0.2 else "SMALL-DENOMINATOR supported" if s < -0.2 else "INCONCLUSIVE - regimes not separated"}')

    np.savez_compressed('mechanism_test_aaa042.npz',
                        osi0=osi0, rrt0=rrt0, tawss0=tawss0, R0=R0,
                        amp_rrt=amp_rrt, amp_osi=amp_osi, areas=areas, eps=e, rho=rho)

    # ---- 3. plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        sub = good & (amp_rrt > 0)
        sub = np.where(sub)[0]
        if sub.size > 20000:
            sub = np.random.default_rng(0).choice(sub, 20000, replace=False)
        ax[0].loglog(osi0[sub], amp_rrt[sub], '.', ms=1.5, alpha=.25, color='#446')
        if len(cx) > 1:
            ax[0].loglog(cx, cy, 'o-', color='#c33', label='binned median')
            xs = np.linspace(cx.min(), min(cx.max(), 0.49), 200)
            ax[0].loglog(xs, cy[0] * (1/(1-2*xs)) / (1/(1-2*cx[0])), '--',
                         color='#2a2', label=r'cancellation $\propto 1/(1-2\,OSI)$')
            ax[0].loglog(xs, cy[0] * (cx[0]/xs), ':', color='#28c',
                         label=r'small-denominator $\propto 1/OSI$')
        ax[0].set_xlabel('baseline OSI'); ax[0].set_ylabel('RRT amplification  (rel.err / field eps)')
        ax[0].set_title('mechanism test'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which='both')

        lbl = [f'{r[0]*100:.0f}%' for r in rows]
        xg = np.arange(len(rows)); wdt = 0.38
        ax[1].bar(xg - wdt/2, [r[2]['detune'][2] for r in rows], wdt, label='detune (systematic)', color='#c44')
        ax[1].bar(xg + wdt/2, [r[2]['smooth'][2] for r in rows], wdt, label='smooth (symmetric)', color='#48a')
        ax[1].set_xticks(xg); ax[1].set_xticklabels(lbl); ax[1].set_yscale('log')
        ax[1].set_xlabel('clock-rate detuning'); ax[1].set_ylabel('RRT error [%]')
        ax[1].set_title('same field eps, opposite consequence'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='y')
        plt.tight_layout(); plt.savefig('mechanism_test_aaa042.png', dpi=140)
        print('\nwrote mechanism_test_aaa042.png and mechanism_test_aaa042.npz')
    except Exception as exc:
        print('plot skipped:', exc)


if __name__ == '__main__':
    main()
