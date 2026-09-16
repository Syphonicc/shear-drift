import numpy as np
d = np.load('wss_data.npz')
wss, t, areas = d['wss'], d['times'], d['areas']
rho = 1060.0

# final cycle: t in (1.8, 2.7], 40 snapshots
m = (t > 1.8) & (t <= 2.7 + 1e-9)
w = wss[m] * rho          # kinematic -> Pa
T = 0.9
dt = 0.0225

mag = np.linalg.norm(w, axis=2)              # (40, nc)
tawss = mag.sum(axis=0) * dt / T             # Pa
netvec = w.sum(axis=0) * dt / T              # (nc, 3)
netmag = np.linalg.norm(netvec, axis=1)

osi = 0.5 * (1.0 - netmag / np.maximum(tawss, 1e-30))
rrt = 1.0 / np.maximum((1.0 - 2.0*osi) * tawss, 1e-30)

np.savez_compressed('biomarkers_cycle3.npz',
                    tawss=tawss, osi=osi, rrt=rrt, areas=areas)

aw = areas / areas.sum()
for n, v in [('TAWSS (Pa)', tawss), ('OSI', osi), ('RRT (1/Pa)', rrt)]:
    print(f'{n:12s} area-avg {np.sum(v*aw):10.4g}  min {v.min():10.4g}  max {v.max():10.4g}')
print('snapshots used:', m.sum(), ' OSI>0.1 area frac: %.3f' % aw[osi>0.1].sum())
