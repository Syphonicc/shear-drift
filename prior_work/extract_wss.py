import numpy as np, pyvista as pv, glob, re

files = sorted(glob.glob('VTK/results_v2_*/boundary/wall.vtp'),
               key=lambda f: int(re.search(r'results_v2_(\d+)', f).group(1)))
print(f'found {len(files)} files')

m0 = pv.read(files[0])
nc = m0.n_cells
sized = m0.compute_cell_sizes(length=False, area=True, volume=False)
areas = np.asarray(sized.cell_data['Area'])
centres = np.asarray(m0.cell_centers().points)

wss = np.empty((len(files), nc, 3), dtype=np.float64)
for i, f in enumerate(files):
    wss[i] = pv.read(f).cell_data['wallShearStress']

times = np.arange(1, len(files)+1) * 0.0225
np.savez_compressed('wss_data.npz', wss=wss, areas=areas,
                    centres=centres, times=times)
print('wss', wss.shape, '| area total %.4e m^2' % areas.sum())
print('t:', times[0], '->', times[-1])
