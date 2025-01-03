from benchmark_util import MethodStatistics

from solvers import SineGordonIntegrator, kink, kink_start, L_2, L_infty, RSME
from solvers import analytical_kink_solution  

import torch
from time import time
import numpy as np
import pickle
import json
import os
import shutil
from itertools import product

def collect_statistics(base_path=None, dev='cpu'):
    torch.set_num_threads(8)
    L = 7
    T = 10.

    nts = [1000, 2500, 5000, 10000]
    N = [50, 100, 200, 500,]

    ## testing
    #nts = [1000]
    #N = [50]
    
    ntrials = 5
 
    methods_data = {
        name: {
            'solutions': {(nt, n): {'u': None, 'v': None} for nt, n in product(nts, N)},
            'data': {attr: {(nt, n): None for nt, n in product(nts, N)} for attr in ['energy', 'RSME', 'L2', 'Linfty']},
            'times': {(nt, n): None for nt, n in product(nts, N)},
            'valid': {(nt, n): True for nt, n in product(nts, N)},
        } for name in ['Energy-conserving-1', 'stormer-verlet-pseudo', 'RK4']
    }

    err_map = {'L2': L_2, 'Linfty': L_infty, 'RSME': RSME}
    initial_u = kink
    initial_v = kink_start

    print("Running", len(nts) * len(N) * ntrials * (len(methods_data.keys()) + 1), "evolutions")

    # data for the time series collected
    single = sum([n ** 2 * nt for nt, n in product(nts, N)])
    # statistics: errors, energy
    s = len(methods_data.keys()) * (2 * single + 4 * sum([nt for nt, n in product(nts, N)])) 
    print("writing to disk >=", s / (1<<30), "GB")
    

    fpaths = []
    fnames = []
    base_path = os.getcwd() + f"/benchmark_data_{dev}" if base_path is None else base_path

    if os.path.exists(base_path):
        print("Found existing benchmark, deleting")
        shutil.rmtree(base_path)
    os.makedirs(base_path)


    data_collection = []
    for method in methods_data.keys(): 
        for nt in nts:
            for n in N:
                solver = SineGordonIntegrator(-L, L, -L, L, n,
                                  n, T, nt, initial_u, initial_v, step_method=method,
                                  boundary_conditions='special-kink-bc',
                                  snapshot_frequency=nt // 100, # always collect 100 samples
                                  c2 = 1,
                                  m = 1,
                                  enable_energy_control=False,
                                  device=dev)
                if solver.dt > solver.dx:
                    methods_data[method]['valid'][(nt, n)] = False
                    print(method, "for", f"{nt=} {n=} invalid")
                    continue

                solver.evolve()
                methods_data[method]['solutions'][(nt, n)]['u'] = solver.u.clone().cpu().numpy()
                methods_data[method]['solutions'][(nt, n)]['v'] = solver.v.clone().cpu().numpy()

                E = torch.stack([solver.energy(u, v) for u, v in zip(solver.u, solver.v)])
                methods_data[method]['data']['energy'][(nt, n)] = E.clone().cpu().numpy()

                analytical = torch.stack([analytical_kink_solution(solver.X, solver.Y, t) for t in
                    solver.tn[::solver.snapshot_frequency]])
 
                for err, err_fun in err_map.items():
                    methods_data[method]['data'][err][(nt, n)] = err_fun(analytical, solver.u.clone()).cpu().numpy() 

 
        for nt in nts:
            for n in N:
                if not methods_data[method]['valid'][(nt, n)]: continue
                walltimes = []
                for trial in range(ntrials):
                    solver = SineGordonIntegrator(-L, L, -L, L, n,
                                  n, T, nt, initial_u, initial_v, step_method=method,
                                  boundary_conditions='special-kink-bc',
                                  snapshot_frequency=nt // 100, # always collect 100 samples
                                  c2 = 1,
                                  m = 1,
                                  enable_energy_control=False,
                                  device=dev)
                    t = -time()
                    solver.evolve()
                    t += time()
                    walltimes.append(t)
                walltimes = np.array(walltimes)
                methods_data[method]['times'][(nt, n)] = walltimes

        
        for nt in nts:
            for n in N:
                data = MethodStatistics(
                        method_name=method,
                        CFL_valid=methods_data[method]['valid'][(nt, n)],
                        num_trials=ntrials,
                        samples_per_trial=100, # hardcoded above in snapshot frequency
                        ti=0,
                        tf=T,
                        nt=nt,
                        dt=T/nt,
                        Lx_min=-L,
                        Lx_max=L,
                        Ly_min=-L,
                        Ly_max=L,
                        dx=2 * L / (n + 1),
                        dy=2 * L / (n + 1),
                        nx=n+2,
                        ny=n+2,

                        solution_u=methods_data[method]['solutions'][(nt, n)]['u'],
                        solution_v=methods_data[method]['solutions'][(nt, n)]['u'],

                        energies=methods_data[method]['data']['energy'][(nt, n)],

                        RSME_errors=methods_data[method]['data']['RSME'][(nt, n)], 
                        L2_errors=methods_data[method]['data']['L2'][(nt, n)],
                        Linfty_errors=methods_data[method]['data']['Linfty'][(nt, n)],

                        walltimes=methods_data[method]['times'][(nt, n)],
                        device=dev,
                        nthreads=torch.get_num_threads(),
                        )

                fname = f"{method}_{nt}_{n}_{dev}.pkl"
                fpath = f"{base_path}/{fname}"
                fpaths.append(fpath)
                fnames.append(fname)
                with open(fpath, 'wb') as f:
                    pickle.dump(data, f)

                data_collection.append(
                                {
                                "absolute-path": fpath,
                                "fname": fname,
                                "method": method,
                                "nt": nt,
                                "n": n,
                                "dev": dev,
                                })
    stat_description = {
        'time_steps': list(nts),
        'grid_sizes': list(N),
        'methods': list(methods_data.keys()),
        't0': 0.,
        'tf': T,
        'L': L,
        'equation-type': 'sine-Gordon undamped',
        'dev': dev,
        'base_path': base_path,
    }
 
    data_description = {
        'description': stat_description,
        'data': data_collection,
    }

    with open(f"{base_path}/benchmark_description.json", 'w') as f:
        json.dump(data_description, f, indent=2)


if __name__ == '__main__':
    collect_statistics(dev='cuda')
