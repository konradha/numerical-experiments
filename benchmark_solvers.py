from solvers import SineGordonIntegrator, kink, kink_start, L_2, L_infty, RSME
from solvers import analytical_kink_solution  


import matplotlib.pyplot as plt
import torch
from time import time
from dataclasses import dataclass, asdict
import numpy as np

import pickle

@dataclass
class MethodStatistics:
    method_name: str
    CFL_valid: bool

    num_trials: int
    samples_per_trial: int

    ti: float
    tf: float
    nt: int
    dt: float

    Lx_min: float
    Lx_max: float
    Ly_min: float
    Ly_max: float
    dx: float
    dy: float
    nx: int
    ny: int

    solution_u: np.ndarray # shape: (nt, nx, ny,)
    solution_v: np.ndarray # shape: (nt, nx, ny,)

    energies: np.ndarray      # shape: (nt,)

    RSME_errors: np.ndarray   # shape: (nt,) 
    L2_errors: np.ndarray     # shape: (nt,)
    Linfty_errors: np.ndarray # shape: (nt,)

    walltimes: np.ndarray # shape: (num_trials,)

    device: str
    nthreads: int


def collect_statistics(dev='cpu'):
    torch.set_num_threads(8)
    L = 7
    T = 10.
    nts = [int(T / (10 ** i)) for i in range(-3, -1)]
    N = [1 << i for i in range(5, 10)]
    #nts = [int(T / (10 ** i)) for i in range(-1, 0)]
    #N = [1 << i for i in range(5, 6)]
    ntrials = 5

    from itertools import product
    methods_data = {
        name: {
            'solutions': {(nt, n): {'u': None, 'v': None} for nt, n in product(nts, N)},
            'data': {attr: {(nt, n): None for nt, n in product(nts, N)} for attr in ['energy', 'RSME', 'L2', 'Linfty']},
            'times': None,
            'valid': {(nt, n): True for nt, n in product(nts, N)}
        } for name in ['Energy-conserving-1', 'stormer-verlet-pseudo', 'RK4']
    }

    err_map = {'L2': L_2, 'Linfty': L_infty, 'RSME': RSME}


    initial_u = kink
    initial_v = kink_start

    print("Running", len(nts) * len(N) * ntrials * (len(methods_data.keys()) + 1), "evolutions")

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

    for method in methods_data.keys(): 
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
                                  device='cpu')
                    t = -time()
                    solver.evolve()
                    t += time()
                    walltimes.append(t)
                walltimes = np.array(walltimes)
                methods_data[method]['times'] = walltimes


    for method in methods_data.keys(): 
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

                        walltimes=methods_data[method]['times'],
                        device=dev,
                        nthreads=torch.get_num_threads(),
                        )
                
                with open(f"benchmark_data/{method}_{nt}_{n}.pkl", 'wb') as f:
                    pickle.dump(data, f)




def calc_energy(u, v, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux ** 2
    uy2 = uy ** 2
    ut2 = ut ** 2
    cos = 2 * (1 - torch.cos(u[1:-1, 1:-1]))
    integrand = torch.sum((ux2 + uy2) + ut2 + (cos))
    return 0.5 * integrand * dx * dy


if __name__ == '__main__':
    collect_statistics()
    """
    torch.set_num_threads(8)
    L = 7
    T = 10.
    nts = [int(T / (10 ** i)) for i in range(-3, -1)]

    N = [1 << i for i in range(5, 10)]
 
    initial_u = kink
    initial_v = kink_start

    methods = {
        'Energy-conserving-1': None,
        'stormer-verlet-pseudo': None,
        #'gauss-legendre': None,
        'RK4': None,
    }

    # data[method][nt][n][u], data[method][nt][n][u]  
    data = {}
    ntries = 5
    for method in methods.keys():     
        data[method] = {} 
         
        for nt in nts:
            data[method][nt] = {n: {'u': None, 'v': None} for n in N}            

            for n in N:
                freq = nt  
                solver = SineGordonIntegrator(-L, L, -L, L, n,
                                  n, T, nt, initial_u, initial_v, step_method=method,
                                  boundary_conditions='special-kink-bc',
                                  snapshot_frequency=nt // 100, # always collect 100 samples
                                  c2 = 1,
                                  m = 1,
                                  enable_energy_control=False,
                                  device='cpu')

                assert solver.dt <= min(solver.dx, solver.dy) # CFL condition for unit wave velocity
                 
                #print(f"{method} {nt} {n} ({solver.dx=:.2f})")

                ##print("C=", torch.mean(solver.initial_v(solver.X, solver.Y)) * solver.dt / solver.dx)
                ## TODO check CFL conditions more carefully
                ##if torch.mean(solver.initial_v(solver.X, solver.Y)) * solver.dt / solver.dx < ???: continue

                t = -time()
                solver.evolve()
                t += time()

                np.save(f"benchmark_data/{}{}.npy")


                #print("walltime:", t)
                #data[method][nt][n]['u'] = solver.u.clone() 
                #data[method][nt][n]['v'] = solver.u.clone()
                #tn = solver.tn[::solver.snapshot_frequency]
                #analytical = torch.stack([
                #    analytical_kink_solution(solver.X, solver.Y, t) for t in tn]) 

                #plt.plot(tn.cpu().numpy(), L_infty(analytical, solver.u.clone()), label=f"{n=} {nt}")
        #plt.title(f"$L_\infty$-norm, {method}")
        #plt.legend()
        #plt.grid(True)
        #plt.yscale("log")
        #plt.show()

    """
