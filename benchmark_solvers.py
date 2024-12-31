from solvers import SineGordonIntegrator, kink, kink_start, L_2, L_infty, RSME
import matplotlib.pyplot as plt
import torch

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
    L = 5
    T = 10.
    nts = [int(T / (10 ** i)) for i in range(-2, 0)]

    N = [1 << i for i in range(5, 7)]
 
    initial_u = kink
    initial_v = kink_start

    methods = {
        'Energy-conserving-1': None,
        #'stormer-verlet-pseudo': None,
        #'gauss-legendre': None,
        #'RK4': None,
    }

    # data[method][nt][n][u], data[method][nt][n][u]  
    data = {}
    for method in methods.keys():     
        data[method] = {} 
        for nt in nts:
            data[method][nt] = {n: {'u': None, 'v': None} for n in N}
            
            for n in N:
                print(f"{method} {nt} {n}")
                solver = SineGordonIntegrator(-L, L, -L, L, n,
                                  n, T, nt, initial_u, initial_v, step_method=method,
                                  boundary_conditions='special-kink-bc',
                                  snapshot_frequency=10,
                                  c2 = 1,
                                  m = 1,
                                  enable_energy_control=False,
                                  device='cpu')
                solver.evolve()
                data[method][nt][n]['u'] = solver.u.clone() 
                data[method][nt][n]['v'] = solver.u.clone()

