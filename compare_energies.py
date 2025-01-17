import numpy as np
import matplotlib.pyplot as plt

from sys import argv

def energy(u, v, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    ut = v[1:-1, 1:-1]
    ux2 = .5 * ux ** 2
    uy2 = .5 * uy ** 2
    ut2 = .5 * ut ** 2
    cos = (1 - np.cos(u[1:-1, 1:-1]))
    integrand = np.sum((ux2 + uy2) + ut2 + (cos))
    return 0.5 * integrand * dx * dy

T = 10.
def plot_energy_with_err(un, vn, names, nt, dx, dy):
    k = len(names)
    assert len(un) == k
    assert len(vn) == k
    fig, axs = plt.subplots(figsize=(20, 20), ncols=2,)

    dt = T / nt
    for l in range(k):
        es = []
        for i in range(nt):
            u, v = un[l][i], vn[l][i]
            es.append(energy(u, v, dx, dy))

        axs[0].plot(np.linspace(1, T, nt-1), es[1:], label=names[l])

        axs[0].set_ylabel("E / [1]")
        axs[0].set_xlabel("T / [1]")

        axs[1].plot(np.linspace(1, T, nt-1), np.log(np.abs(np.array(es[1:]) - es[0])), label=names[l])
        axs[1].set_ylabel("$\\log (E_0 - E(t))$ / [1]")
        axs[1].set_yscale("log")

        axs[0].hlines(es[0], xmin=0 + dt, xmax=T, alpha=.5, linestyle='-.')
        for i in range(2):
            axs[i].grid(True)

    plt.suptitle(f"$h={dx:.4f}$, $\\tau={dt:.4f}$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fname1_u = str(argv[1])
    fname1_v = str(argv[2])

    fname2_u = str(argv[3])
    fname2_v = str(argv[4])
    

    data1_u = np.load(fname1_u)
    data1_v = np.load(fname1_v)

    data2_u = np.load(fname2_u)
    data2_v = np.load(fname2_v)
     
    L = 4.
    nt, nx, ny = data1_u.shape
    #assert nt, nx, ny == data1_v.shape 
    #assert nt, nx, ny == data2_u.shape
    #assert nt, nx, ny == data2_v.shape

    dx = 2 * L / nx
    dy = 2 * L / ny
 
    names = [fname1_u, fname2_u] 

    un = np.zeros((2, nt, nx, ny), dtype=float)
    vn = np.zeros((2, nt, nx, ny), dtype=float)
    
    un[0] = data1_u
    un[1] = data2_u 
    vn[0] = data1_v
    vn[1] = data2_v


    plot_energy_with_err(un, vn, names, nt, dx, dy)
