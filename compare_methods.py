import numpy as np
import matplotlib.pyplot as plt

def energy(u, v, nx, ny, dx, dy):
    f = 2
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (f * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (f * dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux ** 2
    uy2 = uy ** 2
    ut2 = ut ** 2
    cos = 2 * (1 - np.cos(u[1:-1, 1:-1]))
    integrand = np.sum(ux2 + uy2 + ut2 + cos)
    # simple trapeziodal rule
    return 0.5 * integrand * dx * dy

if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    f1 = "rbf-testdata.npy"
    f2 = "sv-test-data.npy"

    rbf_data = np.load(f1)
    stv_data = np.load(f2)

    rbf_data_v = np.load(f1)
    stv_data_v = np.load(f2)

    L = 5
    nx = ny = 36
    dx = dy = 2 * L / nx
    xn, yn = np.linspace(-L, L, nx), np.linspace(-L, L, ny)
    X, Y = np.meshgrid(xn, yn)
    nt = 400
    T = 10.
    dt = T / nt
    tn = np.linspace(0, T, nt)

    assert rbf_data.shape == (nt, nx, ny) and stv_data.shape == (nt, nx, ny)  
    
    fig, axs = plt.subplots(figsize=(20, 20), ncols=3, subplot_kw={"projection":'3d'})
    [ax0, ax1, ax2] = axs 
    def update(frame):
        ax0.clear()
        ax1.clear()
        ax0.plot_surface(X, Y, rbf_data[frame], cmap='viridis')
        ax1.plot_surface(X, Y, stv_data[frame], cmap='viridis')
        ax2.plot_surface(X, Y, np.abs(rbf_data[frame] - stv_data[frame]), cmap='viridis')
        
        ax0.set_title(f"RBF t={frame * dt:.2f}")
        ax1.set_title(f"Stormer-Verlet")
        ax2.set_title("Residual")

    fps = 300
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    plt.show()

    es_rbf, es_stv = [], []
    for i in range(nt):
        t = dt * i
        es_rbf.append(
                energy(rbf_data[i], rbf_data_v[i], nx, ny, dx, dy)
                )
        es_stv.append(
                energy(stv_data[i], stv_data_v[i], nx, ny, dx, dy)
                )
    plt.plot(tn, es_rbf, label="RBF")
    plt.plot(tn, es_stv, label="Stormer-Verlet")
    plt.ylabel("E / 1")
    plt.xlabel("T / 1")
    plt.legend()
    plt.show()


