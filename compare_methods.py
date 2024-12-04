import numpy as np
import matplotlib.pyplot as plt

def energy(u, v, nx, ny, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux ** 2
    uy2 = uy ** 2
    ut2 = ut ** 2
    cos = 2 * (1 - np.cos(u[1:-1, 1:-1]))
    integrand = ux2 + uy2 + ut2 + cos
    # trapezoidal rule
    return np.sum(0.5 * integrand * dx * dy)

def topological_charge(u, dx):
    u_x = np.zeros_like(u)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    topological_charge = (1 / (2 * np.pi)) * u_x[:, 1] * dx
    return np.sum(topological_charge)

if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    f1 = "rbf-testdata.npy"
    f2 = "sv-test-data.npy"

    # this data now only has internal points
    # (no ghost points as needed in Stormer-Verlet FD method)
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
    tc_rbf, tc_stv = [], []
    for i in range(nt):
        t = dt * i
        es_rbf.append(
                energy(rbf_data[i], rbf_data_v[i], nx, ny, dx, dy)
                )
        es_stv.append(
                energy(stv_data[i], stv_data_v[i], nx, ny, dx, dy)
                )

        tc_rbf.append(
                topological_charge(rbf_data[i], dx)
                )
        tc_stv.append(
                topological_charge(stv_data[i], dx)
                )

    fig, axs = plt.subplots(figsize=(20, 20), ncols=2,)
    [ax0, ax1] = axs 

    ax0.plot(tn, es_rbf, label="RBF")
    ax0.plot(tn, es_stv, label="Stormer-Verlet")
    ax0.set_ylabel("E / 1")
    ax0.set_xlabel("T / 1")
    ax0.legend()
    ax0.set_title("Energy over time")

    ax1.plot(tn, tc_rbf, label="RBF")
    ax1.plot(tn, tc_stv, label="Stormer-Verlet")
    ax1.set_ylabel("TC / 1")
    ax1.set_xlabel("T / 1")
    ax1.legend()
    ax1.set_title("Topological charge over time")

    plt.show()
