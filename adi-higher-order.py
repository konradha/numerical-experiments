import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

def compact_adi_sine_gordon(Nx, Ny, Nt, Lx, Ly, T, rho, f, g, g_t, h1, h2, h3, h4):
    hx = Lx / (Nx + 1)
    hy = Ly / (Ny + 1)
    dt = T / Nt
    
    x = np.linspace(-Lx, Lx, Nx + 2)
    y = np.linspace(-Lx, Ly, Ny + 2)
    t = np.linspace(0, T, Nt+1)
    
    rx = dt / (hx * np.sqrt(1 + 0.5 * rho * dt))
    ry = dt / (hy * np.sqrt(1 + 0.5 * rho * dt))
    
    U = np.zeros((Nt+1, Nx+2, Ny+2))
    
    X, Y = np.meshgrid(np.linspace(-Lx, Lx, Nx + 2), np.linspace(-Ly, Ly, Ny + 2))
    U[0] = g(X, Y)

    def apply_boundary_condition(u, t):
        u[0, 1:-1]  = u[1, 1:-1]  - hx * h1(Y[0,  1:-1], t)
        u[-1, 1:-1] = u[-2, 1:-1] + hx * h2(Y[-1, 1:-1], t)

        u[1:-1, 0]  = u[1:-1,  1] - hy * h3(X[1:-1,  0], t)
        u[1:-1, -1] = u[1:-1, -2] + hy * h4(X[1:-1, -1], t)

        u[0,  0] = .5 * (u[0, 1] + u[1,  0])
        u[-1 ,0] = .5 * (u[-2,0] + u[-1,-2])
        u[0, -1] = .5 * (u[0,-2] + u[-2,-1])
        u[-1,-1] = .5 * (u[-2,-1]+ u[-1,-2])


    U[1, 1:-1, 1:-1] = (U[0, 1:-1, 1:-1] + dt * g_t(X, Y)[1:-1, 1:-1]) +\
                        0.5 * dt**2 * ((U[0, 2:, 1:-1] - 2*U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) / hx**2 +
                                       (U[0, 1:-1, 2:] - 2*U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) / hy**2 -
                                       f(U[0, 1:-1, 1:-1]) -
                                       rho * g_t(X, Y)[1:-1, 1:-1])
    apply_boundary_condition(U[1], dt)

    #fig, axs = plt.subplots(figsize=(20, 20),nrows=1, ncols=1,
    #                        subplot_kw={"projection":'3d'})
    #l = 1
    ##axs.plot_surface(X[l:-l, l:-l], Y[l:-l, l:-l], U[1, l:-l, l:-l], cmap='viridis')
    #axs.plot_surface(X, Y, U[1, :, :], cmap='viridis')
    #plt.show()
    
    # Boundary conditions
    U[1:, 0,  :] = h1(y, t[1:, None])
    U[1:, -1, :] = h2(y, t[1:, None])
    U[1:, :,  0] = h3(x, t[1:, None])
    U[1:, :, -1] = h4(x, t[1:, None])
    

    from scipy.sparse import eye
    from scipy.sparse import diags, kron

    def u_xx_stencil(nx, ny):
        dx = 1 / nx
        main_diag = -2 * np.ones(nx)
        off_diag = np.ones(nx-1)
        Lx = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(nx, nx)) / (dx**2)
        return kron(eye(ny), Lx)

    def u_yy_stencil(nx, ny):
        dy = 1 / ny
        main_diag = -2 * np.ones(ny)
        off_diag = np.ones(ny-1)
        Ly = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(ny, ny)) / (dy**2)
        return kron(eye(nx), Ly)
 
    Ax = u_xx_stencil(Nx, Ny)
    Ay = u_yy_stencil(Nx, Ny)
    Id = eye(Nx * Ny)

    Mx = Id + (1/12 - .5 * rx**2) * Ax
    My = Id + (1/12 - .5 * ry**2) * Ay

    Bx = Id + 1/12 * Ay
    By = Id + 1/12 * Ax 

    A1 = Mx @ My
    A2 = rx ** 2 * Ax @ Bx
    A3 = ry ** 2 * Ay @ By
     
    for n in tqdm(range(2, Nt)):    
        ub  = U[n-1, 1:-1, 1:-1].reshape(Nx * Ny)
        ubb = U[n-2, 1:-1, 1:-1].reshape(Nx * Ny)

        F = f(U[n-1, 1:-1, 1:-1]).reshape(Nx * Ny)

        f1 = 1/(1 + .5 * rho * dt) * A1 @ (
                2 * (ub - ubb) - dt ** 2 * F
                )
        f2 = (A2 + A3) @ ub 

        rhs = f1 + f2 
        U_star = spsolve(Mx, rhs)
        contrib = spsolve(My, U_star).reshape((Nx, Ny))

        U[n, 1:-1, 1:-1] = U[n-2, 1:-1, 1:-1] + contrib

    return U

if __name__ == '__main__':
    Nx = Ny = 51
    L = 7.
    Lx = Ly = L
    T = .5
    rho = 0

    dt = 1e-3
    Nt = int(T / dt)

    def u0(x, y):
        return 4 * np.arctan(np.exp(x + y))

    def v0(x, y):
        return -4. * np.exp(x + y) / (1 + np.exp(2*x + 2*y))
    
    def u_t_x(x, y, t):
        return 4 * np.exp(x + y + t) / (np.exp( 2 * t) + np.exp(2*x + 2*y))

    def u_t_y(x, y, t):
        return 4 * np.exp(x + y + t) / (np.exp( 2 * t) + np.exp(2*x + 2*y))

    def h1(y, t):
        return u_t_x(-Lx * np.ones_like(y), y, t)

    def h2(y, t):
        return u_t_x(Lx * np.ones_like(y), y, t)

    def h3(x, t):
        return u_t_y(x, -Ly * np.ones_like(x), t)

    def h4(x, t):
        return u_t_y(x, -Ly * np.ones_like(x), t)

    X, Y = np.meshgrid(np.linspace(-Lx, Lx, Nx + 2), np.linspace(-Ly, Ly, Ny + 2))
    U = compact_adi_sine_gordon(Nx, Ny, Nt, Lx, Ly, T, rho, np.sin, u0, v0, h1, h2, h3, h4)
    for i in range(0, Nt, 100):
        fig, axs = plt.subplots(figsize=(20, 20),nrows=1, ncols=1,
                            subplot_kw={"projection":'3d'})
        l = 1
        axs.plot_surface(X[l:-l, l:-l], Y[l:-l, l:-l], U[i, l:-l, l:-l], cmap='viridis')
        #axs.plot_surface(X, Y, U[i,:,:], cmap='viridis')
        plt.show()
