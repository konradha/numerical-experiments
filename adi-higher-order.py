import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

def analytical(x, y, t):
    return 4 * np.arctan(np.exp(x + y - t))

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
        # paper suggest to only apply the x-boundary operator for \Delta U *
        u[0, 1:-1]  = u[1, 1:-1]  - hx * h1(Y[0,  1:-1], t)
        u[-1, 1:-1] = u[-2, 1:-1] + hx * h2(Y[-1, 1:-1], t)

        u[:, 0]  = u[:,  1] - hy * h3(X[:,  0], t)
        u[:, -1] = u[:, -2] + hy * h4(X[:, -1], t)

        #u[1:-1, 0]  = u[1:-1,  1] - hy * h3(X[1:-1,  0], t)
        #u[1:-1, -1] = u[1:-1, -2] + hy * h4(X[1:-1, -1], t)

        #u[0,  0] = .5 * (u[0, 1] + u[1,  0])
        #u[-1 ,0] = .5 * (u[-2,0] + u[-1,-2])
        #u[0, -1] = .5 * (u[0,-2] + u[-2,-1])
        #u[-1,-1] = .5 * (u[-2,-1]+ u[-1,-2])


    U[1, 1:-1, 1:-1] = (U[0, 1:-1, 1:-1] + dt * g_t(X, Y)[1:-1, 1:-1]) +\
                        0.5 * dt**2 * ((U[0, 2:, 1:-1] - 2*U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) / hx**2 +
                                       (U[0, 1:-1, 2:] - 2*U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) / hy**2 -
                                       f(U[0, 1:-1, 1:-1]) -
                                       rho * g_t(X, Y)[1:-1, 1:-1])
    #apply_boundary_condition(U[1], dt)

    #fig, axs = plt.subplots(figsize=(20, 20),nrows=1, ncols=1,
    #                        subplot_kw={"projection":'3d'})
    #l = 1
    ##axs.plot_surface(X[l:-l, l:-l], Y[l:-l, l:-l], U[1, l:-l, l:-l], cmap='viridis')
    #axs.plot_surface(X, Y, U[1, :, :], cmap='viridis')
    #plt.show()
    
    # Boundary conditions
    # x
    U[1:,  0, :] = analytical(-Lx, y, t[1:, None])
    U[1:, -1, :] = analytical(Lx, y, t[1:, None])

    # y
    U[1:, 1:-1,  0] = analytical(x[1:-1], -Ly, t[1:, None])
    U[1:, 1:-1, -1] = analytical(x[1:-1], Ly, t[1:, None])


    #fig, axs = plt.subplots(figsize=(20, 20),nrows=2, ncols=2,)
    #for i, h in enumerate([h1, h2, h3, h4]):
    #    x_idx = i // 2 
    #    y_idx = i % 2
    #    axs[x_idx][y_idx].plot(
    #            np.linspace(-Lx, Lx, Nx + 2),
    #            h(np.linspace(-Lx, Lx, Nx + 2), 0)
    #            )
    #plt.show()
    

    from scipy.sparse import eye
    from scipy.sparse import diags, kron

    def u_xx_stencil(nx, ny):
        dx = 1 / nx
        main_diag = -2 * np.ones(nx + 2)
        off_diag = np.ones(nx + 2 -1)
        Lx = diags([off_diag, main_diag, off_diag],
                [-1, 0, 1], shape=(nx + 2, nx + 2))
        return kron(eye(ny + 2), Lx)

    def u_yy_stencil(nx, ny):
        dy = 1 / ny
        main_diag = -2 * np.ones(ny + 2)
        off_diag = np.ones(ny + 2 -1)
        Ly = diags([off_diag, main_diag,
            off_diag], [-1, 0, 1], shape=(ny + 2, ny + 2))
        return kron(Ly, eye(nx + 2))

    def u_yy(a,):
        # these are nx + 2, ny + 2 (ie. just for the sake of making this short
        # we name them nx, ny
        nx, ny = a.shape
        dy = 1 / ny
        uyy = np.zeros_like(a)
        uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1,1:-1])
        return uyy

    def u_xx(a,):
        # these are nx + 2, ny + 2 (ie. just for the sake of making this short
        # we name them nx, ny
        nx, ny = a.shape
        dx = 1 / nx
        uxx = np.zeros_like(a)
        uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1,1:-1])
        return uxx
 
    Ax = u_xx_stencil(Nx, Ny)
    Ay = u_yy_stencil(Nx, Ny)
    Id = eye((Nx + 2) * (Ny + 2))

    Mx = Id + (1/12 + .5 * rx**2) * Ax
    My = Id + (1/12 + .5 * ry**2) * Ay

    Bx = Id + 1/12 * Ay
    By = Id + 1/12 * Ax 

    A1 = Mx @ My
    A2 = rx ** 2 * Ax @ Bx
    A3 = ry ** 2 * Ay @ By

    def d2y(a):
        d = np.zeros_like(a)
        d[1:-1] = (a[2:] - 2*a[1:-1] + a[:-2]) 
        return d
 
    for n in tqdm(range(2, Nt)):    
        """
        ub  = U[n - 1, :, :]
        ubb = U[n - 2, :, :]

        F = f(U[n - 1, :, :])

        v1 = 2 * (ub - ubb) - dt ** 2 * F
        v1_star = np.eye(v1.shape[0]) + (1 / 12) * u_yy(v1) 
        v1_star = np.eye(v1.shape[0]) + (1 / 12) * u_xx(v1)
        v1 = 1 / (1 + .5 * rho * dt) * v1_star

        u1a = ry ** 2 * u_yy(np.eye(v1.shape[0]) + (1/12) * u_xx(ubb))
        u1b = rx ** 2 * u_xx(np.eye(v1.shape[0]) + (1/12) * u_yy(ubb))

        rhs = (v1 + u1a + u1b).reshape((Nx + 2) * (Ny + 2))

        U_star = spsolve(Mx, rhs)

        #U_star = U_star.reshape((Nx + 2, Ny + 2))
        #
        #U_star[ 0, :] = (h1(Y[0,:], n * dt) - h1(Y[0,:], (n-2) * dt))  
        #U_star[-1, :] = (h2(Y[-1,:], n * dt) - h2(Y[-1,:], (n-2) * dt))
        #U_star = U_star.reshape((Nx + 2) * (Ny + 2))

        contrib = spsolve(My, U_star).reshape((Nx + 2, Ny + 2)) 
        """


        # first version
        ub  = U[n-1, :, :].reshape((Nx + 2) * (Ny + 2))
        ubb = U[n-2, :, :].reshape((Nx + 2) * (Ny + 2))

        F = f(U[n-1, :, :]).reshape((Nx + 2) * (Ny + 2))
        
        f1 = (eye((Nx + 2) * (Ny + 2)) + 1/12 * Ax @\
                eye((Nx + 2) * (Ny + 2)) + 1/12 * Ay) @\
                ((2 * (ub - ubb) - dt ** 2 * F))  
        f2 = (rx ** 2 * Ax @ (eye((Nx + 2) * (Ny + 2)) + 1/12 * Ay) + \
                ry ** 2 * Ay @ (eye((Nx + 2) * (Ny + 2)) + 1/12 * Ax)) @ ubb

        rhs = f1 + f2 
         

        # \Delta U*
        U_star = spsolve(Mx, rhs)

        U_star = U_star.reshape((Nx + 2, Ny + 2))
        U_star[0, :] = (1 + (1/12 - .5 * ry**2)) * d2y(
    analytical(-Lx, Y[0, :], dt * (n + 1)) - analytical(-Lx, Y[0, :], dt * (n - 1))  
            )
        U_star[-1, :] = (1 + (1/12 - .5 * ry**2)) * d2y(
    analytical(Lx, Y[-1, :], dt * (n + 1)) - analytical(Lx, Y[-1, :], dt * (n - 1))  
            )
        U_star = U_star.reshape((Nx + 2) * (Ny + 2))
        
        # \Delta U**
        contrib = spsolve(My, U_star).reshape((Nx + 2, Ny + 2)) 


        U[n, :, :] = U[n-2, :, :] + contrib

    return U

if __name__ == '__main__':
    Nx = Ny = 51
    L = 7.
    Lx = Ly = L
    T = 5.
    rho = 0

    dt = 1e-2
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
    U = compact_adi_sine_gordon(
            Nx, Ny, Nt, Lx, Ly, T,
            rho, np.sin, u0, v0, h1, h2, h3, h4)

    for i in range(0, Nt, 10):
        fig, axs = plt.subplots(figsize=(20, 20),nrows=1, ncols=3,
                            subplot_kw={"projection":'3d'})  
        axs[0].plot_surface(X, Y,
        analytical(X, Y, i * dt),
                cmap='viridis')

        axs[1].plot_surface(X, Y,
                U[i],
                cmap='viridis')
        axs[2].plot_surface(X, Y,
                np.abs(
        analytical(X, Y, i * dt) - U[i]
                ),
                cmap='viridis')
        
        fig.suptitle(f"residual at timestep {i} = {i * dt}")       
        plt.show()
