import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

def calculate_energy(u, v, nx, ny, dx, dy): 
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux**2
    uy2 = uy**2
    ut2 = ut**2
    cos = 2*(1 - np.cos(u[1:-1, 1:-1]))
    integrand = ux2 + uy2 + ut2 + cos
    # simple trapeziodal rule
    return 0.5 * np.sum(integrand) * dx * dy

class SineGordonIntegrator:
    def __init__(self, L, T, nt, nx, ny):
        self.L = L
        self.T = T
        self.nt = nt
        self.dt = T / (nt - 1)
        self.nx = nx + 2
        self.ny = ny + 2
        self.xn = np.linspace(-L, L, self.nx)
        self.yn = np.linspace(-L, L, self.ny)

        self.xmin, self.xmax = -L , L 
        self.ymin, self.ymax = -L , L 


        self.u = np.zeros(shape=(self.nt, self.nx, self.ny))
        self.v = np.zeros(shape=(self.nt, self.nx, self.ny))

        self.X, self.Y = np.meshgrid(self.xn, self.yn)

        self.lapl_mat = None


    @staticmethod
    def initial_u(x, y):
        return 4 * np.arctan(np.exp(x + y))

    @staticmethod
    def initial_v(x, y):
        return -4 * np.exp(x + y) / (1 + np.exp(2 * x + 2 * y)) 

    @staticmethod
    def boundary_x(X, y, t):
        # x = -L or x = L
        return 4 * np.exp(X + y + t) / (np.exp(2 * t) + np.exp(2 * X + 2 * y))

    @staticmethod
    def boundary_y(x, Y, t):
        # y = -L or y = L
        return 4 * np.exp(x + Y + t) / (np.exp(2 * t) + np.exp(2 * x + 2 * Y))
 
    def lapl(self, u):

        def u_yy(a,):
            nx, ny = a.shape
            dy = abs(self.ymax - self.ymin) / (ny - 1)
            uyy = np.zeros_like(a)

            uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1,1:-1])/(dy ** 2)
            return uyy

        def u_xx(a,):
            nx, ny = a.shape
            dx = abs(self.xmax - self.xmin) / (nx - 1)
            uxx = np.zeros_like(a)

            uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1,1:-1])/(dx ** 2)
            return uxx
        return u_xx(u) + u_yy(u)

    def apply_boundary_condition(self, u, v, t):
        dx = abs(self.xmax - self.xmin) / (self.nx - 1)
        dy = abs(self.ymax - self.ymin) / (self.ny - 1)
        dt = self.dt

        # u's ghost cells get an averaging
        u[0, 1:-1] = u[1, 1:-1] - dx*self.boundary_x(self.xmin, self.yn[1:-1], t)
        u[-1, 1:-1] = u[-2, 1:-1] + dx*self.boundary_x(self.xmax, self.yn[1:-1], t)
        
        u[1:-1, 0] = u[1:-1, 1] - dy*self.boundary_y(self.xn[1:-1], self.ymin, t)
        u[1:-1, -1] = u[1:-1, -2] + dy*self.boundary_y(self.xn[1:-1], self.ymax, t)
        
        u[0, 0] = (u[1, 0] + u[0, 1])/2 - dx*self.boundary_x(self.xmin, self.ymin, t)/2 \
                  - dy*self.boundary_y(self.xmin, self.ymin, t)/2
        u[-1, 0] = (u[-2, 0] + u[-1, 1])/2 + dx*self.boundary_x(self.xmax, self.ymin, t)/2 \
                   - dy*self.boundary_y(self.xmax, self.ymin, t)/2
        u[0, -1] = (u[1, -1] + u[0, -2])/2 - dx*self.boundary_x(self.xmin, self.ymax, t)/2 \
                   + dy*self.boundary_y(self.xmin, self.ymax, t)/2
        u[-1, -1] = (u[-2, -1] + u[-1, -2])/2 + dx*self.boundary_x(self.xmax, self.ymax, t)/2 \
                    + dy*self.boundary_y(self.xmax, self.ymax, t)/2
        
        # v get the hard boundary condition
        v[0, 1:-1] = self.boundary_x(self.xmin, self.yn[1:-1], t)
        v[-1, 1:-1] = self.boundary_x(self.xmax, self.yn[1:-1], t)
        v[1:-1, 0] = self.boundary_y(self.xn[1:-1], self.ymin, t)
        v[1:-1, -1] = self.boundary_y(self.xn[1:-1], self.ymax, t)
 
    def forward_euler_step(self, u, v, dt, t):
        u_n = u + dt * v
        
        v_n = v + dt * (self.lapl(u) - np.sin(u))
        self.apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n

    def leap_frog_step(self, u, v, dt, t):
        half_v = v + .5 * dt * (self.lapl(u) - np.sin(u))
        #self.apply_boundary_condition(u, half_v, t)
        
        u_n = u + dt * half_v
        acc = self.lapl(u_n) - np.sin(u_n)
        v_n = half_v + .5 * dt * acc
        self.apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n
        
    def stormer_verlet_step(self, u, v, dt, t, i):
        if i == 1:
            u_n = u + dt * v
            v = analytical_velocity(self.X, self.Y, dt)
            return u_n, v

        op = self.lapl(u) - np.sin(u)
        u_n = 2 * self.u[i - 1] - self.u[i - 2] + op * dt ** 2
        v_n = (u_n - self.u[i - 1]) / dt
        self.apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n 
        

    def split_step(self, u, v, dt, t):
        nx, ny = self.nx, self.ny
        n = nx * ny
        dx = (self.xmax - self.xmin) / nx
        dy = (self.ymax - self.ymin) / ny
        # TODO: figure out if this could be useful here
        # (cf. NLSE solver using split operator method)
        pass

    def reproducing_paper_step(self, u, v, dt, t, i):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        if i == 1:
            self.lapl_mat = self.laplacian_flat(self.nx, self.ny) 
            u_neg = (self.u[0] - 2 * dt * self.v[0]).flatten() 
        else:
            assert i > 1
            u_neg = self.u[i - 2].flatten()

        N = (self.nx) * (self.ny)
        Id = sparse.eye(N)
        #import pdb; pdb.set_trace() 
        scaled_mat = .25 * self.lapl_mat 
        A = Id - scaled_mat.tocsr()
        
        t1 = (2. / dt ** 2 * Id + .5 * self.lapl_mat).dot(
                self.u[i - 1].flatten())
        t2 = np.sin(self.u[i - 1].flatten())
        t3 = (- 1 / dt ** 2 * Id + .25 * self.lapl_mat).dot(u_neg)

        u_n = np.zeros((self.nx, self.ny))
        u_n = spsolve(A, dt ** 2 * (t1 + t2 + t3)).reshape(
                (self.nx, self.ny)) 

        self.apply_boundary_condition(u_n, self.v[i-1], t)

        v_n = (u_n - self.u[i - 1]) / dt
        return u_n, v_n 

    @staticmethod
    def laplacian_flat(nx, ny):
        from scipy import sparse
        N = nx * ny
        # TODO: check this carefully. Might not be ideal.
        return (sparse.linalg.LaplacianNd(
                (nx, ny), boundary_conditions='neumann')).tosparse()


        """

        indices should be:

    (0, 0) (0, 1) ... ghost cells -- need updating according to BCs 
    (0, 1)
    (0, 2)
        .
        .
        .

        Then: inner:
        (1, 1) first "inner" point where the stencil can be performed


        """


    def evolve(self):
        u0 = self.initial_u(self.X, self.Y)
        v0 = self.initial_v(self.X, self.Y)
    
        self.u[0] = u0
        self.v[0] = v0

        i = 1
        t = 0
        tn = np.linspace(self.dt, self.T-self.dt, self.nt-1)
        for t in tqdm(tn): 
            self.u[i], self.v[i] = self.stormer_verlet_step(
                    self.u[i-1], self.v[i-1], self.dt, t, i) 
            i += 1
            
            
def analytical_solution(x, y, t):
    return 4 * np.arctan(np.exp(x + y - t))

def analytical_velocity(x, y, t):
    return -4 * np.exp(x + y - t) / (1 + np.exp(x + y - t) ** 2)

if __name__ == '__main__':

    L = 7 
    T = 15
    nt = 1001
    nx, ny = 54 + 2, 54 + 2
    solver = SineGordonIntegrator(L, T, nt, nx, ny)
    solver.evolve()
    
    """
    for t in (range(0, solver.nt, solver.nt//10)): 
        ti = solver.dt * t

        fig, axs = plt.subplots(figsize=(20, 20),nrows=2, ncols=3,
                subplot_kw={"projection":'3d'})
    
        axs[0][0].plot_surface(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1],
                    (solver.u[t][1:-1, 1:-1]),
                    cmap='viridis')
        axs[0][1].plot_surface(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1],
                (analytical_solution(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1], ti)),
                            cmap='viridis')


        axs[0][2].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
                    (
                        np.abs(solver.u[t][1:-1, 1:-1] - analytical_solution(
                            solver.X, solver.Y, ti)[1:-1, 1:-1])
                    ), 
                    cmap='viridis')
        axs[1][0].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
                        (solver.v[t][1:-1, 1:-1]),
                            cmap='viridis')
        axs[1][1].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
                        (analytical_velocity(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1], ti)),
                            cmap='viridis')


        axs[1][2].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
                    (
                    np.abs(solver.v[t][1:-1, 1:-1] - analytical_velocity(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1], ti))
                    ), 
                    cmap='viridis')

        axs[1][0].set_title("numerical")
        axs[1][1].set_title("analytical")
        axs[1][2].set_title("residual")
        fig.suptitle(f"{ti=:.2f}")

        # looking at above plots in fourier space (real part)
        #axs[0].plot_surface(solver.X, solver.Y,
        #                np.fft.fft2(solver.u[t]),
        #                    cmap='viridis')
        #axs[1].plot_surface(solver.X, solver.Y,
        #                np.fft.fft2(analytical_solution(solver.X, solver.Y, ti)),
        #                    cmap='viridis')


        #axs[2].plot_surface(solver.X, solver.Y,
        #            np.fft.fft2(
        #            np.abs(solver.u[t] - analytical_solution(solver.X, solver.Y, ti))
        #            ), 
        #            cmap='viridis')

        plt.show()


        fig, axs = plt.subplots(figsize=(20, 20), ncols=4,) 
        axs[0].plot(solver.yn, solver.u[t][0, :], )
        axs[0].plot(solver.yn, solver.u[t][1, :], )

        axs[1].plot(solver.yn, solver.u[t][-1, :],)
        axs[1].plot(solver.yn, solver.u[t][-2, :],)

        axs[2].plot(solver.xn, solver.u[t][:, 0],)
        axs[2].plot(solver.xn, solver.u[t][:, 1],)
        
        axs[3].plot(solver.xn, solver.u[t][:,-1],)
        axs[3].plot(solver.xn, solver.u[t][:,-2],)
        plt.show()
    """ 
    
    es = []
    es_analytical = []
    for t in (range(0, solver.nt)):
        dx = abs(solver.xmax - solver.xmin) / (solver.nx - 1)
        dy = abs(solver.ymax - solver.ymin) / (solver.ny - 1)
        ti = solver.dt * t
        es.append(
            calculate_energy(solver.u[t][1:-1, 1:-1], solver.v[t][1:-1, 1:-1],
                solver.nx, solver.ny, dx, dy)
            )
        es_analytical.append(
            calculate_energy(analytical_solution(solver.X[1:-1, 1:-1],
                solver.Y[1:-1, 1:-1], ti),
                analytical_velocity(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1], ti),
                solver.nx, solver.ny, dx, dy)
            )
            
    plt.plot(es, label="numerical energy")
    plt.plot(es_analytical, label="analytical")
    plt.legend()
    plt.show()
