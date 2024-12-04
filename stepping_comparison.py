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
    integrand = np.sum(ux2 + uy2 + ut2 + cos)
    # simple trapeziodal rule
    return 0.5 * integrand * dx * dy

class SineGordonIntegrator:
    def __init__(self, L, T, nt, nx, ny, stepping_method="stormer_verlet", test_problem=False):
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

        self.stepping_methods = {
                "forward_euler": self.forward_euler_step, "leap_frog": self.leap_frog_step,
                "stormer_verlet": self.stormer_verlet_step,
                "alternating": self.alternating_higher_order}

        if stepping_method not in self.stepping_methods.keys():
            raise NotImplemented
        else:
            self.step = self.stepping_methods[stepping_method]
            self.method_name = stepping_method
        self.ready = False
        self.test_problem = test_problem


    @staticmethod
    def initial_u(x, y):
        return 4 * np.arctan(np.exp(x + y))

    @staticmethod
    def initial_v(x, y):
        return -4 * np.exp(x + y) / (1 + np.exp(2 * x + 2 * y)) 

    def initial_u_grf(self, x, y):
        #sig = np.sin(np.random.rand(self.nx, self.ny))
        #mean = np.zeros(sig.shape[0])
        #f = np.random.multivariate_normal(mean, sig)
        #return f 

        #return np.exp(-(x ** 2 + y ** 2))

        #return 4 * np.arctan(
        #        np.exp(x)) + 4 * np.arctan(np.exp(y))

        #s = 1.5
        #r = 2.

        #d1 = (x - s) ** 2 + (y - s) ** 2  
        #d2 = (x + s) ** 2 + (y + s) ** 2

        #m1 = d1 <= r ** 2 
        #m2 = d2 <= r ** 2

        #vals = np.zeros_like(x)
        #vals[m1] = -np.exp(-(x[m1] ** 2 + y[m1] ** 2))
        #vals[m2] =  np.exp(-(x[m2] ** 2 + y[m2] ** 2))
        #return np.arctan(vals)

        ## single soliton
        #return 4 * np.arctan(np.exp(x + y))

        ## soliton-antisoliton
        #return 4 * np.arctan(np.exp(y)) - 4 * np.arctan(np.exp(x))

        # "static breather-like"
        omega = 1.6
        return 4 * np.arctan(np.sin(omega * x) / np.cosh(omega * y))


        ## periodic lattice solitons
        #m = 15
        #n = m // 2
        #L = self.L / m
        #u = 0
        #for i in range(m):
        #    for j in range(n):
        #        u += np.arctan(np.exp(x - n * L)) 
        #for i in range(m):
        #    for j in range(n):
        #        u += np.arctan(np.exp(y - m * L)) 
        #return 4 * u

        ## ring soliton
        #R = 1.5
        #return 4 * np.arctan((x ** 2 + y ** 2 - R ** 2) / (2 * R))

        #from scipy.special import ellipj
        #m = 0.5
        #u = (X + Y) / (X ** 2 + Y ** 2)
        #sn, cn, dn, ph = ellipj(u, m)
        #return sn
        
        
        

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
        dx = abs(self.xmax - self.xmin) / (self.nx - 2)
        dy = abs(self.ymax - self.ymin) / (self.ny - 2)
        dt = self.dt

        # u's ghost cells get approximation following boundary condition
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

    def apply_neumann_boundary(self, u, v):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        
        u[1:-1, 0] = u[1:-1, 1]
        u[1:-1, -1] = u[1:-1, -2]
        
        u[0, 0] = (u[1, 0] + u[0, 1])/2
        u[-1, 0] = (u[-2, 0] + u[-1, 1])/2
        u[0, -1] = (u[1, -1] + u[0, -2])/2
        u[-1, -1] = (u[-2, -1] + u[-1, -2])/2
         
        v[0, 1:-1]  = 0
        v[-1, 1:-1] = 0
        v[1:-1, 0]  = 0
        v[1:-1, -1] = 0

 
    def forward_euler_step(self, u, v, dt, t, i):
        u_n = u + dt * v
        
        v_n = v + dt * (self.lapl(u) - np.sin(u))
        self.apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n

    def leap_frog_step(self, u, v, dt, t, i):
        half_v = v + .5 * dt * (self.lapl(u) - np.sin(u))
        
        u_n = u + dt * half_v
        acc = self.lapl(u_n) - np.sin(u_n)
        v_n = half_v + .5 * dt * acc
        self.apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n
        
    def stormer_verlet_step(self, u, v, dt, t, i):
        def f(x):
            #return np.sin(u)
            return x + x ** 3

        if i == 1 and self.test_problem:
            u_n = u + dt * v
            v = analytical_velocity(self.X, self.Y, dt)
            return u_n, v
        
        elif i == 1:
            v = np.zeros_like(u)
            u_n = u + dt * v  + .5 * dt ** 2 * (self.lapl(u) - f(u))
            return u_n, v


        op = self.lapl(u) - f(u)
        u_n = 2 * self.u[i - 1] - self.u[i - 2] + op * dt ** 2
        # we don't need the velocity for the störmer-verlet method
        v_n = (u_n - self.u[i - 1]) / dt
        if self.test_problem:
            self.apply_boundary_condition(u_n, v_n, t)
        else:
            self.apply_neumann_boundary(u_n, v_n)
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
        raise NotImplemented
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
    def laplacian_flat(nx, ny, t):
        from scipy import sparse
        N = nx * ny
        # TODO: check this carefully. Might not be ideal.
        return (sparse.linalg.LaplacianNd(
                (nx, ny), boundary_conditions='neumann')).tosparse()

    def alternating_higher_order(self, u, ub, v, vb, dt, t, i):
        dx = abs(self.xmax - self.xmin) / (nx - 1)
        dy = abs(self.ymax - self.ymin) / (ny - 1)
        rx = self.dt / dx
        ry = self.dt / dy
        
        def u_yy(a,):
            uyy = np.zeros_like(a)
            uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1,1:-1])/(dy ** 2)
            return uyy
        def u_xx(a,):
            uxx = np.zeros_like(a)
            uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1,1:-1])/(dx ** 2)
            return uxx
        
        u1 = 2 * (u - ub) - self.dt ** 2 * np.sin(u)
        # apply (1 + 1/12 * delta_x²) * ((1 + 1/12 * delta_y²))
        # ie. (Id + 1/12 L_x) * (Id + 1/12 L_y) @ u1
        u1 = (np.ones_like(u) + 1/12 * u_xx(u1)) * (np.ones_like(u) + 1/12 * u_yy(u1))

        u2 = rx ** 2 * u_xx(ub) * (np.ones_like(ub) + 1/12 * u_yy(ub)) +\
                ry ** 2 * u_yy(ub) * (np.ones_like(ub) + 1/12 * u_xx(ub))
        # might need to fill boundary conditions here
        rhs = (u1 + u2).reshape(((nx + 2) * (ny + 2)))

        u_n, v_n = u, v

        """
        # TODO -- build up two stencil matrices 
        # -> can be done upon instantiating the solver object
        # and can stay in mem without change!
        Ax = lil_matrix((nx * ny, nx * ny)) 
        Ay = lil_matrix((nx * ny, nx * ny))

        # (need to have current boundaries instantiated!)
        un_intermediate = spsolve(Ax, rhs)
        un = spsolve(Ay, un_intermediate)

        u_n = ub + un
        v_n = (u_n - u) / dt
        """

        return u_n, v_n






    def evolve(self):
        if self.test_problem:
            u0 = self.initial_u(self.X, self.Y)
            v0 = self.initial_v(self.X, self.Y)

        else:
            u0 = self.initial_u_grf(self.X, self.Y)
            v0 = np.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0

        if self.method_name == "alternating":
            # boundary condition ĝ
            self.u[1] = u0 + self.dt * v0 + .5 * self.dt ** 2 * (  
                    self.lapl(v0) - np.sin(u0))

        i = 1
        t = 0
        tn = np.linspace(self.dt, self.T-self.dt, self.nt-1)
        for t in tqdm(tn): 
            if self.method_name == "alternating":
                self.u[i], self.v[i] = self.alternating_higher_order(
                        self.u[i - 1], self.u[i - 2],
                        self.v[i - 1], self.v[i - 2],
                        self.dt, t, i)
            else:
                self.u[i], self.v[i] = self.step(
                        self.u[i-1], self.v[i-1], self.dt, t, i) 
            i += 1
        self.ready = True
            
            
def analytical_solution(x, y, t):
    return 4 * np.arctan(np.exp(x + y - t))

def analytical_velocity(x, y, t):
    return -4 * np.exp(x + y - t) / (1 + np.exp(x + y - t) ** 2)

def compare_energy(solver):
    assert solver.ready
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
            
    tn = np.linspace(0, solver.T, solver.nt)
    plt.plot(tn, es, label="numerical energy")
    plt.plot(tn, es_analytical, label="analytical")
    plt.plot(tn, np.abs(np.array(es) - np.array(es_analytical)), label="diff")
    plt.title(f"Energy: solver {solver.method_name}")
    plt.xlabel("T / [1]")
    plt.ylabel("E / [1]")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_comparison_at_t(solver, t):
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


def compare_methods():
    L = 7 
    T = 30
    nt = 1001
    nx, ny = 130, 130
    dt = T / nt
    xn = yn = np.linspace(-L,L,nx+2)
    X, Y = np.meshgrid(xn,yn)

    tn = np.linspace(0, T, nt)
    analytical_sol = np.zeros((nt, nx+2, ny+2))
    for t in range(nt):
        analytical_sol[t] = analytical_solution(X, Y, t) 
    # "forward_euler",
    methods = [ "stormer_verlet", "leap_frog"]
    errs_l_infty = np.zeros((nt, len(methods))).T 
    errs_l2 = np.zeros((nt, len(methods))).T
    for i, method in enumerate(methods):  
        solver = SineGordonIntegrator(L, T, nt, nx, ny, method)
        solver.evolve()
        if not solver.ready: raise Exception    
        compare_energy(solver)
        for j in range(int(nt)):
            errs_l_infty[i, j] = np.abs(solver.u[j, 1:-1, 1:-1] - analytical_sol[j, 1:-1, 1:-1]).max()   
            errs_l2[i, j] = np.sqrt(((solver.u[j, 1:-1, 1:-1] - analytical_sol[j, 1:-1, 1:-1]))**2).mean()

    
    for i, method in enumerate(methods):
        plt.scatter(range(nt), errs_l_infty[i], label=f"L_infty {method}") 
        plt.scatter(range(nt), errs_l2[i], label=f"L_2 {method}")
    plt.grid(True)
    plt.yscale("log")
    plt.xlabel("T / [1]")
    plt.ylabel("Err / [1]")
    plt.legend()
    plt.show()


    
if __name__ == '__main__':
    #compare_methods()

    L = 7 
    T = 8
    nt = 301
    nx, ny = 30, 30#130, 130
    dt = T / nt
    xn = yn = np.linspace(-L,L,nx+2)
    X, Y = np.meshgrid(xn,yn)

    tn = np.linspace(0, T, nt)
    analytical_sol = np.zeros((nt, nx+2, ny+2))
    for t in range(nt):
        analytical_sol[t] = analytical_solution(X, Y, t) 
    solver = SineGordonIntegrator(L, T, nt, nx, ny, test_problem=False)
    solver.evolve()

    #L = 7 
    #T = 30
    #nt = 1001
    ##nx, ny = 54 + 2, 54 + 2
    #nx, ny = 130, 130
    #solver = SineGordonIntegrator(L, T, nt, nx, ny)
    #solver.evolve()
    
    #compare_energy(solver)
    
    #for t in (range(0, solver.nt, solver.nt//10)): 
    #    plot_comparison_at_t(solver, t)
    #    #ti = solver.dt * t

    #    #fig, axs = plt.subplots(figsize=(20, 20),nrows=2, ncols=3,
    #    #        subplot_kw={"projection":'3d'})
    #
    #    #axs[0][0].plot_surface(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1],
    #    #            (solver.u[t][1:-1, 1:-1]),
    #    #            cmap='viridis')
    #    #axs[0][1].plot_surface(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1],
    #    #        (analytical_solution(solver.X[1:-1,1:-1], solver.Y[1:-1,1:-1], ti)),
    #    #                    cmap='viridis')
    #    #print(((
    #    #                (solver.u[t][1:-1, 1:-1] - analytical_solution(
    #    #                    solver.X, solver.Y, ti)[1:-1, 1:-1])
    #    #            ) ** 2).mean()
    #    #            ,
    #    #            (np.abs(
    #    #                (solver.u[t][1:-1, 1:-1] - analytical_solution(
    #    #                    solver.X, solver.Y, ti)[1:-1, 1:-1])
    #    #            )).max()
    #    #            )

    #    #axs[0][2].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
    #    #            (
    #    #                np.abs(solver.u[t][1:-1, 1:-1] - analytical_solution(
    #    #                    solver.X, solver.Y, ti)[1:-1, 1:-1])
    #    #            ), 
    #    #            cmap='viridis')
    #    #axs[1][0].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
    #    #                (solver.v[t][1:-1, 1:-1]),
    #    #                    cmap='viridis')
    #    #axs[1][1].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
    #    #                (analytical_velocity(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1], ti)),
    #    #                    cmap='viridis')


    #    #axs[1][2].plot_surface(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1],
    #    #            (
    #    #            np.abs(solver.v[t][1:-1, 1:-1] - analytical_velocity(solver.X[1:-1, 1:-1], solver.Y[1:-1, 1:-1], ti))
    #    #            ), 
    #    #            cmap='viridis')

    #    #axs[1][0].set_title("numerical")
    #    #axs[1][1].set_title("analytical")
    #    #axs[1][2].set_title("residual")
    #    #fig.suptitle(f"{ti=:.2f}")

    #    # looking at above plots in fourier space (real part)
    #    #axs[0].plot_surface(solver.X, solver.Y,
    #    #                np.fft.fft2(solver.u[t]),
    #    #                    cmap='viridis')
    #    #axs[1].plot_surface(solver.X, solver.Y,
    #    #                np.fft.fft2(analytical_solution(solver.X, solver.Y, ti)),
    #    #                    cmap='viridis')


    #    #axs[2].plot_surface(solver.X, solver.Y,
    #    #            np.fft.fft2(
    #    #            np.abs(solver.u[t] - analytical_solution(solver.X, solver.Y, ti))
    #    #            ), 
    #    #            cmap='viridis')

    #    plt.show()


    #    #fig, axs = plt.subplots(figsize=(20, 20), ncols=4,) 
    #    #axs[0].plot(solver.yn, solver.u[t][0, :], )
    #    #axs[0].plot(solver.yn, solver.u[t][1, :], )

    #    #axs[1].plot(solver.yn, solver.u[t][-1, :],)
    #    #axs[1].plot(solver.yn, solver.u[t][-2, :],)

    #    #axs[2].plot(solver.xn, solver.u[t][:, 0],)
    #    #axs[2].plot(solver.xn, solver.u[t][:, 1],)
    #    #
    #    #axs[3].plot(solver.xn, solver.u[t][:,-1],)
    #    #axs[3].plot(solver.xn, solver.u[t][:,-2],)
    #    #plt.show()
     
    
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    data = solver.u
     
    X, Y = solver.X, solver.Y
    #zmin, zmax = np.min(data), np.max(data)
    #surf = ax.plot_surface(X, Y, data[0], cmap='viridis', vmin=zmin, vmax=zmax)
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',) 
    def update(frame):
        ax.clear()  # Clear the axis for the next frame
        #ax.set_zlim(np.min(data), np.max(data))
        ax.plot_surface(X, Y, data[frame], cmap='viridis')   

    fps = 30
    ani = FuncAnimation(fig, update, frames=solver.nt, interval=solver.nt / fps, )
   
    ani.save("kg-static-breather-like-large-omega.gif")
    
