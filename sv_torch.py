import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SineGordonIntegrator:
    def __init__(self, L, T, nt, nx, ny, device='cuda',):
        self.device = torch.device(device)
        self.L = L
        self.T = T
        self.nt = nt
        self.dt = T / nt
        self.nx = nx + 2
        self.ny = ny + 2

        self.tn = torch.linspace(0, T, nt)
        
        xn = torch.linspace(-L, L, self.nx, device=self.device)
        yn = torch.linspace(-L, L, self.ny, device=self.device)

        self.xmin, self.xmax = -L, L 
        self.ymin, self.ymax = -L, L 

        self.u = torch.zeros((self.nt, self.nx, self.ny), device=self.device)
        self.v = torch.zeros((self.nt, self.nx, self.ny), device=self.device)

        self.X, self.Y = torch.meshgrid(xn, yn, indexing='ij')

    def initial_u_grf(self, x, y):
        # single soliton solutions: need velocity boundary conditions
        # to yield coherent energy / topological charge

        ## single soliton -- this has an issue with the von neumann boundary condition!
        #return 4 * torch.arctan(torch.exp(x + y))

        ## single antisoliton -- this has an issue with the von neumann boundary condition!
        #return 4 * torch.arctan(torch.exp(-(x + y) / .2))

        ## soliton-antisoliton
        #return 4 * torch.arctan(torch.exp(y)) - 4 * torch.arctan(torch.exp(x))

        ## "static breather-like"
        #omega = .1
        #return 4 * torch.arctan(torch.sin(omega * x) / torch.cosh(omega * y))

        ## periodic lattice solitons
        #m = 25
        #n = m // 2
        #L = self.L / m ** 2 
        #u = 0
        #for i in range(m):
        #    for j in range(n):
        #        u += torch.arctan(torch.exp(x - n * L))
        #for i in range(m):
        #    for j in range(n):
        #        u += torch.arctan(torch.exp(y - m * L))
        #return u

        # ring soliton
        R = 1.001
        # stability assertion
        assert R > 1 and R ** 2 < 2 * (2 * self.L) ** 2
        return 4 * torch.arctan((x ** 2 + y ** 2 - R ** 2) / (2 * R))

        ## method to construct other ring solitons?
        #R = 1.5
        #c1 = 3
        #c2 = -c1
        #assert c1 > R ** 2
        #m1 = (x - c1) ** 2 + (y - c1) ** 2 < R
        #m2 = (x - c2) ** 2 + (y - c2) ** 2 < R
        #
        #u = torch.zeros_like(x)
        #u[m1] =  torch.exp(-(x[m1] ** 2 + y[m1] ** 2) / R ** 2)
        #u[m2] = -torch.exp(-(x[m2] ** 2 + y[m2] ** 2) / R ** 2)
        #return u


        ## "circular" elliptic Jacobi function -- easily yields instabilities!
        #from scipy.special import ellipj
        #X, Y = self.X, self.Y
        #m = .5
        #u = ((X + Y) / (X ** 2 + Y ** 2)).detach().numpy()
        #sn, cn, dn, ph = ellipj(u, m)
        #return torch.tensor(sn)

        ## elliptic Jacobi function -- easily yields instabilities for small dx, dy!
        #from scipy.special import ellipj
        #X, Y = self.X, self.Y
        #m = .5
        #u = (X + Y).detach().numpy()
        #sn, cn, dn, ph = ellipj(u, m)
        #return torch.tensor(sn)

 
    def lapl(self, u):
        def u_yy(a):
            dy = abs(self.ymax - self.ymin) / (a.shape[1] - 1)
            uyy = torch.zeros_like(a)
            uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
            return uyy

        def u_xx(a):
            dx = abs(self.xmax - self.xmin) / (a.shape[0] - 1)
            uxx = torch.zeros_like(a)
            uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2)
            return uxx
        
        return u_xx(u) + u_yy(u)


    def stormer_verlet_step(self, u, v, dt, t, i):
        def f(x):
            # sine-Gordon
            return torch.sin(x)

            ## Klein-Gordon
            #return x + x ** 3

        # first step needs special treatment as we cannot yet use the pre-preceding
        if i == 1:
            v = torch.zeros_like(u)
            u_n = u + dt * v + 0.5 * dt ** 2 * (self.lapl(u) - f(u))
            return u_n, v

        op = self.lapl(u) - f(u)
        u_n = 2 * self.u[i - 1] - self.u[i - 2] + op * dt ** 2
        v_n = (u_n - self.u[i - 1]) / dt
        
        self.apply_neumann_boundary(u_n, v_n)
        return u_n, v_n 

    def apply_neumann_boundary(self, u, v):
        # Boundary conditions similar to NumPy version
        u[0,  1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:,  0]    = u[:, 1]
        u[:, -1]    = u[:, -2]
 
        v[0, 1:-1] = 0
        v[-1, 1:-1] = 0
        v[1:-1, 0] = 0
        v[1:-1, -1] = 0

    def evolve(self):
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0

        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            self.u[i], self.v[i] = self.stormer_verlet_step(
                self.u[i-1], self.v[i-1], self.dt, t, i)

def calculate_energy(u, v, nx, ny, dx, dy):
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

def topological_charge(u, dx):
    u_x = np.zeros_like(u)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    topological_charge = (1 / (2 * np.pi)) * u_x[:, 1] * dx
    return np.sum(topological_charge)

if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from sys import argv

    L, T, nt, nx, ny, device = 5., 10., 400, 36, 36, 'cpu'
    solver = SineGordonIntegrator(L, T, nt, nx, ny, device)
    solver.evolve()
    X, Y = solver.X, solver.Y
    data = solver.u.detach().numpy() 
    assert len(argv) > 1
    animate = argv[1].lower() == 'true' or int(argv[1].lower()) == 1  
    
    if animate:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, data[-1], cmap='viridis',)
        plt.show()
        def update(frame):
            ax.clear()
            ax.plot_surface(X, Y, (data[frame]), cmap='viridis')
        fps = 300
        ani = FuncAnimation(fig, update, frames=solver.nt, interval=solver.nt / fps, )
        plt.show()
        
    else:
        es = []
        vs = []
        dx = dy = 2 * L / nx
        for i in range(1, solver.nt):
            u = data[i]
            v = (data[i] - data[i - 1]) / (solver.dt)
            vs.append(v)
            es.append(
                calculate_energy(u, v, nx, ny, dx, dy)
                )
        plt.plot(es)
        plt.title("Energy")
        plt.xlabel("T / [1]")
        plt.ylabel("E / [1]")
        plt.show()

        tc = []
        for i in range(0, solver.nt):
            u = data[i]
            tc.append(topological_charge(u, dx))
              
        plt.plot(solver.tn, tc,)
        plt.title("Topological Charge")
        plt.xlabel("T / [1]")
        plt.ylabel("")
        plt.show()
        vs = np.array(vs)
        

    with open('sv-ring-soliton.npy', 'wb') as f:
        np.save(f, data[:, 1:-1, 1:-1])
        np.save(f, vs[:, 1:-1, 1:-1])
    with open('sv-ring-soliton-tn.npy', 'wb') as f:
        np.save(f, solver.tn.detach().numpy()) 
