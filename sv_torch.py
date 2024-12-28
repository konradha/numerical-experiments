import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import diags
from scipy import sparse as sp


def u_x(u, Lx,):
    dx = 2 * Lx / (u.shape[0] - 1)
    du_x = torch.zeros_like(u) if isinstance(u, torch.Tensor) else np.zeros_like(u)
    du_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    du_x[0, :] = (u[1, :] - u[0, :]) / dx
    du_x[-1, :] = (u[-1, :] - u[-2, :]) / dx
    return du_x

def u_y(u, Ly):
    dy = 2 * Ly / (u.shape[1] - 1)
    du_y = torch.zeros_like(u) if isinstance(u, torch.Tensor) else np.zeros_like(u)
    du_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
    du_y[:, 0] = (u[:, 1] - u[:, 0]) / dy
    du_y[:, -1] = (u[:, -1] - u[:, -2]) / dy
    return du_y

def L_op(u, Lx, Ly):
    nx, ny = u.shape
    xn = torch.linspace(-Lx, Lx, nx) if isinstance(u, torch.Tensor) else np.linspace(-Lx, Lx, nx) 
    yn = torch.linspace(-Ly, Ly, ny) if isinstance(u, torch.Tensor) else np.linspace(-Ly, Ly, ny)
    ux = u_x(u, Lx)
    uy = u_y(u, Ly)
    return yn * ux  - xn * uy

def analytical_soliton_solution(X, Y, t):
    return 4 * np.arctan(np.exp(X + Y - t))

class SineGordonIntegrator:
    def __init__(self, Lx_min, Lx_max, Ly_min, Ly_max, T, nt, nx, ny, device='cuda',):
        self.device = torch.device(device)
        self.Lx_min, self.Lx_max = Lx_min, Lx_max
        self.Ly_min, self.Ly_max = Ly_min, Ly_max

        self.T = T
        self.nt = nt
        self.dt = T / nt
        self.nx = nx + 2
        self.ny = ny + 2

        self.dx = (self.Lx_max - self.Lx_min) / (nx + 1)
        self.dy = (self.Ly_max - self.Ly_min) / (ny + 1)

        self.tn = torch.linspace(0, T, nt)
        
        xn = torch.linspace(Lx_min, Lx_max, self.nx, device=self.device)
        yn = torch.linspace(Ly_min, Ly_max, self.ny, device=self.device)

        self.xn, self.yn = xn, yn

        self.xmin, self.xmax = Lx_min, Lx_max 
        self.ymin, self.ymax = Ly_min, Ly_max 

        self.u = torch.zeros((self.nt, self.nx, self.ny), device=self.device, dtype=torch.float64)
        self.v = torch.zeros((self.nt, self.nx, self.ny), device=self.device, dtype=torch.float64)

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

        # "static breather-like"
        omega = .6
        return 4 * torch.arctan(torch.sin(omega * x) / torch.cosh(omega * y))

        ## periodic lattice solitons
        #m = 25
        #n = m // 2
        #L = self.Lx_max / m ** 2 
        #u = 0
        #for i in range(m):
        #    for j in range(n):
        #        u += torch.arctan(torch.exp(x - n * L / np.pi))
        #for i in range(m):
        #    for j in range(n):
        #        u += torch.arctan(torch.exp(y - m * L / np.pi))
        #return u

        ## ring soliton
        ##R = 1.001
        ## stability assertion
        ##assert R > 1 and R ** 2 < 2 * (2 * self.L) ** 2
        #R = .5
        #return 4 * torch.arctan(((x - 5.) ** 2 + (y - 5.) ** 2 - R ** 2) / (2 * R))


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

        def karhunen_loeve_sample1(cov_func=None,): 
            def gaussian_covariance(points):
                length_scale = 2 * self.L / self.nx
                dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
                return np.exp(-dists**2 / (2 * length_scale**2))

            if cov_func is None: cov_func = gaussian_covariance 

            X, Y = self.X, self.Y
            points = np.vstack([X.ravel(), Y.ravel()]).T 
            cov_matrix = cov_func(points)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)   
            z = np.random.normal(size=(len(eigenvalues)))

            eps = 1
            x0, y0 = np.random.uniform(low=-self.L + eps, high=self.L - eps, size=(2,))
            width = .5
            mean = 4 * np.arctan(np.exp(-(points[:, 0] - x0) + (points[:, 1] - y0) / width))
            mean /= (points[:, 0] ** 2 + points[:, 1] ** 2) / 0.1 ** 2

            sample = mean + np.sum(np.sqrt(eigenvalues)[:, None] * eigenvectors.T * z, axis=0)
            sample = sample.reshape((self.nx, self.ny))
            from scipy import signal
            k = 6
            for i in range(k//2,k):
                sample = signal.wiener(sample, mysize=(i+1,i+1))

            return torch.tensor(sample)

        #def sample2(): 
        #    X, Y = self.X, self.Y
        #    N = np.random.randint(low=3, high=12,)
        #    Ai = np.random.beta(.5, .5, (N,))

        #    kix = np.random.uniform(low=-self.L, high=self.L, size=(N,))
        #    kiy = np.random.uniform(low=-self.L, high=self.L, size=(N,))
        #    
        #    u = np.zeros_like(X) 
        #    for i in range(N): 
        #        u += Ai[i] * (1./np.cosh(kix[i] * X)) * (1./np.cosh(kiy[i] * Y)) 
        #    return torch.tensor(u)

        #return sample2()

         
    def lapl(self, u):
        def u_yy(a):
            dy = abs(self.ymax - self.ymin) / (a.shape[1] - 1)
            uyy = torch.zeros_like(a) if isinstance(a, torch.Tensor) else np.zeros_like(a)
            uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
            return uyy

        def u_xx(a):
            dx = abs(self.xmax - self.xmin) / (a.shape[0] - 1)
            uxx = torch.zeros_like(a) if isinstance(a, torch.Tensor) else np.zeros_like(a)
            uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2)
            return uxx
        
        return u_xx(u) + u_yy(u)


    def stormer_verlet_step(self, u, v, dt, t, i):
        @staticmethod
        def boundary_x(x, Y, t):
            # x = -L or x = L
            x = float(x)
            t = t * torch.ones_like(Y)
            #print(type(x), f"{x=}")
            #print(type(Y), f"{Y=}")
            #print(type(t), f"{t=}")
            return 4 * torch.exp(x + Y + t) / (torch.exp(2 * t) + torch.exp(2 * x + 2 * Y))
    
        @staticmethod
        def boundary_y(X, y, t):
            # y = -L or y = L
            y = float(y)
            t = t * torch.ones_like(X)
            #print(type(X), f"{X=}")
            #print(type(y), f"{y=}")
            #print(type(t), f"{t=}")
            return 4 * torch.exp(X + y + t) / (torch.exp(2 * t) + torch.exp(2 * X + 2 * y))
     
    
        def apply_boundary_condition(u, v, t):
            dx = abs(self.xmax - self.xmin) / (self.nx - 1)
            dy = abs(self.ymax - self.ymin) / (self.ny - 1)
            dt = self.dt

            xm, xM = torch.tensor(self.xmin, dtype=torch.float64), torch.tensor(self.xmax, dtype=torch.float64)
            ym, yM = torch.tensor(self.ymin, dtype=torch.float64), torch.tensor(self.ymax, dtype=torch.float64)
   
            # u's ghost cells get approximation following boundary condition
            u[0,  1:-1] = u[1, 1:-1] - dx * boundary_x(xm, self.yn[1:-1], t)
            u[-1, 1:-1] = u[-2, 1:-1] + dx * boundary_x(xM, self.yn[1:-1], t)
            
            u[1:-1,  0] = u[1:-1, 1] - dy * boundary_y(self.xn[1:-1], ym, t)
            u[1:-1, -1] = u[1:-1, -2] + dy * boundary_y(self.xn[1:-1], yM, t)
            
            u[0, 0] = (u[1, 0] + u[0, 1])/2 - dx*boundary_x(xm, ym, t)/2 \
                      - dy*boundary_y(xm, ym, t)/2
            u[-1, 0] = (u[-2, 0] + u[-1, 1])/2 + dx*boundary_x(xM, ym, t)/2 \
                       - dy*boundary_y(xM, ym, t)/2
            u[0, -1] = (u[1, -1] + u[0, -2])/2 - dx*boundary_x(xm, yM, t)/2 \
                       + dy*boundary_y(xm, yM, t)/2
            u[-1, -1] = (u[-2, -1] + u[-1, -2])/2 + dx*boundary_x(xM, yM, t)/2 \
                        + dy*boundary_y(xM, yM, t)/2
            
            # v get the hard boundary condition
            v[0, 1:-1]  = boundary_x(xm, self.yn[1:-1], t)
            v[-1, 1:-1] = boundary_x(xM, self.yn[1:-1], t)
            v[1:-1, 0]  = boundary_y(self.xn[1:-1], ym, t)
            v[1:-1, -1] = boundary_y(self.xn[1:-1], yM, t)
    
    
    
        def f(x):
            # sine-Gordon
            return torch.sin(x)
            
            ## Klein-Gordon
            #return x + x ** 3

        # first step needs special treatment as we cannot yet use the pre-preceding
        if i == 1:
            v = torch.zeros_like(u)
            u_n = u + dt * v + 0.5 * dt ** 2 * (self.lapl(u) - f(u))
            self.apply_neumann_boundary(u_n, v)
            #apply_boundary_condition(u_n, v, self.dt)
            return u_n, v

        op = (self.lapl(u) - f(u)) 
        u_n = 2 * self.u[i - 1] - self.u[i - 2] + op * dt ** 2
        v_n = (u_n - self.u[i - 1]) / dt
        
        self.apply_neumann_boundary(u_n, v_n)
        #apply_boundary_condition(u_n, v_n, t)
        return u_n, v_n 

    def pseudo_stormer_verlet_step(self, u, v, dt, t, i):     
        def f(x):
            # sine-Gordon
            return torch.sin(x)
            
            ## Klein-Gordon
            #return x + x ** 3

        # first step needs special treatment as we cannot yet use the pre-preceding
        if i == 1:
            v = torch.zeros_like(u)
            u_n = u + dt * v + 0.5 * dt ** 2 * (self.lapl(u) - f(u))
            self.apply_neumann_boundary(u_n, v)
            #apply_boundary_condition(u_n, v, self.dt)
            return u_n, v

        op = (self.lapl(u) - f(u)) 
        v_n = v + dt * op
        u_n = u + dt * v_n
        self.apply_neumann_boundary(u_n, v_n)

        return u_n, v_n

    def step_stormer_verlet_conv2d(self):
        #def conv2d(u, Lmin, Lmax,):
        #    def get_stencil(nx, ny, dx, dy, device='cpu'):
        #        stencil = torch.zeros((1, 1, 3, 3), dtype=torch.float64, device=device)
        #        stencil[0, 0, 1, 0] = 1.0/dy**2
        #        stencil[0, 0, 0, 1] = 1.0/dx**2
        #        stencil[0, 0, 1, 1] = -2.0/dx**2 - 2.0/dy**2
        #        stencil[0, 0, 2, 1] = 1.0/dx**2
        #        stencil[0, 0, 1, 2] = 1.0/dy**2
        #        return stencil
        #    dx = abs(Lmax - Lmin) / (u.shape[0] - 1)
        #    dy = abs(Lmax - Lmin) / (u.shape[1] - 1)

        #    s = get_stencil(u.shape[0], u.shape[1], dx, dy)

        #    u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant')
        #    return F.conv2d(u_pad, s, padding=0).squeeze()

        def get_stencil(device='cpu'):
            stencil = torch.zeros((1, 1, 3, 3), dtype=torch.float64, device=device)
            stencil[0, 0, 1, 0] = 1.0/self.dy**2
            stencil[0, 0, 0, 1] = 1.0/self.dx**2
            stencil[0, 0, 1, 1] = -(2.0/self.dx**2 + 2.0/self.dy**2)
            stencil[0, 0, 2, 1] = 1.0/self.dx**2
            stencil[0, 0, 1, 2] = 1.0/self.dy**2
            return stencil

        def f(x):
            # sine-Gordon
            return torch.sin(x)
            
            ## Klein-Gordon
            #return x + x ** 3
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0) 
        self.u[0] = u0
        self.v[0] = v0
        s = get_stencil()
        dt = self.dt

        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float64)
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0) 
        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            if i == 1:
                v = torch.zeros_like(u0)
                u_n = u0 + dt * v + 0.5 * dt ** 2 * (self.lapl(u0) - f(u0))
                self.apply_neumann_boundary(u_n, v)
                continue
            u, v = self.u[i - 1], self.v[i - 1]
            
            # this loop is still broken but once corrected might become much better than
            # the hand-rolled global convolution!
            u_padded = F.pad(u[1:-1, 1:-1].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            un = u.clone()
            un[1:-1, 1:-1] = F.conv2d(u_padded, laplacian_kernel).squeeze(0).squeeze(0)
            un = un - torch.sin(u)
            un = 2 * u - self.u[i - 2] + dt * un
             
            self.apply_neumann_boundary(un, un) 
            self.u[i] = un
            vn = (self.u[i] - self.u[i-1]) / dt
            self.v[i] = vn
             



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

    def evolve_pseudo(self):
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0) 
        self.u[0] = u0
        self.v[0] = v0
        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            self.u[i], self.v[i] = self.pseudo_stormer_verlet_step(
                self.u[i-1], self.v[i-1], self.dt, t, i)

         

    def evolve(self):
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0
    
        dt = self.dt

        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            self.u[i], self.v[i] = self.stormer_verlet_step(
                self.u[i-1], self.v[i-1], self.dt, t, i)
        for i in range(1, self.nt - 1):
            self.v[i] = (self.u[i+1] - self.u[i-1]) / (2 * dt)
        self.v[-1] = self.v[-2]

    def evolve_energy_conserving(self):
        # might be off due to the nature of the FDM-no-matrix-operators
        # formulation :(
        def f(x):
            return torch.sin(x)
        # Marazzato et al.
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0

        dt = self.dt

        # follow paper implementation
        vhalf = v0 #+ .5 * dt * (self.lapl(u0) - torch.sin(u0))

        def free_flight(u, t, tf, vhalf):
            un = u + (tf - t) * vhalf
            return self.lapl(un) - torch.sin(un)

        from scipy import special
        def integrate(u, ti, tf, vhalf, n=10):
            # Gauss-Legendre does not yet do the trick unfortunately
            vals = torch.zeros(n, u.shape[0], u.shape[1])
            dtt = (tf - ti) / n # != dt in usual program! 
            qp, weights = special.roots_legendre(n)
            weights /= weights.sum()
            for k in range(n):
                vals[k] = weights[k] * free_flight(u, ti + dtt * qp[k], tf, vhalf) 
            return torch.sum(vals, axis=0)

        def integrate_exp(u, ti, tf, vhalf, n=10): 
            vals = torch.zeros(n, u.shape[0], u.shape[1])
            dtt = (tf - ti) / n # != dt in usual program! 
            xn = torch.linspace(0, 1, n)
            weights = torch.exp(-.1 * xn ** 2)
            for k in range(n):
                vals[k] = weights[k] * free_flight(u, ti + dtt * xn[k], tf, vhalf) 
            return torch.sum(vals, axis=0)

        def integrate_mid(u, ti, tf, vhalf):
            vals = torch.zeros(2, u.shape[0], u.shape[1])
            dtt = (tf + ti) / 2
            force = free_flight(u, ti, dtt, vhalf) 
            return dtt * force
        
        for i, t in (enumerate(tqdm(self.tn))):
            if i == 0: continue # we already initialized u0, v0
            un = self.u[i-1] 
            self.u[i] = un + dt * vhalf

            vhalf = vhalf - 2 * integrate_mid(un, 0, 2 * dt, vhalf) 


        for i in range(1, self.nt - 1):
            self.v[i] = (self.u[i+1] - self.u[i-1]) / (2 * dt) 
        self.v[-1] = self.v[-2]

    def evolve_rk4(self,):
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0
    
        dt = self.dt

        def f(x):
            return self.lapl(x) - torch.sin(x)

        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            u, v = self.u[i - 1], self.v[i - 1]

            k1_v = dt * f(u) 
            k2_v = dt * f(u + 0.5 * dt * v)  
            k3_v = dt * f(u + 0.5 * dt * (v + 0.5 * k1_v)) 
            k4_v = dt * f(u + dt * (v + k3_v)) 
            
            k1_u = dt * v
            k2_u = dt * (v + 0.5 * k1_v)
            k3_u = dt * (v + 0.5 * k2_v)
            k4_u = dt * (v + k3_v)

            un = u + (1/6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
            vn = v + (1/6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

            self.u[i] = un
            self.v[i] = vn

        
    def stormer_verlet_filter(self, u, v, dt, t, i, omega):
        def f(u):
            return torch.sin(u)

        if i == 1:
            v = torch.zeros_like(u)
            u_n = u + dt * v + 0.5 * dt ** 2 * (self.lapl(u) - f(u))
            self.apply_neumann_boundary(u_n, v)
            return u_n, v

        cos_hw = torch.cos(self.dt * omega,)
        sinc_hw = torch.sinc(self.dt * omega / np.pi,)
 
        op = (self.lapl(u) - f(u)) 
        u_n = 2 * self.u[i - 1] - self.u[i - 2] + op * dt ** 2

        v_n = (u_n - self.u[i - 2]) / (2 * dt) 
        self.apply_neumann_boundary(u_n, v_n)
        return u_n, v_n


    def evolve_filter(self):
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0

        jx = np.fft.fftfreq(self.nx, self.dx)
        jy = np.fft.fftfreq(self.ny, self.dy)
        JX, JY = np.meshgrid(jx, jy)
        omega = torch.tensor(np.sqrt(JX**2 + JY**2), dtype=torch.float64)

        for i, t in enumerate(tqdm(self.tn)):
            if i == 0: continue # we already initialized u0, v0
            self.u[i], self.v[i] = self.stormer_verlet_filter(
                self.u[i-1], self.v[i-1], self.dt, t, i, omega)

    def evolve_ETD1_sparse(self):
        # ETD1
        show_matrix_structure = False
        def f(x):
            if isinstance(x, np.ndarray):
                # sine-Gordon
                return -np.sin(x)

                ## Klein-Gordon
                #return x + x ** 3
            elif isinstance(x, torch.Tensor):
                # sine-Gordon
                return -torch.sin(x)

                ## Klein-Gordon
                #return x + x ** 3
            else:
                raise Exception

        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0
        
        nx = self.nx - 2 
        ny = self.ny - 2
        dt = self.dt
        def textbook_laplacian(nx, ny,):
            assert nx == ny
            # have only thought about square domain for now
            assert self.Lx_min == self.Ly_min
            assert self.Lx_max == self.Ly_max
            dx = (self.Lx_max - self.Lx_min) / (nx + 1)

            middle_diag = -4 * np.ones((nx + 2))
            middle_diag[0] = middle_diag[-1] = -3
            left_upper_diag = lower_right_diag = middle_diag + np.ones((nx + 2))

            diag = np.concat([left_upper_diag, *(nx * [middle_diag]), lower_right_diag])
            offdiag_pos = np.ones((nx + 2) * (nx + 2) - 1)

            inner_outer_identity = np.ones((nx + 2) * (nx + 2) - (nx + 2))

            full = diags(
                    [diag, offdiag_pos, offdiag_pos, inner_outer_identity, inner_outer_identity],
                    [0, -1, 1, nx+2, -nx-2], shape=((nx + 2) ** 2, (nx + 2) ** 2)
                    )

            return ((1/dx) ** 2 * full).tocsc()
           
        self.Lap = textbook_laplacian(self.nx - 2, self.ny - 2)

            
        from scipy.linalg import expm
        from numpy.linalg import cond
        torch_lapl = torch.tensor(self.Lap.todense(), dtype=torch.float64)
        exp_L_dt = torch.tensor(expm(self.Lap.todense() * dt), dtype=torch.float64)
        Id = torch.eye(exp_L_dt.shape[0], dtype=torch.float64)
        torch_lapl_inv = torch.tensor(sp.linalg.inv(self.Lap).todense(), dtype=torch.float64)

        nz_mask = exp_L_dt != 0.
        assert np.abs(cond(exp_L_dt.detach().numpy())) < 1e6
        
        zeros = torch.zeros_like(Id) 
        L = torch.cat([
                            torch.cat([zeros,         Id], dim=1),
                            torch.cat([torch_lapl, zeros], dim=1),
                        ], dim=0)
        L_inv = torch.cat([
                            torch.cat([zeros, torch_lapl_inv], dim=1),
                            torch.cat([Id,             zeros], dim=1),
                            ], dim=0)
        assert torch.allclose(L @ L_inv, torch.eye(L.shape[0], dtype=torch.float64))

        small_id = diags([np.ones((nx + 2) ** 2)], [0,], shape=((nx + 2) ** 2, (nx + 2) ** 2)).tocsr()

        L_top_row    = sp.hstack([sp.csr_matrix(small_id.shape), small_id]) 
        L_bottom_row = sp.hstack([self.Lap, sp.csr_matrix(small_id.shape)])
        L = sp.vstack([L_top_row, L_bottom_row]).tocsc()
        Id_sparse = diags( [np.ones((L.shape[0]),)], [0,], shape=(L.shape)).tocsc()
        L_inv = sp.linalg.spsolve(L, Id_sparse).tocsc()
 
        expon = torch.tensor(sp.linalg.expm(self.dt * L).todense(), dtype=torch.float64)
        #propagator = torch.tensor(L_inv.todense()) @ (expon - torch.tensor(Id_sparse.todense()))
        #propagator_x = L_inv @ expon @ x - L_inv @ x
        #propagator_x = L_inv @ (expon @ x - x)

        def expon_mult(x, dt=dt, L=L):
            return (sp.linalg.expm_multiply(dt * L, x))

        self.u = self.u.cpu().numpy() 
        self.v = self.v.cpu().numpy()
        # dense MMM loop
        for i in tqdm(range(1, self.nt)):
            u, v = self.u[i - 1].ravel(), self.v[i - 1].ravel()
            uv    = np.concat([u, v],)
            gamma = np.concat([np.zeros_like(u), -np.sin(u)],)
            u_vec = expon_mult(uv) + L_inv @ (expon_mult(gamma) - gamma)

            un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
            vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
            self.apply_neumann_boundary(un, vn)

            self.u[i], self.v[i] = un, vn

        #self.u = self.u.cpu().numpy() 
        #self.v = self.v.cpu().numpy()

        ### _really_ crude approximation; not at all useful long-term
        ##expon = diags([
        ##    expon.diagonal(0), expon.diagonal(-1), expon.diagonal(1)
        ##    ], [0, -1, 1]).tocsc()

        ##propagator = diags([
        ##    propagator.diagonal(0), propagator.diagonal(-1), propagator.diagonal(1)
        ##    ], [0, -1, 1]).tocsc()

        #for i in tqdm(range(1, self.nt)):
        #    u, v = self.u[i - 1].ravel(), self.v[i - 1].ravel()
        #    uv    = np.concat([u, v],)
        #    gamma = np.concat([np.zeros_like(u), f(u)],)

        #    u_vec = expon.dot(uv) + propagator.dot(gamma)

        #    un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        #    vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        #    self.apply_neumann_boundary(un, vn)

        #    self.u[i], self.v[i] = un, vn

    def evolve_eec(self):
        # this one diverges, needs fixing
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0) 
        self.u[0] = u0
        self.v[0] = v0

        psi = torch.zeros_like(u0)
        dt = self.dt
        for i in tqdm(range(self.nt)):
            if i == 0: continue
            u, v = self.u[i-1], self.v[i-1]
            g = torch.sin(u)
            u_next = u + dt * v
            psi_next = psi + 0.5 * g * (u_next - u)
            v_next = v - 0.5 * dt * (g * (psi_next + psi) - self.lapl(u))
            self.apply_neumann_boundary(u_next, v_next)
            self.u[i], self.v[i] = u_next, v_next
            psi = psi_next 

    def evolve_symplectic_euler(self):
        # this one diverges, needs fixing
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0) 
        self.u[0] = u0
        self.v[0] = v0

        dt = self.dt
        for i in tqdm(range(self.nt)):
            if i == 0: continue
            u, v = self.u[i-1], self.v[i-1]
            un = u + dt * v
            vn = v + dt * (self.lapl(un) - torch.sin(un)) 
            self.apply_neumann_boundary(un, vn)
            self.u[i], self.v[i] = un, vn

    def evolve_gl(self):
        # this one diverges, needs fixing
        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0) 
        self.u[0] = u0
        self.v[0] = v0

        dt = self.dt
        b1 = b2 = .5

        a11 = a22 = .25
        a12 = a21 = .25 - np.sqrt(3) / 6

        def fixed_point_solve(u, v, dt, stencil=self.lapl, tol=1e-10, max_iter=100): 
            device = u.device
            nx, ny = u.shape

            a11 = a22 = 0.25
            a12 = a21 = 0.25 - (3.0**0.5)/6
            b1 = b2 = 0.5

            k1u = v.clone()
            k1v = self.lapl(u) - torch.sin(u)
            k2u = v.clone()
            k2v = k1v.clone()

            # this fix point iteration does not take into account the 
            # boundary conditions nicely
            for i in range(max_iter):
                k1u_old = k1u.clone()
                k1v_old = k1v.clone()
                k2u_old = k2u.clone()
                k2v_old = k2v.clone()
                u1 = u + dt * (a11 * k1u + a12 * k2u)
                u2 = u + dt * (a21 * k1u + a22 * k2u)
                k1v = self.lapl(u1) - torch.sin(u1)
                k2v = self.lapl(u2) - torch.sin(u2)
                self.apply_neumann_boundary(k1v, k2v)
                k1u = v + dt * (a11 * k1v + a12 * k2v)
                k2u = v + dt * (a21 * k1v + a22 * k2v)
                err = np.sum([
                    torch.sum(torch.abs(k1u - k1u_old)**2),
                    torch.sum(torch.abs(k1v - k1v_old)**2),
                    torch.sum(torch.abs(k2u - k2u_old)**2),
                    torch.sum(torch.abs(k2v - k2v_old)**2)
                    ])

                if err < tol:
                    break
            u_new = u + dt * (b1 * k1u + b2 * k2u)
            v_new = v + dt * (b1 * k1v + b2 * k2v)
            return u_new, v_new

        def fixed_point_matrix_free(u, v, dt, tol=1e-7, max_iter=10, krylov_tol=1e-6, krylov_max_iter=10):
            device = u.device
            nx, ny = u.shape
            a11 = a22 = 1/4
            a12 = a21 = 1/4 - torch.sqrt(torch.tensor(3.0)) / 6
            b1 = b2 = 1/2
            k1u = v.clone()
            k1v = self.lapl(u) - torch.sin(u)
            k2u = k1u.clone()
            k2v = k1v.clone()

            def error(k1u, k2u, k1v, k2v):
                u1 = u + dt * (a11 * k1u + a12 * k2u)
                u2 = u + dt * (a21 * k1u + a22 * k2u)

                err_k1u = k1u - (v + dt * (a11 * k1v + a12 * k2v))
                err_k2u = k2u - (v + dt * (a21 * k1v + a22 * k2v))
                err_k1v = k1v - (self.lapl(u1) - torch.sin(u1))
                err_k2v = k2v - (self.lapl(u2) - torch.sin(u2))

                return torch.cat([err_k1u.flatten(), err_k2u.flatten(), 
                                  err_k1v.flatten(), err_k2v.flatten()])

            def jacobian_action(v_vec, k1u, k2u, k1v, k2v):
                d = nx * ny
                v1u = v_vec[:d].view(nx, ny)
                v2u = v_vec[d:2*d].view(nx, ny)
                v1v = v_vec[2*d:3*d].view(nx, ny)
                v2v = v_vec[3*d:].view(nx, ny)

                u1 = u + dt * (a11 * k1u + a12 * k2u)
                u2 = u + dt * (a21 * k1u + a22 * k2u)

                lap_v1u = self.lapl(v1u)
                lap_v2u = self.lapl(v2u)

                jv1u = v1u - dt * (a11 * v1v + a12 * v2v)
                jv2u = v2u - dt * (a21 * v1v + a22 * v2v)
                jv1v = v1v - (lap_v1u - torch.cos(u1) * v1u)
                jv2v = v2v - (lap_v2u - torch.cos(u2) * v2u)

                return torch.cat([jv1u.flatten(), jv2u.flatten(), 
                                  jv1v.flatten(), jv2v.flatten()])

            for iteration in range(max_iter):
                er = error(k1u, k2u, k1v, k2v)
                err_norm = torch.norm(er)

                if err_norm < tol:
                    break

                def matvec(v):
                    return jacobian_action(v, k1u, k2u, k1v, k2v)
                delta, _ = conjugate_gradient(matvec, -er, krylov_tol, krylov_max_iter)
                delta_k1u = delta[:nx * ny].view(nx, ny)
                delta_k2u = delta[nx * ny:2 * nx * ny].view(nx, ny)
                delta_k1v = delta[2 * nx * ny:3 * nx * ny].view(nx, ny)
                delta_k2v = delta[3 * nx * ny:].view(nx, ny)
                k1u = k1u + delta_k1u
                k2u = k2u + delta_k2u
                k1v = k1v + delta_k1v
                k2v = k2v + delta_k2v

            if err_norm > tol:
                raise RuntimeError(f"Newton iteration did not converge in {max_iter} iterations.")
            u_new = u + dt * (b1 * k1u + b2 * k2u)
            v_new = v + dt * (b1 * k1v + b2 * k2v)
            return u_new, v_new

        def conjugate_gradient(matvec, b, tol, max_iter):
            x = torch.zeros_like(b)
            r = b.clone()
            p = r.clone()
            rsold = torch.dot(r, r)
            for i in (range(max_iter)):
                Ap = matvec(p)
                alpha = rsold / torch.dot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = torch.dot(r, r)
                if torch.sqrt(rsnew) < tol:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
            return x, i

        e0 = calculate_energy(u0.cpu().numpy(), v0.cpu().numpy(), self.nx, self.ny, self.dx, self.dy)
        for i in tqdm(range(self.nt)):
            if i == 0: continue
            u, v = self.u[i-1], self.v[i-1]

            un, vn = fixed_point_solve(u, v, dt) 
            #un, vn = fixed_point_matrix_free(u, v, dt)

            self.apply_neumann_boundary(un, vn)
            self.u[i], self.v[i] = un, vn
            E = calculate_energy(un.cpu().numpy(), vn.cpu().numpy(), self.nx-2, self.ny-2, self.dx, self.dy) 
            if E <= .5 * e0 or E >= 2 * e0:
                print(f"Aborting time stepping at {i}")
                self.u[i:] = self.v[i:] = torch.nan
                break


    def evolve_ETD1(self):
        # ETD1
        show_matrix_structure = False
        def f(x):
            # sine-Gordon
            return -torch.sin(x)

            ## Klein-Gordon
            #return x + x ** 3

        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0
        
        dt = self.dt
        def textbook_laplacian(nx, ny,):
            assert nx == ny
            # have only thought about square domain for now
            assert self.Lx_min == self.Ly_min
            assert self.Lx_max == self.Ly_max
            dx = (self.Lx_max - self.Lx_min) / (nx + 1)

            middle_diag = -4 * np.ones((nx + 2))
            middle_diag[0] = middle_diag[-1] = -3
            left_upper_diag = lower_right_diag = middle_diag + np.ones((nx + 2))

            diag = np.concat([left_upper_diag, *(nx * [middle_diag]), lower_right_diag])
            offdiag_pos = np.ones((nx + 2) * (nx + 2) - 1)

            inner_outer_identity = np.ones((nx + 2) * (nx + 2) - (nx + 2))

            full = diags(
                    [diag, offdiag_pos, offdiag_pos, inner_outer_identity, inner_outer_identity],
                    [0, -1, 1, nx+2, -nx-2], shape=((nx + 2) ** 2, (nx + 2) ** 2)
                    )

            return ((1/dx) ** 2 * full).tocsc()
            
            #middle_diag = -2 * np.ones((nx + 2))
            #offdiag = np.ones((nx + 1))
            #full = diags(
            #        [middle_diag, offdiag, offdiag],
            #        [0, -1, 1,], shape=((nx + 2) ** 2, (nx + 2) ** 2)
            #        )
            #return full


            
        #self.Lap = neumann_laplacian(self.nx - 2, self.ny - 2, self.L, self.L)
        self.Lap = textbook_laplacian(self.nx - 2, self.ny - 2)

        #eigenvalues, _ = scipy.sparse.linalg.eigs(self.Lap)
        #print(eigenvalues)
        #raise Exception
            
        from scipy.linalg import expm
        from numpy.linalg import cond
        torch_lapl = torch.tensor(self.Lap.todense(), dtype=torch.float64)
        exp_L_dt = torch.tensor(expm(self.Lap.todense() * dt), dtype=torch.float64)
        Id = torch.eye(exp_L_dt.shape[0], dtype=torch.float64)
        torch_lapl_inv = torch.tensor(sp.linalg.inv(self.Lap).todense(), dtype=torch.float64)

        nz_mask = exp_L_dt != 0.
        assert np.abs(cond(exp_L_dt.detach().numpy())) < 1e6
        
        zeros = torch.zeros_like(Id) 
        L = torch.cat([
                            torch.cat([zeros,         Id], dim=1),
                            torch.cat([torch_lapl, zeros], dim=1),
                        ], dim=0)
        L_inv = torch.cat([
                            torch.cat([zeros, torch_lapl_inv], dim=1),
                            torch.cat([Id,             zeros], dim=1),
                            ], dim=0)
        assert torch.allclose(L @ L_inv, torch.eye(L.shape[0], dtype=torch.float64))

        expon = torch.tensor(
                expm(self.dt * L.detach().numpy()),
                dtype=torch.float64
                )

        #eigenvalues, eigenvectors = eigs(L., k=2)
        #D = np.diag(eigenvalues)
        #exp_tD = np.diag(np.exp(dt * eigenvalues))
        #expon = eigenvectors @ exp_tD @ np.linalg.inv(eigenvectors)

        propagator = L_inv @ (expon - torch.eye(expon.shape[0], dtype=torch.float64))

        if show_matrix_structure:
            plt.matshow(expon)
            plt.title("expon")
            plt.show()

            plt.matshow(L)
            plt.title("L")
            plt.show()

            plt.matshow(L_inv)
            plt.title("L_inv")
            plt.show()

            plt.matshow(propagator)
            plt.title("propagator")
            plt.show()
 
        for i in tqdm(range(1, self.nt)):
            u, v = self.u[i - 1].ravel(), self.v[i - 1].ravel()

            uv    = torch.cat([u, v], dim=0)
            gamma = torch.cat([torch.zeros_like(u), f(u)],)

            u_vec = expon @ uv + propagator @ gamma

            un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
            vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
            self.apply_neumann_boundary(un, vn)

            self.u[i], self.v[i] = un, vn

    def evolve_ETD2(self):
        # ETD2
        show_matrix_structure = False
        def f(x):
            # sine-Gordon
            return -torch.sin(x)

            ## Klein-Gordon
            #return x + x ** 3

        u0 = self.initial_u_grf(self.X, self.Y)
        v0 = torch.zeros_like(u0)
    
        self.u[0] = u0
        self.v[0] = v0
        
        dt = self.dt
        def textbook_laplacian(nx, ny,):
            assert nx == ny
            # have only thought about square domain for now
            assert self.Lx_min == self.Ly_min
            assert self.Lx_max == self.Ly_max
            dx = (self.Lx_max - self.Lx_min) / (nx + 1)

            middle_diag = -4 * np.ones((nx + 2))
            middle_diag[0] = middle_diag[-1] = -3
            left_upper_diag = lower_right_diag = middle_diag + np.ones((nx + 2))

            diag = np.concat([left_upper_diag, *(nx * [middle_diag]), lower_right_diag])
            offdiag_pos = np.ones((nx + 2) * (nx + 2) - 1)

            inner_outer_identity = np.ones((nx + 2) * (nx + 2) - (nx + 2))

            full = diags(
                    [diag, offdiag_pos, offdiag_pos, inner_outer_identity, inner_outer_identity],
                    [0, -1, 1, nx+2, -nx-2], shape=((nx + 2) ** 2, (nx + 2) ** 2)
                    )

            return ((1/dx) ** 2 * full).tocsc()
           
        self.Lap = textbook_laplacian(self.nx - 2, self.ny - 2)

        from scipy.linalg import expm
        from numpy.linalg import cond
        torch_lapl = torch.tensor(self.Lap.todense(), dtype=torch.float64)
        exp_L_dt = torch.tensor(expm(self.Lap.todense() * dt), dtype=torch.float64)
        Id = torch.eye(exp_L_dt.shape[0], dtype=torch.float64)
        torch_lapl_inv = torch.tensor(sp.linalg.inv(self.Lap).todense(), dtype=torch.float64)

        nz_mask = exp_L_dt != 0.
        assert np.abs(cond(exp_L_dt.detach().numpy())) < 1e6
        
        zeros = torch.zeros_like(Id) 
        L = torch.cat([
                            torch.cat([zeros,         Id], dim=1),
                            torch.cat([torch_lapl, zeros], dim=1),
                        ], dim=0)
        L_inv = torch.cat([
                            torch.cat([zeros, torch_lapl_inv], dim=1),
                            torch.cat([Id,             zeros], dim=1),
                            ], dim=0)
        assert torch.allclose(L @ L_inv, torch.eye(L.shape[0], dtype=torch.float64))

        expon = torch.tensor(
                expm(self.dt * L.detach().numpy()),
                dtype=torch.float64
                )


        # ETD-1 propagatro
        propagator = L_inv @ (expon - torch.eye(expon.shape[0], dtype=torch.float64))
        
        # define ETD-2 propagator
        # that's still wrong!
        propagator2 =  L_inv @ L_inv @ (
                expon - torch.eye(expon.shape[0], dtype=torch.float64) - dt * L)
         
        gamma_prev = torch.cat([torch.zeros_like(self.u[0]), f(self.u[0])],) 
        for i in tqdm(range(1, self.nt)):
            u, v = self.u[i - 1].ravel(), self.v[i - 1].ravel()

            uv    = torch.cat([u, v], dim=0)
            gamma = torch.cat([torch.zeros_like(u), f(u)],)

            # this matmul is unnecessarily dense. We can make ETD methods much faster
            # TODO use sparse matrices actually
            u_vec = expon @ uv + propagator @ gamma

            un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
            vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))

            if i > 1:
                u_vec += propagator2 @ (gamma - gamma_prev)

            gamma_prev = gamma
            un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
            vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
            self.apply_neumann_boundary(un, vn)
            self.u[i], self.v[i] = un, vn
                       

def calculate_energy(u, v, nx, ny, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
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

def compare_energy_all():
    L, T, nt, nx, ny, device = 10, 10., 200, 64, 64,'cpu'
    dx = 2 * L / (nx + 1)
    dy = 2 * L / (ny + 1)
    assert (T / nt) / ((L) ** 2 / (dx * dy)) < 1 
    dt = T / nt
    solver = SineGordonIntegrator(-L, L, -L, L, T, nt, nx, ny, device)

    print(f"Allocating {2 * 3 * np.prod(solver.u.shape) / (1<<30)=:.2f} GB") 

    def read_data(solver):
        data_u = solver.u.cpu().numpy() if isinstance(solver.u, torch.Tensor) else solver.u
        data_v = solver.v.cpu().numpy() if isinstance(solver.v, torch.Tensor) else solver.v
        return data_u, data_v

    solver.evolve()
    u, v = read_data(solver)
    solver.u = torch.zeros_like(torch.tensor(u))
    solver.v = torch.zeros_like(torch.tensor(v))
    solver.evolve_rk4()
    u_rk4, v_rk4 = read_data(solver)
    solver.u = torch.zeros_like(torch.tensor(u))
    solver.v = torch.zeros_like(torch.tensor(v))

    #solver.evolve_ETD1_sparse()
    #u_etd1, v_etd1 = read_data(solver)
    #solver.u = torch.zeros_like(torch.tensor(u))
    #solver.v = torch.zeros_like(torch.tensor(v))

    solver.step_stormer_verlet_conv2d()
    u_ps, v_ps = read_data(solver)
    solver.u = torch.zeros_like(torch.tensor(u))
    solver.v = torch.zeros_like(torch.tensor(v))

    us = np.array([u, u_rk4, u_ps])
    vs = np.array([v, v_rk4, v_ps])

    for i, nme in enumerate(["Stormer-Verlet", "RK-4", "Stormer-Verlet Conv2d"]):
        es = [calculate_energy(u, v, nx, ny, dx, dy) for u, v in zip(us[i], vs[i])]
        plt.plot(np.linspace(0, solver.T, nt), np.array(es), label=nme) 
    plt.legend()
    plt.show()

if __name__ == None:
    compare_energy_all()



if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from sys import argv

    L, T, nt, nx, ny, device = 12, 1., 50, 32, 32,'cpu'
    dx = 2 * L / (nx + 1)
    dy = 2 * L / (ny + 1)
    assert (T / nt) / ((L) ** 2 / (dx * dy)) < 1 
    dt = T / nt

    # initialize with square domain
    solver = SineGordonIntegrator(-L, L, -L, L, T, nt, nx, ny, device)
   
    solver.evolve_gl()
    #solver.evolve_rk4()
    #solver.evolve_filter()
    #solver.evolve_ETD1_sparse() 
    #solver.evolve_energy_conserving()
    #solver.evolve_gautschi_lf()
    X, Y = solver.X.cpu().numpy(), solver.Y.cpu().numpy()
    data = solver.u.cpu().numpy() if isinstance(solver.u, torch.Tensor) else solver.u
 
    assert len(argv) > 1
    animate = argv[1].lower() == 'true' or int(argv[1].lower()) == 1  
    
    if animate:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
        def update(frame):
            ax.clear()

            ax.plot_surface(X, Y,
                    data[frame],
                    #np.abs(data[frame] - analytical_soliton_solution(X, Y, (frame) * (T/nt))),
                    #analytical_soliton_solution(X, Y, frame * (T/nt)),
                    cmap='viridis')
            ax.set_title(f"t={(dt * frame):.2f}")
        fps = 300
        ani = FuncAnimation(fig, update, frames=solver.nt, interval=solver.nt / fps, )
        plt.show()
        
    else:
        
        es = []
        vs = []
        dx = dy = L / nx
        for i in range(1, solver.nt):
            u = data[i]
            v = (data[i] - data[i - 1]) / (solver.dt)
            vs.append(v)
            es.append(
                calculate_energy(u, v, nx, ny, dx, dy)
                )
        plt.plot(np.linspace(0, T, len(es)), es)
        plt.title("Energy")
        plt.xlabel("T / [1]")
        plt.ylabel("E / [1]")
        plt.show()

        #tc = []
        #for i in range(0, solver.nt):
        #    u = data[i]
        #    tc.append(topological_charge(u, dx))
        #      
        #plt.plot(solver.tn, tc,)
        #plt.title("Topological Charge")
        #plt.xlabel("T / [1]")
        #plt.ylabel("")
        #plt.show()
        #vs = np.array(vs)
        

        ## data saved when calling without animation
        #with open('sv-analytical-soliton.npy', 'wb') as f:
        #    np.save(f, data[:, 1:-1, 1:-1])
        #    np.save(f, vs[:, 1:-1, 1:-1])
        ##with open('sv-ring-soliton-tn.npy', 'wb') as f:
        ##    np.save(f, solver.tn.detach().numpy()) 
