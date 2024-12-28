import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Callable
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch.fft as fft


#@torch.jit.script
def u_yy(a, dy):
    uyy = torch.zeros_like(a)
    uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] -
                       2 * a[1:-1, 1:-1]) / (dy ** 2)
    return uyy

#@torch.jit.script
def u_xx(a, dx):
    uxx = torch.zeros_like(a)
    uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] -
                       2 * a[1:-1, 1:-1]) / (dx ** 2)
    return uxx

# torch eager mode might be perfectly well-suited here
# and fusing by hand does most of the work!
@torch.jit.script
def u_xx_yy(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1, 1:-1] = (
        (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2) +
        (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
    )
    return uxx_yy


@dataclass
class SolitonParameters:
    velocity: torch.Tensor
    position: torch.Tensor
    amplitude: torch.Tensor
    phase: torch.Tensor

def ring_soliton(X, Y, xc, yc, R):
    return 4 * torch.arctan(((X - xc) ** 2 + (X - yc) ** 2 - R ** 2) / (2 * R))

def ring_soliton_center(X, Y):
    return ring_soliton(X, Y, 0, 0, 1)

def static_breather(X, Y):
    omega = .6
    return 4 * torch.arctan(torch.sin(omega * X) / torch.cosh(omega * Y))

def zero_velocity(X, Y):
    return torch.zeros_like(X)


def solutions_general(lam1, lam2, theta1, theta2):
    return 4 * torch.arctan(
            ((lam1 + lam2) / (lam1 - lam2)) * (
                torch.sinh(theta1 - theta2) / torch.cosh(theta1 - theta2)
                )
            )

def solutions1(x, y):
    lam1 = .9
    A1 = 1
    A2 = B1 = 1j
    def f(x, y):
        return x + y

    return torch.abs(
            -2 * 1j * torch.log(
                (A1 * torch.exp(2 * lam1 * y + 2 * f(x, y)) - B1) /\
                (A1 * torch.exp(2 * lam1 * y + 2 * f(x, y)) + B1) 
                ))

def solutions_first(x, y):
    lam1 = 2
    A1 = 1
    A2 = B1 = 1j
    def f(x, y):
        #return ((x + y) - (x + y) ** 3 ) * .5
        return ((x + y) **2)


    r =  torch.abs(
             -2 * 1j * torch.log(
                (A1 * torch.exp(2 * lam1 * y + 2 * f(x, y)) - B1) /\
                (A1 * torch.exp(2 * lam1 * y + 2 * f(x, y)) + B1) 
                ))

    r = torch.where(torch.isnan(r), torch.tensor(0., dtype=r.dtype), r)
    return r


def solutions_more(x, y):
    lam1 = 2
    lam2 = .5
    A1 = 1
    A2 = B1 = 1j

    def f1(x, y):
        return ((x + y) ** 3 ) * .5

    def f2(x, y):
        return ((x + y) ** 3 ) * .5 / (x + y)

    theta1 = lam1 * y + f1(x, y)
    theta2 = lam2 * y + f2(x, y)

    r = solutions_general(lam1, lam2, theta1, theta2) 
    return torch.where(torch.isnan(r), torch.tensor(0., dtype=r.dtype), r)


def two_kink_interaction(x, y, x1=0, y1=0, v1=0.3, x2=5, y2=0, v2=-0.4):
    s1 = 4 * torch.atan(torch.exp(-(x - x1 - v1 * y) /\
            torch.sqrt(torch.tensor(1 - v1**2))))
    s2 = 4 * torch.atan(torch.exp(-(x - x2 - v2 * y) /\
            torch.sqrt(torch.tensor(1 - v2**2))))
    s = s1 + s2
    return s

def bubble(x, y):
    # example 1
    #B = .1
    #sigma = 15

    B = .27
    sigma = 25
    return 2 * (B ** 2 - 1) * (torch.sinh(B * x) / (torch.cosh(B * x))) * \
            torch.exp(-y ** 2 / sigma ** 2)

def kink(x, y):
    return 4 * torch.atan(torch.exp( x + y ))

class SineGordonIntegrator(torch.nn.Module):
    """
    Finite Differences solver for the sine-Gordon equation on a rectangular domain.
    Optimized to run on both CPU and GPU.
    """

    def __init__(self,
                 Lx_min, Lx_max, Ly_min, Ly_max,
                 nx, ny,
                 T, nt,
                 initial_u: Callable,
                 initial_v: Callable,
                 c2=1,  # can also be a function c2 = c2(x, y)
                 m=1,  # can also be a function m = m(x, y)
                 enable_energy_control=False,
                 snapshot_frequency=10,
                 step_method='RK4',
                 boundary_conditions='homogeneous-neumann',
                 device='cuda', dtype=torch.float64,):
        super().__init__()
        implemented_methods = {
            'stormer-verlet-pseudo': self.stormer_verlet_pseudo_step,
            'gauss-legendre': self.gauss_legendre_step,
            'RK4': self.rk4_step,
            'ETD1-sparse': None,
            'ETD1-krylov': None,
        }
        save_last_k = {
            'stormer-verlet-pseudo': 0,
            'gauss-legendre': 0,
            'RK4': 0,
        }

        implemented_boundary_conditions = {
            'homogeneous-neumann': self.neumann_bc}

        if step_method not in implemented_methods.keys():
            raise NotImplemented
        if boundary_conditions not in implemented_boundary_conditions.keys():
            raise NotImplemented

        self.device = torch.device(device)
        self.Lx_min, self.Lx_max = Lx_min, Lx_max
        self.Ly_min, self.Ly_max = Ly_min, Ly_max

        self.T = T
        self.nt = nt
        self.dt = T / nt

        self.snapshot_frequency = snapshot_frequency
        self.num_snapshots = self.nt // snapshot_frequency
        self.dtype = dtype

        # ghost cells due to implementation of laplacian
        self.nx = nx + 2
        self.ny = ny + 2

        self.dx = (self.Lx_max - self.Lx_min) / (nx + 1)
        self.dy = (self.Ly_max - self.Ly_min) / (ny + 1)

        self.tn = torch.linspace(0, T, nt, dtype=self.dtype)

        self.xn = torch.linspace(
            Lx_min,
            Lx_max,
            self.nx,
            device=self.device,
            dtype=self.dtype)
        self.yn = torch.linspace(
            Ly_min,
            Ly_max,
            self.ny,
            device=self.device,
            dtype=self.dtype)

        self.u = torch.zeros(
            (self.num_snapshots,
             self.nx,
             self.ny),
            device=self.device,
            dtype=self.dtype)
        self.v = torch.zeros(
            (self.num_snapshots,
             self.nx,
             self.ny),
            device=self.device,
            dtype=self.dtype)

        self.X, self.Y = torch.meshgrid(self.xn, self.yn, indexing='ij')

        self.method = step_method
        self.step = implemented_methods[self.method]
        self.apply_bc = implemented_boundary_conditions[boundary_conditions]

        self.save_last_k = save_last_k[self.method]

        self.enable_energy_control = enable_energy_control

        self.c2 = c2(self.X, self.Y) if callable(c2) else c2
        self.c2 = torch.tensor(self.c2) if not isinstance(
                self.c2, torch.Tensor) else self.c2

        self.m = m(self.X, self.Y) if callable(m) else m
        self.m = torch.tensor(self.m) if not isinstance(
                self.m, torch.Tensor) else self.m

        self.initial_u = initial_u
        self.initial_v = initial_v

        if step_method == 'ETD1-sparse' or step_method == 'ETD1-krylov':
            # construct all torch.SparseTensor
            # L, L inverse
            # special matrices for gamma
            # use two modes: krylov and approximation until O(t^4)
            raise Exception

 
    # enable for long time-stepping!
    # TODO benchmark the different variants
    @torch.compile
    def grad_Vq(self, u):
        out = u_xx_yy(torch.zeros_like(u), u, torch.tensor(self.dx, dtype=self.dtype), torch.tensor(self.dy, dtype=self.dtype))
        out.mul_(self.c2)
        out.sub_(self.m * torch.sin(u))
        return out
       
 
    def grad_Vq_small_nt(self, u):
        return self.c2 * (
                u_xx(
                    u, torch.tensor(self.dx, dtype=self.dtype)) + u_yy(
                        u, torch.tensor(self.dy, dtype=self.dtype))) - self.m * torch.sin(u)

    def energy(self, u, v):
        # a crude approximation for the energy integral of the Hamiltonian
        # TODO build more accurate version for analysis purposes
        ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * self.dx)
        uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * self.dy)
        ut = v[1:-1, 1:-1]
        ux2 = ux ** 2
        uy2 = uy ** 2
        ut2 = ut ** 2
        cos = 2 * (1 - torch.cos(u[1:-1, 1:-1]))

        # TODO incorporate c, m here 
        integrand = torch.sum((ux2 + uy2) + ut2 + (cos))
        return 0.5 * integrand * self.dx * self.dy

    def neumann_bc(self, u, v):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

        v[0, 1:-1] = 0
        v[-1, 1:-1] = 0
        v[1:-1, 0] = 0
        v[1:-1, -1] = 0

    def stormer_verlet_pseudo_step(self, u, v, last_k, i):
        vn = v + self.dt * self.grad_Vq(u)
        un = u + self.dt * vn
        return un, vn, []

    def gauss_legendre_step(self, u, v, last_k, i):
        dt = self.dt
        b1 = b2 = .5

        a11 = a22 = .25
        a12 = a21 = .25 - np.sqrt(3) / 6

        def fixed_point_solve(u, v, tol=1e-10, max_iter=100):
            device = u.device
            nx, ny = u.shape

            a11 = a22 = 0.25
            a12 = a21 = 0.25 - (3.0**0.5) / 6
            b1 = b2 = 0.5

            k1u = v.clone()
            k1v = self.grad_Vq(u)

            k2u = v.clone()
            k2v = k1v.clone()

            # this fix point iteration does not take into account the
            # boundary conditions nicely -- which may make it decay
            # under certain spatial discretization for example
            for i in range(max_iter):
                k1u_old = k1u.clone()
                k1v_old = k1v.clone()
                k2u_old = k2u.clone()
                k2v_old = k2v.clone()
                u1 = u + dt * (a11 * k1u + a12 * k2u)
                u2 = u + dt * (a21 * k1u + a22 * k2u)
                k1v = self.grad_Vq(u1)
                k2v = self.grad_Vq(u2)
                self.apply_bc(k1v, k2v)
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
        u, v = fixed_point_solve(u, v)
        return u, v, []

    def rk4_step(self, u, v, last_k, i):
        def f(u):
            return self.grad_Vq(u)
        k1_v = self.dt * f(u)
        k2_v = self.dt * f(u + 0.5 * self.dt * v)
        k3_v = self.dt * f(u + 0.5 * self.dt * (v + 0.5 * k1_v))
        k4_v = self.dt * f(u + self.dt * (v + k3_v))

        k1_u = self.dt * v
        k2_u = self.dt * (v + 0.5 * k1_v)
        k3_u = self.dt * (v + 0.5 * k2_v)
        k4_u = self.dt * (v + k3_v)

        un = u + (1 / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        vn = v + (1 / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return un, vn, []

    def etd1_step(self, u, v, i):
        # TODO: finish implementation of this guy and benchmark!
        u = u.reshape()
        v = v.reshape()
        uv    = torch.cat([u, v],)
        gamma = torch.cat([torch.zeros_like(u), -torch.sin(u)],)
        # TODO: the second matmul can be halved: top half of gamma is zeros:
        # just use right blocks of matrix
        u_vec = torch.sparse.mm(self.Exp, uv) + torch.sparse.mm(self.Lstar, gamma) 
        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny)) 
        return un, vn
        
    def evolve(self,):
        u0 = self.initial_u(self.X, self.Y)
        v0 = self.initial_v(self.X, self.Y)

        u, v = u0, v0
        E0 = self.energy(u, v)

        self.u[0], self.v[0] = u0, v0
        # heterogenous container to collect last few steps of u, v
        # depending on method might have to be used and well-updated
        last_k = [[u], [v]] if self.save_last_k else []
        abort = False
        for i, t in enumerate(tqdm(self.tn)):
            if i == 0:
                continue  # we already initialized u0, v0
            u, v, last_k = self.step(u, v, last_k, i)
            self.apply_bc(u, v)

            if i % self.snapshot_frequency == 0:
                self.u[i // self.snapshot_frequency] = u
                self.v[i // self.snapshot_frequency] = v

            if self.enable_energy_control:
                E = self.energy(u, v)
                if E > 2 * E0 or E < .5 * E0:
                    abort = True
                    print("Aborting, energy diverges")
                    #raise Exception("Method diverged, aborting timestepping")
            if abort:
                self.u[(i // self.snapshot_frequency):] = torch.nan
                self.v[(i // self.snapshot_frequency):] = torch.nan
                break

    def reset(self):
        self.u = torch.zeros_like(self.u)
        self.v = torch.zeros_like(self.v)

def animate(X, Y, data, dt, num_snapshots, nt):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                data[frame],
                cmap='viridis')
        ax.set_title(f"t={(frame * dt * (nt / num_snapshots)):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()

if __name__ == '__main__':
    L = 50
    nx = ny = 400
    T = 200
    nt = 20000
    #initial_u = static_breather
    initial_u = bubble
    initial_v = zero_velocity
    
    implemented_methods = {
        'stormer-verlet-pseudo': None,
        #'gauss-legendre': None,
        #'RK4': None,
    }

    #solutions = {
    #    'stormer-verlet-pseudo': None,
    #    'gauss-legendre': None,
    #    'RK4': None,

    #        }
    for method in implemented_methods.keys():
        solver = SineGordonIntegrator(-L, L, -L, L, nx,
                                  ny, T, nt, initial_u, initial_v, step_method=method,
                                  #c2 = lambda X, Y: .1 * torch.exp(-(X ** 2 + Y ** 2)),
                                  #c2 = lambda X, Y: torch.exp(-(1/X**2 + 1/Y**2)),
                                  c2 = 1,
                                  enable_energy_control=False,
                                  device='cpu')
        solver.evolve()
        implemented_methods[method] = solver.u.clone().cpu().numpy()
        animate(
                solver.X.cpu().numpy(), solver.Y.cpu().numpy(),
                solver.u, solver.dt, solver.num_snapshots, solver.nt)

        
        es = []
        for i in range(solver.num_snapshots):
            u, v = solver.u[i], solver.v[i]
            es.append(solver.energy(u, v).cpu().numpy())

        plt.plot(
            solver.tn.cpu().numpy()[
                ::solver.snapshot_frequency][0:len(es)],
            es,
            label=method)         
    plt.legend()
    plt.show()
    

    """

    baseline = implemented_methods['stormer-verlet-pseudo']
    for method in implemented_methods:
        if method == 'stormer-verlet-pseudo': continue
        # L_infty
        #diff = np.max(np.abs(solutions[method] - baseline), axis=(1,2))

        # L_2
        diff = np.sqrt(np.sum(np.abs(implemented_methods[method] - baseline) ** 2, axis=(1,2)))
        plt.plot(diff, label=method)
    plt.title("Differences to Stoermer-Verlet solution")
    plt.legend()
    plt.show()
    """



"""

class ISTAnalyzer:
    def __init__(self, Lx: float, Ly: float, device: str = 'cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.device = device
        
    def setup_grid(self, solution: torch.Tensor):
        self.nt, self.nx, self.ny = solution.shape
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = torch.linspace(-self.Lx/2, self.Lx/2, self.nx, device=self.device)
        self.y = torch.linspace(-self.Ly/2, self.Ly/2, self.ny, device=self.device)
        self.kx = 2*np.pi*fft.fftfreq(self.nx, self.dx)
        self.ky = 2*np.pi*fft.fftfreq(self.ny, self.dy)

    def compute_spectral_quantities(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_x = torch.fft.ifft2(1j * self.kx.reshape(-1, 1) * torch.fft.fft2(u)).real
        u_y = torch.fft.ifft2(1j * self.ky.reshape(1, -1) * torch.fft.fft2(u)).real
        energy_density = 0.5 * (u_x**2 + u_y**2) + (1 - torch.cos(u))
        momentum_density = u_x * torch.fft.ifft2(-1j * self.ky.reshape(1, -1) * torch.fft.fft2(u)).real
        return energy_density, momentum_density

    def compute_floquet_discriminant(self, u: torch.Tensor, lambda_val: float) -> torch.Tensor:
        L = self._construct_lax_operator(u, lambda_val)
        L_flattened = L.reshape(2, 2, -1)
        monodromy = torch.eye(2, dtype=torch.complex64, device=self.device)
        
        for i in range(L_flattened.shape[2]):
            local_L = L_flattened[:, :, i]
            monodromy = torch.matmul(local_L, monodromy)
            
        return 0.5 * torch.trace(monodromy)

    def find_eigenvalues(self, u: torch.Tensor, lambda_range: Tuple[float, float], 
                        n_points: int = 100) -> torch.Tensor:
        lambda_vals = torch.linspace(lambda_range[0], lambda_range[1], n_points, device=self.device)
        disc_vals = []
        
        for lam in lambda_vals:
            disc = self.compute_floquet_discriminant(u, lam)
            disc_vals.append(disc.abs() - 1)
            
        disc_vals = torch.stack(disc_vals)
        eigenvalues = []        
        for i in range(len(lambda_vals)-1):
            if torch.sign(disc_vals[i].real) != torch.sign(disc_vals[i+1].real):
                root = lambda_vals[i] + (lambda_vals[i+1] - lambda_vals[i]) * (
                    disc_vals[i].real / (disc_vals[i].real - disc_vals[i+1].real)
                )
                eigenvalues.append(root)
                
        return torch.tensor(eigenvalues, device=self.device)
        
    def _construct_lax_operator(self, u: torch.Tensor, lambda_val: float) -> torch.Tensor:
        u_x = torch.fft.ifft2(1j * self.kx.reshape(-1, 1) * torch.fft.fft2(u)).real
        u_y = torch.fft.ifft2(1j * self.ky.reshape(1, -1) * torch.fft.fft2(u)).real
        
        L = torch.zeros((2, 2, self.nx, self.ny), dtype=torch.complex64, device=self.device)
        L[0,0] = -1j*lambda_val
        L[0,1] = 0.25*(u_x + 1j*u_y)
        L[1,0] = 0.25*(u_x - 1j*u_y)
        L[1,1] = 1j*lambda_val
        return L
        
    def extract_soliton_parameters(self, eigenvalues: torch.Tensor, 
                                 u: torch.Tensor) -> List[SolitonParameters]:
        parameters = []
        for eigenval in eigenvalues:
            L = self._construct_lax_operator(u, eigenval)
            L_flat = L.reshape(2, 2, -1)
            eigvals, eigvecs = [], []
            
            for i in range(L_flat.shape[2]):
                local_eigvals, local_eigvecs = torch.linalg.eig(L_flat[:,:,i])
                idx = torch.argmax(torch.abs(local_eigvals))
                eigvals.append(local_eigvals[idx])
                eigvecs.append(local_eigvecs[:,idx])
            
            eigvecs = torch.stack(eigvecs)
            pos_x = torch.sum(self.x.reshape(-1, 1) * torch.abs(eigvecs[:,0])**2) / torch.sum(torch.abs(eigvecs[:,0])**2)
            pos_y = torch.sum(self.y.reshape(1, -1) * torch.abs(eigvecs[:,0])**2) / torch.sum(torch.abs(eigvecs[:,0])**2)
            
            velocity = 2 * torch.imag(eigenval)
            amplitude = 4 * torch.arctan(1 / torch.real(eigenval))
            phase = torch.angle(torch.mean(eigvecs[:,0]))
            
            parameters.append(SolitonParameters(
                velocity=velocity,
                position=torch.stack([pos_x, pos_y]),
                amplitude=amplitude,
                phase=phase
            ))
        return parameters
        
    def analyze_evolution(self, solution: torch.Tensor, dt: float, 
                         lambda_range: Tuple[float, float]) -> dict:
        self.setup_grid(solution)
        results = {
            'soliton_parameters': [],
            'eigenvalues': []
        }
        for t in tqdm(range(self.nt)):
            eigenvals = self.find_eigenvalues(solution[t], lambda_range)
            soliton_params = self.extract_soliton_parameters(eigenvals, solution[t])
            results['eigenvalues'].append(eigenvals)
            results['soliton_parameters'].append(soliton_params)
             
        return results

"""
