import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from typing import Callable
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch.fft as fft

import scipy.sparse as sp
import scipy.sparse.linalg as spla

def get_progress_bar(iterable=None, total=None, **kwargs):
    try:
        from tqdm import tqdm
        if iterable is not None:
            return tqdm(iterable, **kwargs)
        return tqdm(total=total, **kwargs)
    except ImportError:
        return iterable if iterable is not None else range(total)


def extract_lower_blocks(sparse_matrix, n):
    """
    get C and D 2n²×2n² block matrix [[A, B], [C, D]]
    """
    N = n**2
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()

    mask = indices[0] >= N
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]

    mask_B = filtered_indices[0] < N
    B_indices = filtered_indices[:, mask_B]
    B_indices[1] -= N
    B_values = filtered_values[mask_B]

    mask_D = filtered_indices[1] >= N
    D_indices = filtered_indices[:, mask_D] - N
    D_values = filtered_values[mask_D]

    B = torch.sparse_coo_tensor(
        B_indices, B_values,
        size=(N, N),
        dtype=sparse_matrix.dtype
    )

    D = torch.sparse_coo_tensor(
        D_indices, D_values,
        size=(N, N),
        dtype=sparse_matrix.dtype
    )

    return B, D

@torch.compile
def arnoldi_iteration_compiled(A, v, k, t):
    m = A.shape[0]
    Q = torch.zeros((m, k+1), dtype=A.dtype)
    H = torch.zeros((k+1, k), dtype=A.dtype)
    Q[:, 0] = v / torch.norm(v)
    for j in range(k):
        w = torch.sparse.mm(A, Q[:, j].unsqueeze(1)).squeeze()
        for i in range(j+1):
            H[i, j] = torch.dot(w, Q[:, i])
            w -= H[i, j] * Q[:, i]
        if j < k-1:
            H[j+1, j] = torch.norm(w)
            if H[j+1, j] > 1e-12:
                Q[:, j+1] = w / H[j+1, j]
            else:
                return Q, H
    return Q, H

def arnoldi_iteration(A, v, k, t):
    m = A.shape[0]
    Q = torch.zeros((m, k+1), dtype=A.dtype)
    H = torch.zeros((k+1, k), dtype=A.dtype)
    Q[:, 0] = v / torch.norm(v)
    for j in range(k):
        w = torch.sparse.mm(A, Q[:, j].unsqueeze(1)).squeeze()
        for i in range(j+1):
            H[i, j] = torch.dot(w, Q[:, i])
            w -= H[i, j] * Q[:, i]
        if j < k-1:
            H[j+1, j] = torch.norm(w)
            if H[j+1, j] > 1e-12:
                Q[:, j+1] = w / H[j+1, j]
            else:
                return Q, H
    return Q, H

def expm_multiply(A, v, t, k=30, compiled=False):
    m = A.shape[0]
    beta = torch.norm(v)
    V, H = arnoldi_iteration(A, v/beta, k, t) if not compiled else arnoldi_iteration_compiled(A, v/beta, k, t)
    tol = 1e-5
    for k in range(1, k+1 // 2):
        F = torch.matrix_exp(t * H[:k, :k])
        w = beta * V[:, :k] @ F[:, 0]
        error = torch.norm(t * H[k, k-1] * F[k-1, 0])
        if error < tol:
            return w
    return w

def torch_sparse_to_scipy_sparse(torch_sparse, shape): 
    indices = torch_sparse.indices().cpu().numpy()
    values = torch_sparse.values().cpu().numpy()

    scipy_sparse = sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=shape
    )
    return scipy_sparse.tocsr()


# this method builds a sparse matrix containing
# the discretized Laplacian on a square grid
def build_D2(nx, ny, dx, dy, dtype):
    assert nx == ny
    assert dx == dy

    N = (nx + 2) ** 2
    middle_diag = -4 * torch.ones(nx + 2, dtype=dtype)
    middle_diag[0] = middle_diag[-1] = -3
    left_upper_diag = lower_right_diag = middle_diag + torch.ones(nx + 2, dtype=dtype)
    diag = torch.cat([left_upper_diag] + [middle_diag] * nx + [lower_right_diag])

    offdiag_pos = torch.ones(N - 1, dtype=dtype)
    inner_outer_identity = torch.ones(N - (nx + 2),  dtype=dtype)

    indices_main = torch.arange(N, dtype=dtype)
    indices_off1 = torch.arange(1, N, dtype=dtype)
    indices_off2 = torch.arange(0, N - 1, dtype=dtype)

    row_indices = torch.cat([
        indices_main, indices_off1, indices_off2,
        indices_main[:-(nx+2)], indices_main[nx+2:]
    ])

    col_indices = torch.cat([
        indices_main, indices_off2, indices_off1,
        indices_main[nx+2:], indices_main[:-nx-2]
    ])

    values = torch.cat([
        diag, offdiag_pos, offdiag_pos,
        inner_outer_identity, inner_outer_identity
    ])

    L = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=(N, N),
        dtype=dtype,
    )
    L *= (1 / dx) ** 2
    return L

# this is a bad approximation :(
def build_D2_inverse(nx, ny, dx, dy, dtype):
    assert nx == ny
    assert dx == dy
    N = (nx + 2) ** 2
    i = torch.arange(nx + 2, dtype=dtype)
    j = torch.arange(nx + 2, dtype=dtype)
    ii, jj = torch.meshgrid(i, j, indexing='ij')

    # ev: lambda_ij = -4 + 2*cos(πi/N) + 2*cos(πj/N)
    eigenvalues = -4 + 2*torch.cos(torch.pi * ii / (nx + 1)) + 2*torch.cos(torch.pi * jj / (nx + 1))
    eigenvalues = eigenvalues.reshape(-1)
    indices_main = torch.arange(N, dtype=torch.int64)
    epsilon = 1e-10
    values = 1.0 / (eigenvalues + epsilon) * (dx ** 2)
    L_inv = torch.sparse_coo_tensor(
        indices=torch.stack([indices_main, indices_main]),
        values=values,
        size=(N, N),
        dtype=dtype,
    )
    return L_inv

def apply_inverse_lapl(u, n, dx):
    u = u.view(n, n) 
    kx = torch.fft.fftfreq(n, d=dx) * 2 * np.pi
    ky = torch.fft.fftfreq(n, d=dx) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    u_hat = torch.fft.fft2(u) 
    u_hat = u_hat / (-k2)
    result = torch.fft.ifft2(u_hat).real
    return result.view(n * n, 1).squeeze() 


def sparse_block(block_list, row_offsets, col_offsets, shape):
    combined_indices, combined_values = [], []
    for entry, row_offset, col_offset in zip(block_list, row_offsets, col_offsets):
        if isinstance(entry, int): continue
        new_entry = entry.clone()
        new_idx = new_entry.indices()
        val = new_entry.values()
        new_idx[0] += row_offset
        new_idx[1] += col_offset
        combined_indices.append(new_idx)
        combined_values.append(val)
 
    combined_indices = torch.cat(combined_indices, dim=1)
    combined_values  = torch.cat(combined_values)
    return torch.sparse_coo_tensor(combined_indices, combined_values, size=shape)

#@torch.jit.script
def u_yy(buf, a, dy):
    uyy = torch.zeros_like(a)
    uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] -
                       2 * a[1:-1, 1:-1]) / (dy ** 2)
    return uyy

#@torch.jit.script
def u_xx(buf, a, dx):
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

def u_xx_yy_9(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1,1:-1] = (1/(6*dx*dy)) * (
        4*(a[1:-1,2:] + a[1:-1,:-2] + a[2:,1:-1] + a[:-2,1:-1]) + 
        a[2:,2:] + a[:-2,:-2] + a[2:,:-2] + a[:-2,2:] - 
        20*a[1:-1,1:-1])    
    return uxx_yy

def u_xx_yy_13(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1, 1:-1] = (1 / (12 * dx * dy)) * (
        10 * (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1]) +
        5 * (a[2:, 2:] + a[2:, :-2] + a[:-2, 2:] + a[:-2, :-2]) -
        60 * a[1:-1, 1:-1]
    )
    return uxx_yy

def cos_diff(un, u): 
    diff = un - u
    mask = torch.abs(diff) < 1e-6
    safe_ratio = (torch.cos(un) - torch.cos(u)) / diff
    taylor = -torch.sin((un + u) / 2)
    return torch.where(mask, taylor, safe_ratio)

def safe_div(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

def safe_sqrt(x, eps=1e-8):
    return torch.sqrt(torch.clamp(x, min=eps))


@dataclass
class SolitonParameters:
    velocity: torch.Tensor
    position: torch.Tensor
    amplitude: torch.Tensor
    phase: torch.Tensor

def ring_soliton(X, Y, xc, yc, R):
    return 4 * torch.arctan(torch.exp(-torch.sqrt(((X - xc) ** 2 + (X - yc) ** 2)) - R))
    #return 4 * torch.arctan(((X - xc) ** 2 + (X - yc) ** 2) / (2 * R))
    #return 4 * torch.arctan(((X - xc) ** 2 + (X - yc) ** 2 - R ** 2) / (2 * R))

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

def kink_start(x, y):
    return -4 * torch.exp(x + y) / (1 + torch.exp(2*x + 2*y))

def analytical_kink_solution(x, y, t):
    if isinstance(x, torch.Tensor):
        return 4 * torch.atan(torch.exp( x + y - t)) 
    else:
        return 4 * np.atan(np.exp( x + y - t))

def circular_ring(x, y):
    return 2 * torch.atan(torch.exp(3 - 5 * torch.sqrt(x ** 2 + y ** 2)))

def lbf_circular_ring(x, y):
    r2 = x ** 2 + y ** 2
    alpha = 4
    return alpha * torch.arctan(torch.exp(3 - torch.sqrt(r2)))

def lapl_eigenvalues(nx, ny, dx, dy, tau=None):
    # findable from REU paper using google search
    ev = torch.zeros((nx, ny))
    Nx = int(1 / dx) 
    Ny = int(1 / dy)
    Lx = nx * dx
    Ly = ny * dy
    norm = dx * dy
    if tau is None: tau = 1
    for i in range(-nx//2, nx//2):
        for j in range(-ny//2, ny // 2): 
            ev[i, j] = 4 * np.pi ** 2 / (nx ** 2 * dx ** 2) * (i ** 2 + j ** 2)
            #ev[i, j] = -(4 /nx/ny) * (
            #        torch.sin(torch.tensor(tau * torch.pi * i / nx / 2)) ** 2 + torch.sin(tau * torch.tensor(torch.pi * j / ny / 2)) ** 2
            #        )
    return ev


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
            'ETD1-sparse': self.etd1_sparse_step,
            'ETD2-sparse': self.etd2_sparse_step,
            'ETD2-RK': self.etd2_rk_step,
            'ETD1-sparse-opt': self.etd1_sparse_opt_step,
            'ETD1-krylov': self.etd1_krylov_step,
            'Energy-conserving-1': self.eec_sparse_step,
            'Strang-split': self.strang_step,
            'Leap-frog': self.leapfrog_step,
            'stormer-verlet': self.stormer_verlet_step,
            'lie-euler': self.lie_euler_step,
            'yoshida-4': self.yoshida4_step,
            'yoshida-6': self.yoshida6_step,
            'candy-neri6': self.candy_neri6_step,
            'ruth': self.ruth_step,
            'symplectic-euler': self.symplectic_euler_step,
            'AVF': self.avf_step,
            'AVF-explicit': self.avf_explicit_step,
            'HBVM': self.hbvm_step,
            'newmark': self.newmark_step,
            'gautschi-erkn': self.gautschi_erkn_step,
            'erkn-3': self.erkn3_step,
            'etd-spectral': self.etd_spectral_step,
            'linear-DG': self.linear_energy_conserving_step,
            'modified-law': self.lin_implicit_step,
            'locally-energy-conserving-implicit': self.transformed_step,
            'avf-4th-fixed-point': self.avf_fixed_point_step,
            'bratsos-corrector': self.bratsos_explicit_step, # amounts to Stormer-Verlet
            'bratsos-implicit': self.bratsos_corrector_step,
            'crank-nicholson': self.crank_nicholson_step,
            'lattice-boltzmann': self.lbm_step,
        }
        save_last_k = {
            'stormer-verlet-pseudo': 0,
            'gauss-legendre': 0,
            'RK4': 0,
            'Strang-split': None,
            'ETD2-RK': None,
            'ETD1-sparse': None,
            'ETD2-sparse': None,
            'ETD1-sparse-opt': None,
            'ETD1-krylov': None,
            'Energy-conserving-1': None,
            'Leap-frog': None,
            'stormer-verlet': None,
            'lie-euler': None,
            'yoshida-4': None,
            'yoshida-6': None,
            'candy-neri6': None,
            'ruth': None,
            'symplectic-euler': None,
            'AVF': None,
            'AVF-explicit': None,
            'HBVM': None,
            'newmark': None,
            'gautschi-erkn': None,
            'erkn-3': None,
            'etd-spectral': None,
            'linear-DG': None,
            'modified-law': None,
            'locally-energy-conserving-implicit': None,
            'avf-4th-fixed-point': None,
            'bratsos-corrector': None,
            'bratsos-implicit': None,
            'crank-nicholson': None,
            'lattice-boltzmann': None,
        }

        implemented_boundary_conditions = {
            'homogeneous-neumann': self.neumann_bc,
            'special-kink-bc': self.special_kink_bc
            }

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

        self.buf = torch.zeros((self.nx, self.ny)).to(self.device)

        self.iteration_stats = None
        if 'ETD' in step_method:
            """
            This evolution uses approximations of exp(tau * L) where L is a block-sparse matrix
            
                 | 0  |  Id | 
            L =  | -------- |
                 | D2 |   0 |

            where Id is the identity with shape=(self.nx², self.nx²) (we're assuming nx == ny)
            and D2 is the discretized Laplacian (5-point stencil) of the finite differences
            method and homogenous von Neumann bounday conditions (no flux conditions).

            None, one can show that very quickly terms of order O(tau⁴) appear.
            Assuming tau in the order of 1e-2, one can assume that this approximation should work
            fairly nicely, as we reach machine precision after that. To be confirmed in numerical
            experiments.

            Due to the nature of the ETD-1 update step, we can totally avoid computing the inverse
            of D2, making our lives much easier.
            """
            self.D2 = build_D2(nx, ny, self.dx, self.dy, self.dtype).coalesce()
            self.D4 = torch.sparse.mm(self.D2, self.D2).coalesce()
            self.D6 = torch.sparse.mm(self.D2, self.D4).coalesce()
            self.Id = torch.sparse.spdiags(
                    torch.ones( (nx+2) ** 2, dtype=self.dtype),
                    offsets=torch.tensor([0]),
                    shape=((nx+2) ** 2, (nx+2)**2),)

            self.D2_inv = build_D2_inverse(nx, ny, self.dx, self.dy, self.dtype)
             
            self.L = sparse_block(
                    [0,
                        self.Id.to_sparse_coo().coalesce(), self.D2.to_sparse_coo().coalesce(),
                        0], 
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()



            self.L_inv = sparse_block([0, self.D2_inv.to_sparse_coo().coalesce(), self.Id.to_sparse_coo().coalesce(), 0], 
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()


            self.Id = self.Id.coalesce()
            self.D2 = self.D2.coalesce()
            self.D4 = self.D4.coalesce()
            self.D6 = self.D6.coalesce()
            self.T0 = sparse_block([self.Id, 0, 0, self.Id], row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr() 
            self.T1 = sparse_block([0, self.dt * self.Id, self.dt * self.D2, 0],
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()
            self.T2 = sparse_block([.5 * self.dt ** 2 * self.D2, 0, 0, .5 * self.dt ** 2 * self.D2],
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()
            self.T3 = sparse_block([0, (1/6) * self.dt ** 3 * self.D2, (1/6) * self.dt ** 3 * self.D4],
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()
            self.T4 = self.dt ** 4 / 24 * sparse_block([self.D4, 0, 0, self.D4],
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()
            self.T5 = self.dt ** 5 / 120 * sparse_block([0, self.D4, self.D6, 0],
                    row_offsets = [0, 0, self.nx ** 2, self.nx ** 2],
                    col_offsets = [0, self.ny ** 2, 0, self.ny ** 2],
                    shape=(2 * self.nx ** 2, 2 * self.nx ** 2) 
                    ).coalesce().to_sparse_csr()

            self.exp_t_L_approx = self.T0 + self.T1 + self.T2 + self.T3 + self.T4 + self.T5 
            self.exp_t_L_half_approx = self.T0 + .5 * self.T1 + .5 ** 2 * self.T2 + .5 ** 3 * self.T3 +\
                    .5 ** 4 * self.T4 + .5 ** 5 * self.T5

            self.exp_t_L_no_Id  = self.T1 + self.T2 + self.T3 + self.T4 + self.T5
            self.exp_t_L_no_first_two  = self.T2 + self.T3 + self.T4 + self.T5

            self.T = (self.L_inv @ self.exp_t_L_no_Id).to_sparse_coo().coalesce()

            self.T_no_inv =  self.exp_t_L_no_Id.to_sparse_coo().coalesce()
            self.lower_left_no_inv, self.lower_right_no_inv = extract_lower_blocks(self.T_no_inv, nx + 2)
            self.lower_left_no_inv = self.lower_left_no_inv.coalesce().to_sparse_csr() 
            self.lower_right_no_inv = self.lower_right_no_inv.coalesce().to_sparse_csr()

            self.lower_left, self.lower_right = extract_lower_blocks(self.T, nx + 2)
            self.lower_left = self.lower_left.coalesce().to_sparse_csr()
            self.lower_right = self.lower_right.coalesce().to_sparse_csr()
            self.T = self.T.to_sparse_csr()

            self.T_next = self.L_inv @ self.L_inv @ (
                    (self.exp_t_L_no_Id.to_sparse_coo() - self.dt * self.L.to_sparse_coo()).to_sparse_csr()
                    )

            # needed for inverse laplacian
            self.kx = torch.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
            self.ky = torch.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
            self.kx, ky = torch.meshgrid(self.kx, self.ky, indexing='ij')
            self.k2 = self.kx**2 + self.ky**2
            self.k2[0, 0] = 1



    # enable for long time-stepping!
    # TODO benchmark the different variants
    @torch.compile
    def grad_Vq(self, u):
        out = u_xx_yy(self.buf, u, torch.tensor(self.dx, dtype=self.dtype), torch.tensor(self.dy, dtype=self.dtype))
        out.mul_(self.c2)
        out.sub_(self.m * torch.sin(u))
        return out
       
 
    def grad_Vq_small_nt(self, u):
        return self.c2 * (
                u_xx(
                    u, torch.tensor(self.dx, dtype=self.dtype)) + u_yy(
                        u, torch.tensor(self.dy, dtype=self.dtype))) - self.m * torch.sin(u)

    def energy_approx(self, u, v):
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

    def energy(self, u, v):
        ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * self.dx)
        uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * self.dy)
        ut = v[1:-1, 1:-1]
        ux2 = .5 * ux ** 2
        uy2 = .5 * uy ** 2
        ut2 = .5 * ut ** 2
        cos = self.m * (1 - torch.cos(u[1:-1, 1:-1])) 
        integrand = torch.sum((ux2 + uy2) + ut2 + (cos))
        return 0.5 * integrand * self.dx * self.dy


    def energy_density(self, u, v):
        ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * self.dx)
        uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * self.dy)
        ut = v[1:-1, 1:-1]

        return (.5 * (ut**2 + ux**2 + uy**2) + (1 - torch.cos(u[1:-1, 1:-1]))) 

    def energy_flux(self, u, v):
        lap = (u_xx(self.buf, u, torch.tensor(self.dx, dtype=self.dtype))
                + u_yy(self.buf, u, torch.tensor(self.dy, dtype=self.dtype)))[1:-1, 1:-1]
        
        # v \Delta u
        t1 = v[1:-1, 1:-1] * lap
        grad_v = torch.stack([
            (v[1:-1, 2:] - v[1:-1, :-2]) / (2. * self.dx),
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2. * self.dy)
            ]).reshape(2, (self.nx-2) ** 2)
        grad_u = torch.stack([
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * self.dx),
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * self.dy)
            ]).reshape(2, (self.nx-2) ** 2)
        
        # grad u dot grad v
        t2 = grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1] 
        return t1 + t2.reshape((self.nx-2, self.nx-2))

    def neumann_bc(self, u, v, i, tau=None,):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
 
        v[0, 1:-1] = 0
        v[-1, 1:-1] = 0
        v[1:-1, 0] = 0
        v[1:-1, -1] = 0



    def special_kink_bc(self, u, v, i, tau=None):
        
        def boundary_x(x, Y, t): 
            x = float(x) * torch.ones_like(Y)
            t = t * torch.ones_like(Y)
            return -4 * torch.exp(x + Y + t) / (torch.exp(2 * t) + torch.exp(2 * x + 2 * Y))
    
        def boundary_y(X, y, t):
            y = float(y) * torch.ones_like(X) 
            t = t * torch.ones_like(X)
            return -4 * torch.exp(X + y + t) / (torch.exp(2 * t) + torch.exp(2 * X + 2 * y))

        # TODO figure out correct form to finally compare
        # precision of all methods!
        dx = self.dx
        dy = self.dy
        dt = self.dt
        t = i * self.dt if tau is None else tau

        xm, xM = torch.tensor(self.Lx_min, dtype=self.dtype), torch.tensor(self.Lx_max, dtype=self.dtype)
        ym, yM = torch.tensor(self.Ly_min, dtype=self.dtype), torch.tensor(self.Lx_max, dtype=self.dtype)

        # u's ghost cells get approximation following boundary condition
        u[0,  1:-1] = u[1, 1:-1] + dx * boundary_x(xm, self.yn[1:-1], t)
        u[-1, 1:-1] = u[-2, 1:-1] - dx * boundary_x(xM, self.yn[1:-1], t)

        u[1:-1,  0] = u[1:-1, 1] + dy * boundary_y(self.xn[1:-1], ym, t)
        u[1:-1, -1] = u[1:-1, -2] - dy * boundary_y(self.xn[1:-1], yM, t)

        #u[0,  1:-1] = u[1, 1:-1] - dy * boundary_y(self.xn[1:-1], ym, t)
        #u[-1, 1:-1] = u[-2, 1:-1] + dy * boundary_y(self.xn[1:-1], yM, t)

        #u[1:-1,  0] = u[1:-1, 1] - dx * boundary_x(xm, self.yn[1:-1], t)
        #u[1:-1, -1] = u[1:-1, -2] + dx * boundary_x(xM, self.xn[1:-1], t)

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

    def symplectic_euler_step(self, u, v, last_k, i):
        un = u + self.dt * v
        vn = v + self.dt * self.grad_Vq(un)
        return un, vn, []

    def ruth_step(self, u, v, last_k, i):
        c = 2/3
        vn = v + c * self.dt * self.grad_Vq(u)
        un = u + c * self.dt * vn
        vn = vn - c * self.dt * self.grad_Vq(un)
        un = un - c * self.dt * vn
        vn = vn + self.dt * self.grad_Vq(un)
        un = un + self.dt * vn
        return un, vn, []

    def candy_neri6_step(self, u, v, last_k, i): 
        a = [
            0.37693334772664,
            -0.15828265660178,
            0.39590138677536,
            -0.18977338823387,
            -0.18977338823387,
            0.39590138677536,
            -0.15828265660178,
            0.37693334772664
        ]

        un, vn = u, v
        for ai in a:
            vn = vn + ai * self.dt * self.grad_Vq(un)
            un = un + ai * self.dt * vn
            self.apply_bc(un, vn, i)

        return un, vn, []


    def leapfrog_step(self, u, v, last_k, i):
        uhalf = u + .5 * self.dt * v
        vn = v + self.dt * self.grad_Vq(uhalf)
        un = uhalf + .5 * self.dt * vn
        return un, vn, []

    def yoshida4_step(self, u, v, last_k, i):
        w0 = 1.0/(2.0 - 2.0**(1.0/3.0))
        w1 = -2.0**(1.0/3.0)/(2.0 - 2.0**(1.0/3.0)) 
        
        weights = [w1, w0, w1]
        un, vn = u, v
        un_prev = u
        for w in weights: 
            w = .5 * w
            vn = vn + w * self.dt * self.grad_Vq(un) 
            un = un + w * self.dt * vn
            self.apply_bc(un, vn, i) 
        return un, vn, []

    def yoshida6_step(self, u, v, last_k, i):
        w0 = 1.0 / (2.0 - 2.0**(1.0 / 5.0))
        w1 = -2.0**(1.0 / 5.0) / (2.0 - 2.0**(1.0 / 5.0))
        w2 = w0
        weights = [w0, w1, w2, w1, w0]

        un, vn = u.clone(), v.clone()
        for w in weights:
            vn = vn + w * self.dt * self.grad_Vq(un)
            un = un + w * self.dt * vn
            self.apply_bc(un, vn)

        return un, vn, []

    def lie_euler_step(self, u, v, last_k, i):
        # Explain: Solution has issues with boundary
        # -> L probably still ill-formed
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])
        uv_new = uv + self.dt * self.L @ uv
        un = uv_new[:self.nx ** 2].reshape((self.nx, self.nx))
        v_part = uv_new[self.nx ** 2:].reshape((self.nx, self.nx)) 
        vn = v_part - self.dt * self.m * torch.sin(un)
        return un, vn, []

    def stormer_verlet_step(self, u, v, last_k, i):
        if i == 1:
            vn = v + self.dt * self.grad_Vq(u)
            un = u + self.dt * vn
            return un, vn, [u]
         
        u_past = last_k[-1]
        un = 2 * u - u_past + self.dt ** 2 * self.grad_Vq(u) 
        vn = (un - u) / self.dt
        last_k.append(u)
        return un, vn, last_k

    def hbvm_step(self, u, v, last_k, i):
        from scipy.special import legendre

        def buildAB(k, s):
            V = torch.zeros((k, s+1), device=self.device, dtype=self.dtype)
            for i in range(s+1):
                V[:, i] = self._shifted_legendre(self.nodes, i)

        # TODO!
        # no energy correction
        k = 5
        s = 5
        nodes, weights = np.polynomial.legendre.leggauss(k)
        nodes = torch.tensor(nodes, dtype=self.dtype)
        weights = torch.tensor(weights, dtype=self.dtype)
        P = torch.zeros((k, s), dtype=self.dtype)
        for i in range(s):
            leg_poly = legendre(i)
            P[:, i] = torch.tensor([leg_poly(x) for x in nodes], dtype=self.dtype)

        def H(u, v):
            u_x = (u[1:, :] - u[:-1, :]) / self.dx
            u_y = (u[:, 1:] - u[:, :-1]) / self.dy
            h = 0.5 * torch.sum(v**2) + \
                0.5 * self.c2 * (torch.sum(u_x**2) + torch.sum(u_y**2)) - \
                self.m * torch.sum(1 - torch.cos(u))
            return h

        H0 = H(u, v)
        for i in range(k):
            pass



    def ieq_step(self, q, p, last_k, i):
        # TODO: understand method fully
        if i == 0 or i == 1: self.epsilon = 1e-6

        V = self.potential_energy(q)
        psi = torch.sqrt(2 * V + self.epsilon)
        g = self.grad_psi(q, psi)
        p_half = p - 0.5 * self.dt * psi * g
        q_new = q + self.dt * p_half
        psi_new = psi + self.dt * g.T @ p_half

        p_new = p_half - 0.5 * self.dt * psi_new * self.grad_psi(q_new, psi_new)

        return q_new, p_new, []

    def eec_sparse_step_gl(self, u, v, last_k, i):
        weights = torch.tensor([0.236926885, 0.478628670, 0.568888889, 0.478628670, 0.236926885], device=self.device)
        points = torch.tensor([-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846], device=self.device)
        scaled_points = (points + 1) / 2 * self.dt
        def integrate_forces_gl(u_free):
            forces = [self.grad_Vq(u_free(t)) for t in scaled_points]
            [self.apply_bc(u, f, i) for f in forces]
            forces = torch.stack(forces) 
            [self.apply_bc(forces[k], v, i) for k in range(5)]
            force_sum = torch.einsum('k,knm->nm', weights, forces)
            return -.5 * self.dt * force_sum

        if i == 1:
            phalf = v
        else:
            phalf = last_k[0]

        u_free = lambda t: u + t * phalf
        force_integral = integrate_forces_gl(u_free)
        v_new = v - force_integral
        u_new = u + self.dt * v_new
        return u_new, v_new, [v_new]

    def eec_sparse_step_pass(self, u, v, last_k, i):
        if i == 1:
            return u + self.dt * v, v + self.dt * self.grad_Vq(u), [u, v]
        return u, v, [] 

    def eec_sparse_step_incomplete(self, u, v, last_k, i):
        if i == 1:
            return u + self.dt * v, v + self.dt * self.grad_Vq(u), [u, v]
        def ux_uy(u):
            u_x = torch.zeros_like(u)
            u_y = torch.zeros_like(u)
            u_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2
            u_x[0, :] = u[1, :] - u[0, :]
            u_x[-1, :] = u[-1, :] - u[-2, :]
            u_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2 
            u_y[:, 0] = u[:, 1] - u[:, 0]
            u_y[:, -1] = u[:, -1] - u[:, -2]
            return u_x, u_y

        def f(u, v):
            dx = torch.tensor(self.dx)
            dy = torch.tensor(self.dy)
            un = u + self.dt * v 
            d = un - u + 1e-10 
            vn = v + self.dt * (u_xx_yy_9(self.buf, un + u, dx, dy)/ 2 - self.m * (
                -2 * torch.sin((u+un) / 2) * torch.sin((un-u)/2)  / d)) 
            return un, vn

        un, vn = f(u, v)  
        return un, vn, []

    def avf_step(self, u, v, last_k, i):
        def F(U):
            return -self.m * torch.sin(U)

        def get_A():
            return build_D2(self.nx-2, self.ny-2, self.dx, self.dy, self.dtype)

        def fun(u, v):
            return self.energy_density(u, v)

        def fac(n):
            if n <= 1: return 1
            s = 1
            for i in range(2, n+1):
                s *= i
            return s
        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)

        # first order only for phi_l(K) @ x
        def apply_phi0(x, order=3):
            return x

        def apply_phi1(x, order=3):
            return x 
           
        def apply_phi2(x, order=3):
            return .5 * x  

        def u_next(u, un, v, dg):
            t1 = apply_phi0(u)
            t2 = apply_phi1(v)
            t3 = apply_phi2(dg) 
            return t1 + self.dt * t2 - self.dt ** 2 * t3

        def v_next(u, un, v, dg):
            t1 = u_xx_yy(self.buf, apply_phi1(u), torch.tensor(self.dx), torch.tensor(self.dy))
            t2 = apply_phi0(v)
            t3 = apply_phi1(dg)
            return -self.dt * t1 + t2 - self.dt * t3

        def neumann_bc(u, v):
            un, vn = u.clone(), v.clone()
            un[0, 1:-1] = u[1, 1:-1]
            un[-1, 1:-1] = u[-2, 1:-1]
            un[:, 0] = u[:, 1]
            un[:, -1] = u[:, -2]
 
            vn[0, 1:-1] = 0
            vn[-1, 1:-1] = 0
            vn[1:-1, 0] = 0
            vn[1:-1, -1] = 0
            return un, vn

        def solve(u, v):
            un = u + self.dt * v 
               
            max_iter = 10
            alpha = 1. 
            for curr in range(max_iter):     
                un.requires_grad_(True) 
                dg = (self.grad_Vq(un) - self.grad_Vq(u))    

                unn = u_next(u, un, v, dg)
                vnn = v_next(u, un, v, dg)  
 
                if (self.energy(u, v) - self.energy(unn, vnn)).norm() < 1e-6:
                    break

                r_u = unn - un
                r_v = vnn - (v + self.dt * self.grad_Vq(un))

                residual = torch.cat([r_u.ravel(), r_v.ravel()])
                residual_norm = torch.norm(residual ** 2)
                grad_u = torch.autograd.grad(residual_norm, un, retain_graph=True)[0]
                with torch.no_grad():
                    un = unn - grad_u
                un = un.detach()
            
            vn = v + self.dt * self.grad_Vq(un)
            return un, vn 

        def fun(u, v, un, vn,):
            dg = (self.grad_Vq(un) - self.grad_Vq(u))
            y = torch.stack([torch.zeros_like(u), torch.zeros_like(v)])
            y[0] = un - (apply_phi0(u) + self.dt * apply_phi1(v) - self.dt ** 2 * apply_phi2(dg))
            y[1] = vn - (-self.dt * u_xx_yy(self.buf, apply_phi1(u), dx, dy) + apply_phi0(u) - self.dt * apply_phi1(dg))
            return y


        def solve_diff(u, v):
            initial_u = u + self.dt * v
            initial_v = v

            max_iter = 5
            un, vn = initial_u.clone(), initial_v.clone()
            un = un.requires_grad_(True)
            vn = vn.requires_grad_(True)
            N = u.shape[0] * u.shape[1]
            for it in range(max_iter):
                r1, r2 = fun(u, v, un, vn)
                r_norm = torch.norm(r1 ** 2 + r2 ** 2)
                if r_norm < 1e-6:
                    break
                r1 = r1.reshape(-1)
                r2 = r2.reshape(-1)

                J11 = torch.zeros((N, N), dtype=self.dtype) 
                J12 = torch.zeros((N, N), dtype=self.dtype)
                J21 = torch.zeros((N, N), dtype=self.dtype)
                J22 = torch.zeros((N, N), dtype=self.dtype)

                
                for e in range(N):
                    if un.grad is not None: un.grad.zero_()
                    if vn.grad is not None: vn.grad.zero_()

                    r1[e].backward(retain_graph=True)

                    if un.grad is not None:
                        J11[:, e] = un.grad.reshape(-1)
                    if vn.grad is not None:
                        J12[:, e] = vn.grad.reshape(-1)

                    r2[e].backward(retain_graph=True)
                    if un.grad is not None:
                        J21[:, e] = un.grad.reshape(-1)
                    if vn.grad is not None:
                        J22[:, e] = vn.grad.reshape(-1)

                # this matrix is singular unfortunately
                J = torch.cat([
                    torch.cat([J11, J12], dim=1),
                    torch.cat([J21, J22], dim=1),])
                R = torch.cat([r1, r2])

               

                d = torch.linalg.solve(J, -R)
                du = d[:N].reshape((self.nx, self.ny))
                dv = d[N:].reshape((self.nx, self.ny))

                with torch.no_grad():
                    un = torch.tensor(un + du, requires_grad=True,)
                    vn = torch.tensor(vn + dv, requires_grad=True,)
            return un, vn 

        def solve_exact(u, v):
            initial_u = u + self.dt * v
            initial_v = v

            max_iter = 100
            un, vn = initial_u.clone(), initial_v.clone()
 
            N = u.shape[0] * u.shape[1]

            

            def fun(u, v, un, vn):
                unn = un - self.dt * (v + vn) / 2 
                vnn = vn + self.dt * (-u_xx_yy(self.buf, (un - un) / 2, dx, dy) +
                            - self.m * (cos_diff(un, u)))
                return unn, vnn

            def Jac(u, v, un, vn):
                # Id
                J11 = u
                # - tau / 2 Id
                J12 = -self.dt / 2 * v
                # - tau / 2 (\Delta + m tau (-sin(un) + cos(un) / (un - u) ** 2)
                #J21 = -self.dt / 2 * u_xx_yy(self.buf, u, dx, dy) + self.m * self.dt * (
                #        (-(torch.sin(un)  + torch.cos(un) / (u - un) ** 2))
                #        )
                J21 = -self.dt / 2 * u_xx_yy(self.buf, u, dx, dy) + self.m * self.dt * (-cos_diff(un, u))
                # Id
                J22 = v
                
                return J11, J12, J21, J22

            def mult_Jac(u, v, un, vn):
                J11, J12, J21, J22 = Jac(u, v, un, vn)

                top = J11 * u + J12 * v 
                bot = J21 * u + J22 * v

                return torch.stack([top, bot])
            
            for it in range(max_iter):

                Jw = mult_Jac(u, v, un, vn)
                import pdb; pdb.set_trace()

                #r_u = torch.max((un - u).abs())
                #r_v = torch.max((un - u).abs())
                #if max(r_u, r_v) < 1e-6:
                #    break
            success = ((it + 1) != max_iter) 
            return un, vn, success

        def loss_fn(u, v, un, vn):
            r_u = (un - u) / self.dt - (vn + v) / 2
            r_v = -u_xx_yy(self.buf, (un + u) / 2, dx, dy) - self.m * cos_diff(un, vn)
            f1 = torch.sum(r_u ** 2 + r_v ** 2)  
            f2 = torch.norm(self.energy(u, v) - self.energy(un, vn)) ** 2
            return f1 + f2

        def solve_opt(u, v):
            un = u + self.dt * v
            vn = v
            u_next = un.clone().detach().requires_grad_(True)
            v_next = vn.clone().detach().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                    [u_next, v_next],
                    max_iter=2,
                    line_search_fn="strong_wolfe")
            max_iter = 10

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(un, vn, u_next, v_next)
                loss.backward(retain_graph=True)
                return loss

            for it in range(max_iter):
                loss = optimizer.step(closure)
                if loss.item() < 1e-3:
                    break
            return u_next, v_next


        un, vn= solve(u, v)
        #if not converged:
        #    print("Fixed point iteration failed in AVF integrator")
        ##print("E diff:", torch.norm(self.energy(un, vn) - self.energy(u, v)))  
        return un, vn, []
            

    def avf_explicit_step_past(self, u, v, last_k, i):
        if i < 4:
            return self.stormer_verlet_step(u, v, last_k, i)

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)

        u2, u1, u0 = last_k[::-1]
        u2 = u

        r = self.dt
        assert dx == dy
  
        lap_u21 = u_xx_yy(self.buf, u2 + u1, dx, dy)
        #lap_u1 = u_xx_yy(self.buf, u1, dx, dy)

        G_u2 = 1 - torch.cos(u2)
        G_u1 = 1 - torch.cos(u1)

        #G_u2 = torch.cos(u2)
        #G_u1 = torch.cos(u1)
 
        un = (u2 + u1 - u0 + 
              2 * r**2 * (lap_u21 - (G_u2 - G_u1) / (u2 - u1)))
        #print(torch.max(2 * r**2 * lap_u21 - 2 * r**2 * (G_u2 - G_u1) / (u2 - u1)))
        #import pdb; pdb.set_trace()
        
        vn = (un - u1) / (2 * self.dt)
        return un, vn, [un, u2, u1]

    def avf_explicit_step(self, u, v, last_k, i):
        if i == 1:
            L = build_D2(self.nx-2, self.ny-2, self.dx, self.dy, dtype=self.dtype).coalesce()
            L_scipy = torch_sparse_to_scipy_sparse(L, ((self.nx) ** 2, (self.nx) ** 2)) 
            I = sp.eye(self.nx * self.nx, format='csr')
            self.A = I + self.dt ** 2 / 2 * L_scipy

            self.iteration_stats = []
        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)
        max_iter = 10
        un = u.clone() + self.dt * v
        alpha = self.dx

        un = u + self.dt * v
        vn = v - .5 * self.dt * (un - u) * (self.grad_Vq(un) + self.grad_Vq(u))
        return un, vn, []


        for it in range(max_iter): 
            #u_old = un.clone() 
            # Not entirely sure why this form now works
            #un = u - (self.dt * v + (self.dt ** 2 / 4) * ((u_xx_yy(self.buf, (un + u)/2, dx, dy))) -\
            #        (self.dt ** 2/2) * (cos_diff(un, u))) 

            #vn = v - (self.dt/2) * (u_xx_yy(self.buf, u + un, dx, dy)) - self.dt * cos_diff(un, u)
        

            #rhs_solve = u - (self.dt * v - self.dt ** 2 / 2 * u_xx_yy(self.buf, u, dx, dy)\
            #        + self.dt ** 2 / 2 * cos_diff(un, u))

            #rhs_solve = rhs_solve.reshape(self.nx ** 2).cpu().numpy()
            #un = spla.spsolve(self.A, rhs_solve)
            #un = torch.from_numpy(un).reshape((self.nx, self.nx))

            #lhs_u = (un - u) / self.dt
            #rhs_u = v + self.dt ** 2 / 4 * ((u_xx_yy(self.buf, (un + u), dx, dy))) + self.dt ** 2 /2 * (cos_diff(un, u))

            def F(x):
                return (self.dt * v + (self.dt ** 2 / 4) * ((u_xx_yy(self.buf, (x + u)/2, dx, dy))) -\
                    (self.dt ** 2/2) * (cos_diff(x, u)))
           
 
            u_old = un.clone()  
            un = u - F(un)  
            

            #lhs_v = (vn - v) / self.dt
            #rhs_v = v - (self.dt/2) * (u_xx_yy(self.buf, u + un, dx, dy)) - self.dt * cos_diff(un, u) 

            #res_u = torch.norm(lhs_u - rhs_u)
            #res_v = torch.norm(lhs_v - rhs_v) 
            #res = res_u + res_v

            #u_old = un.clone()
            #un = u - self.dt * v - self.dt ** 2 / 4 * u_xx_yy(self.buf, u + un, dx, dy)\
            #        - self.dt ** 2 / 2 * cos_diff(un, u)

            res = torch.norm(un - u_old) 
            if res < 1:
                break
        if torch.isnan(res):
            raise Exception
        
        self.iteration_stats.append(res) 
        vn = v - self.dt / 2 * u_xx_yy(self.buf, un + u, dx, dy) - self.dt * cos_diff(un, u)
        return un, vn, []

    def avf_fixed_point_step(self, u, v, last_k, i):
        if i == 1:
            vn = v + self.dt * self.grad_Vq(u)
            un = u + self.dt * vn
            return un, vn, [u]

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)

        def cos_diff(x, y):
            return -torch.sin(x + y) / 2

        def u_next(u, v, un, vn):
            v_s = (v + vn)
            lap_s = u_xx_yy(self.buf, v_s, dx, dy)
            Lambda = torch.cos((un + u) / 2)

            return u + self.dt * (
                    .5 * v_s - self.dt ** 2 / 24 * (
                        lap_s - self.m * Lambda * v_s)
                    )

        def v_next(u, v, un, vn):
            cos = cos_diff(un, u)
            lap_s = u_xx_yy(self.buf, un + u, dx, dy)
            Lambda = torch.cos((un + u) / 2)

            t1 = .5 * (lap_s + 2 * self.m * cos)
            f1 = u_xx_yy(self.buf, lap_s, dx, dy)
            f2 = -self.m * Lambda * cos 
            f3 = 2 * self.m * u_xx_yy(self.buf, cos, dx, dy)
            f4 = -2 * self.m ** 2 * Lambda * cos

            return v + self.dt * (t1 - self.dt ** 2 / 24 * (f1 + f2 + f3 + f4))
      
        def contract(u, v):
            un = u.clone()
            vn = v.clone()
            max_iter = 25
            for it in range(max_iter):
                u_old = un.clone()
                v_old = vn.clone()
                un = u_next(u, v, u_old, v_old)
                vn = v_next(u, v, u_old, v_old) 
                if it > 10 and max(torch.norm(un - u_old).item(), torch.norm(vn - v_old).item()) < 1e-8:
                    break
            return un, vn
        
        un, vn = contract(u, v)
         
        return un, vn, []

    def lbm_step(self, u, v, last_k, i):
        # kick off ???? 
        #v = self.grad_Vq(u)
    

        def grad(u):
            dx = torch.tensor(self.dx)
            dy = torch.tensor(self.dy)
            u_x, u_y = torch.zeros_like(u), torch.zeros_like(u)
            u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
            u_x[:, 0] = (u[:, 1] - u[:, 0]) / dx
            u_x[:, -1] = (u[:, -1] - u[:, -2]) / dx
    
            u_y[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
            u_y[0, :] = (u[1, :] - u[0, :]) / dy
            u_y[-1, :] = (u[-1, :] - u[-2, :]) / dy
            return u_x, u_y

        def feq(u, u_t,):
            assert u.shape == u_t.shape
            nx, ny = u.shape
            f = torch.zeros((nx, ny, self.l), dtype=self.dtype)
            ux, uy = grad(u) 
            #u_diff_expanded = self.cs ** 2 * (u - u_t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
            scaled_id = self.cs ** 2 * torch.eye(2, dtype=self.dtype)
            u_expanded   = u.unsqueeze(-1).unsqueeze(-1) 
            u_t_expanded = u_t.unsqueeze(-1).unsqueeze(-1)


            for j, cxy in zip(self.d_idx, self.cxys):   
                cjcj = torch.outer(cxy, cxy)
                diff_expanded   = (cjcj - scaled_id).unsqueeze(0).unsqueeze(1) 
                double_dot = (self.cs ** 2 * (u_expanded - u_t_expanded) * diff_expanded).sum(dim=(-2, -1)) 
                f[:, :, j] = self.weights[j] * (u_t + double_dot / (2 * self.cs ** 4))
            return f

        def F(u):
            nx, ny = u.shape
            term = torch.zeros((nx, ny, self.l), dtype=self.dtype)
            return self.weights.view(1, 1, -1) * u.unsqueeze(-1)
            #for j in range(self.l):
            #    term[:, :, j] = self.weights[j] * u
            #return term

        # D2EQ
        self.l = 9
        self.d_idx = torch.arange(self.l)
        self.d_idx_p = torch.tensor([0, 5, 6, 7, 8, 1, 2, 3, 4])
 
        if i == 1:     
            # some constants we need
            self.c = max(self.dx, self.dy) / self.dt
            self.cs = np.sqrt(self.c ** 2 / 3)
            self.tau = .5 + 1 / (self.dt *  self.cs ** 2) 
            # too much 
            #self.tau = 1e5 * (.5 + 1 / self.cs ** 2)


            

            self.cxs       = self.c * torch.tensor([0, 0, 1, 1, 1, 0,-1,-1,-1], dtype=self.dtype)
            self.cys       = self.c * torch.tensor([0, 1, 1, 0,-1,-1,-1, 0, 1], dtype=self.dtype)
            self.cxys      = torch.vstack((self.cxs, self.cys)).T
            self.weights   = torch.tensor([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36], dtype=self.dtype)

            self.dxdy = torch.tensor([self.dx, self.dy], dtype=self.dtype)

            # vec(dx) = eps c_j
            m = self.dxdy / self.cxys == torch.inf
            z = self.dxdy / self.cxys
            z[m] = 0 
            self.eps  = z

            self.f_eq = feq(u, v)
            self.f = 1 / self.tau * self.f_eq
        
            bdry_x = torch.logical_or(self.X == self.Lx_min, self.X == self.Lx_max)
            bdry_y = torch.logical_or(self.Y == self.Ly_min, self.Y == self.Ly_max)
            self.bdry_mask = torch.logical_or(bdry_x, bdry_y)

            # sum_j w_j c_j == 0
            assert torch.allclose(self.weights @ self.cxys, 1e-10 * torch.ones(2, dtype=self.dtype))
            
            # sum_j w_j c_j c_j.T == cs² Id
            outers = torch.bmm(self.cxys.unsqueeze(2), self.cxys.unsqueeze(1))
            z = (outers * self.weights.view(-1, 1, 1)).sum(dim=0)
            assert torch.allclose(z, self.cs ** 2 * torch.eye(2, dtype=self.dtype))
        
        
        boundary_f = self.f[self.bdry_mask, :] 
        boundary_f = boundary_f[:, self.d_idx_p] # just take reflection at first
        self.f_eq = feq(u, v)
        # forget about second order for now
        self.f = self.f - (1/self.tau) * (self.f - self.f_eq) +\
                self.dt * F(-self.m * torch.sin(u)) + .5 * self.dt ** 2 * F(-self.m * torch.cos(u))
        
        self.f[self.bdry_mask, :] = boundary_f 
        vn = torch.zeros_like(u)
        for j in range(self.l):
            vn += self.weights[j] * self.f[:, :, j] 
        un = u + self.dt * vn

       
        return un, vn, []

       

    def crank_nicholson_step(self, u, v, last_k, i):
        if i == 1:
            L = build_D2(self.nx-2, self.ny-2, self.dx, self.dy, dtype=self.dtype).coalesce()
            L_scipy = torch_sparse_to_scipy_sparse(L, ((self.nx) ** 2, (self.nx) ** 2)) 
            I = sp.eye(self.nx * self.nx, format='csr')
            self.Id = I
            self.LHS1 = I - self.dt ** 2 / 2 * L_scipy
            return self.stormer_verlet_step(u, v, last_k, i)

        u_past = last_k[0]
        dx, dy = torch.tensor(self.dx), torch.tensor(self.dy)
        sin = torch.sin(u)
        rhs = 2 * u - u_past + self.dt ** 2 / 2 * (u_xx_yy(self.buf, u, dx, dy) - self.m * sin)

        rhs = rhs.reshape(self.nx ** 2).cpu().numpy()
        
        LHS = self.LHS1# + self.dt ** 2 * float(self.m) / 2 * self.Id @ sin.reshape(self.nx ** 2).cpu().numpy()

        un = torch.from_numpy(spla.spsolve(LHS, rhs))
        un = un.reshape((self.nx, self.nx))
        vn = (un - u) / self.dt

        return un, vn, [u]

        
    def newmark_step(self, u, v, last_k, i):
        # broken
        # needs parameters gamma, beta
        if i == 1:
            a1 = -self.grad_Vq(u)
            un = u + self.dt * v + .5 * self.dt ** 2 * a1
            vn = v + self.dt * a1
            return un, vn, [a1]

        a_old = last_k[0]
        an = -self.grad_Vq(u)
        un = u + self.dt * v + .5 * self.dt ** 2 
        vn = v + self.dt * an
        anew = -self.grad_Vq(un)
        return un, vn, [anew]


    def gautschi_erkn_step(self, u, v, last_k, i):
        # kind of like stormer-verlet, with filtering 
        if i == 1:
            # build_D2(nx, ny, dx, dy, dtype)
            self.ev0 = lapl_eigenvalues(self.nx, self.ny, self.dx, self.dy,).real
            self.ev0 = self.ev0.reshape((self.nx, self.nx))

            self.ev = lapl_eigenvalues(self.nx, self.ny, self.dx, self.dy, self.dt).real
            self.ev = self.ev.reshape((self.nx, self.nx))

            self.ev_half = lapl_eigenvalues(self.nx, self.ny, self.dx, self.dy, self.dt/2).real
            self.ev_half = self.ev.reshape((self.nx, self.nx))

            self.kx = 2 * torch.pi * torch.fft.fftfreq(self.nx, self.dx, device=u.device)
            self.ky = 2 * torch.pi * torch.fft.fftfreq(self.ny, self.dy, device=u.device)
            self.KX, self.KY = torch.meshgrid(self.kx, self.ky, indexing='xy')

            vn = v + self.dt * self.grad_Vq(u)
            un = u + self.dt * vn
            return un, vn, [u]

        def sinc(x):
            return torch.where(x != 0, torch.sin(x) / x, torch.ones_like(x))

        def sinc2(x):
            return sinc(x) ** 2

        # maybe we can save this (or avoid the PBC assumption)
        # by introducing some smart padding
        def spectral_op(
                u: torch.Tensor,
                op, # Callable
                t: float,
                dx: float,
                dy: float,
                eps: float = 1e-14  
            ) ->         torch.Tensor: 
            KX, KY = self.KX, self.KY 
            ny, nx = u.shape

            # |ξ| = √(ξ_x² + ξ_y²)
            xi_magnitude = torch.sqrt(KX**2 + KY**2)
            
            buffer_freq = torch.fft.fft2(u)
            # make sure to have t correctly scaled!
            multiplier = op(t * xi_magnitude) 
            buffer_freq *= multiplier
            return torch.real(torch.fft.ifft2(buffer_freq))
             
        def spectral_step(u, v, last_k):
            t1 = 2 * spectral_op(u, torch.cos, self.dt, self.dx, self.dy,)
            
            g_intern = spectral_op(u, sinc2, self.dt, self.dx, self.dy,)
            g = torch.sin(g_intern)
            t2 = spectral_op(g, sinc2, self.dt / 2, self.dx, self.dy,)

            u_past = last_k[-1] 
            un = t1 - u_past + self.dt ** 2 * t2
            vn = (un - u) / self.dt
            return un, vn

        def pseudo_analytical_step(u, v, last_k):
            # u_{n+1} = 2 cos(t \Omega) u - u_past - t² sinc²(t/2 \Omega) g (phi(t \Omega) u)

            def phi(x):
                return torch.sinc(x / 2) ** 2
            dx = torch.tensor(self.dx)
            dy = torch.tensor(self.dy) 
            u_past = last_k[-1]
            u_hat = fft.fft2(u)
            t1 = 2 * fft.ifft2(torch.cos(self.dt * torch.sqrt(self.ev)) * u_hat).real 
            t2 = -u_past
            t3_s = fft.fft2(torch.sin(fft.ifft2(phi(self.dt * torch.sqrt(self.ev) * u_hat))))
            t3 = -fft.ifft2(sinc2(self.dt / 4 * torch.sqrt(self.ev)) * t3_s).real
             
            un = t1 + t2 + self.dt ** 2 * t3 
            vn = (un - u) / self.dt
            return un, vn
 
            # this should work now
            #u_hat = fft.fft2(u) 
            #lapl = self.ev0 * u_hat
            #sin_u = -self.m *torch.sin(u)
  
            
            ##vn = v + self.dt * (u_xx_yy_9(self.buf, u, dx, dy) + sin_u)
            #vn = v + self.dt * (fft.ifft2(lapl).real + sin_u)
            #un = u + self.dt * vn
            #return un, vn

        un, vn = pseudo_analytical_step(u, v, last_k) 
        return un, vn, [u]

    def etd_spectral_step(self, u, v, last_k, i):
        nx, ny = u.shape
        device = u.device
        if i == 1:
            kx = fft.fftfreq(nx, d=1/nx).to(device)
            ky = fft.fftfreq(ny, d=1/ny).to(device)
            self.kx, self.ky = torch.meshgrid(kx, ky, indexing='ij')
            self.k_squared = self.kx**2 + self.ky**2

            Lx = self.dx * self.nx
            Ly = self.dy * self.ny

            self.k_squared *= 4 / (Lx * Ly)
            self.k_squared[0, 0] = 1.0 

            self.exp_term = torch.exp(-self.k_squared * self.dt)
            self.phi_term = (1 - self.exp_term) / self.k_squared

        u_hat = fft.fftn(u)
        v_hat = fft.fftn(v)
        g_u = -torch.sin(u)
        g_u_hat = fft.fftn(g_u)
        u_next_hat = self.exp_term * u_hat + self.phi_term * v_hat
        v_next_hat = -self.k_squared * self.phi_term * u_hat + self.exp_term * v_hat + self.phi_term * g_u_hat
        u_next = fft.ifftn(u_next_hat).real
        v_next = fft.ifftn(v_next_hat).real
        u_next[0, 0] = u[0, 0] + self.dt * v[0, 0]
        v_next[0, 0] = v[0, 0] + self.dt * g_u[0, 0]
        return u_next, v_next, []

    def erkn3_step(self, u, v, last_k, i):
        k1 = self.grad_Vq(u)
        u_half = u + 0.5*self.dt*v + 0.125*(self.dt**2)*k1
        k2 = self.grad_Vq(u_half)
        un = u + self.dt*v + (self.dt**2/6)*(k1 + 2*k2)
        vn = v + (self.dt/6)*(k1 + 4*k2 + self.grad_Vq(un))
        return un, vn, []

    def lin_implicit_lower_order_step(self, u, v, last_k, i):
        

        if i == 1:
            a = self.grad_Vq(u)
            un = u + self.dt * v 
            vn = v + self.dt * a
            rn = torch.sqrt(2 - torch.cos(un)) 
            return un, vn, [rn]

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)
        r = last_k[0]

        sqrt = safe_sqrt(2 - torch.cos(u))
        div = safe_div(torch.sin(u), sqrt)

        u_t = v
        v_t = u_xx_yy(self.buf, u, dx, dy) - div * r
        r_t = .5 * div * v

        un = u + self.dt * u_t
        vn = v + self.dt * v_t  
        rn = r + self.dt * r_t
        return un, vn, [rn]
    
    def lin_implicit_step(self, u, v, last_k, i):
        def safe_div(numerator, denominator, eps=1e-8):
            return numerator / (denominator + eps)

        def safe_sqrt(x, eps=1e-8):
            return torch.sqrt(torch.clamp(x, min=eps))
        if i == 1:
            a = self.grad_Vq(u)
            un = u + self.dt * v
            vn = v + self.dt * a
            rn = torch.sqrt(2 - torch.cos(un))
            return un, vn, [rn]

        k1_u = torch.zeros_like(u)
        k1_v = torch.zeros_like(v)
        k1_r = torch.zeros_like(last_k[0])
        k2_u = torch.zeros_like(u)
        k2_v = torch.zeros_like(v)
        k2_r = torch.zeros_like(last_k[0])

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)
        r = last_k[0]

        sqrt = safe_sqrt(2 - torch.cos(u))
        div = safe_div(torch.sin(u), sqrt)
        k1_u = v
        k1_v = u_xx_yy(self.buf, u, dx, dy) - div * r
        k1_r = 0.5 * div * v
        u_mid = u + 0.5 * self.dt * k1_u
        v_mid = v + 0.5 * self.dt * k1_v
        r_mid = r + 0.5 * self.dt * k1_r

        sqrt_mid = safe_sqrt(2 - torch.cos(u_mid))
        div_mid = safe_div(torch.sin(u_mid), sqrt_mid)
        k2_u = v_mid
        k2_v = u_xx_yy(self.buf, u_mid, dx, dy) - div_mid * r_mid
        k2_r = 0.5 * div_mid * v_mid

        un = u + self.dt * k2_u
        vn = v + self.dt * k2_v
        rn = r + self.dt * k2_r

        return un, vn, [rn]

    def bratsos_explicit_step(self, u, v, last_k, i):
        if i == 1:
            vn = v + self.dt * self.grad_Vq(u)
            un = u + self.dt * vn
            return un, vn, [u]

        u_past = last_k[0]

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)
        corrector = u_xx_yy(self.buf, u, dx, dy) - self.m * torch.sin(u)

        un = 2 * u - u_past + self.dt ** 2 * corrector
        vn = (un - u) / self.dt
        return un, vn, [u]

    def bratsos_corrector_step(self, u, v, last_k, i):
        if i == 1:
            vn = v + self.dt * self.grad_Vq(u)
            un = u + self.dt * vn
            return un, vn, [u]
        u_past = last_k[0]
        
        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)        

        def intermediate(u_past, u, u_next):
            del_u_past = u_xx_yy(self.buf, u_past, dx, dy)
            del_u      = u_xx_yy(self.buf, u, dx, dy) 
            del_u_next = u_xx_yy(self.buf, u_next, dx, dy)
            return self.dt ** 2/12 * (del_u_past + 10 * del_u + del_u_next)

        def corrector(u_past, u, u_next):
            s1 = torch.sin(u_next)
            s2 = torch.sin(u)
            s3 = torch.sin(u_past)
            s = s1 + s2 * 10 + s3
            return -self.dt ** 2 / 12 * s

        def contract():
            #un, _, _ = self.bratsos_explicit_step(u, v, last_k, i)  
            un = u.clone()
            max_iter = 10
            for it in range(max_iter):
                u_old = un.clone()
                un = 2 * u - u_past + intermediate(u_past, u, un) + corrector(u_past, u, un)                 
                if torch.norm(un - u_old) < 1e-8:
                    break 
            return un

        un = contract()
        vn = (un - u) / self.dt

        return un, vn, [u]
              

    def transformed_step(self, u, v, last_k, i):
        raise Exception("TODO: correct the single-step implicit update")
        def safe_div(numerator, denominator, eps=1e-8):
            return numerator / (denominator + eps)
        def safe_sqrt(x, eps=1e-8):
            return torch.sqrt(torch.clamp(x, min=eps))

        def b(x):
            return safe_div(torch.sin(x), safe_sqrt(2 - torch.cos(x))) 

        if i == 1:
            L = build_D2(self.nx-2, self.ny-2, self.dx, self.dy, dtype=self.dtype).coalesce()
            L_scipy = torch_sparse_to_scipy_sparse(L, ((self.nx) ** 2, (self.nx) ** 2)) 
            I = sp.eye(self.nx * self.nx, format='csr')

            self.Laplacian = sp.csr_matrix(L_scipy)
            self.Id = I
            self.A = self.Id - self.dt ** 2 / 4 * (self.Laplacian)
            a = self.grad_Vq(u)
            un = u + self.dt * v 
            vn = v + self.dt * a
            rn = torch.sqrt(2 - torch.cos(un)) 
            return un, vn, [u, rn]

        u_past = last_k[0]
        r = last_k[1]

        u_half = ((3 * u - u_past) / 2).reshape(self.nx ** 2)  
        u_half_2 = u_half ** 2

        A = self.A - self.dt ** 2 / 4 * (.5 * self.Id * u_half_2)
        max_iter = 2

        un = u.clone()  
        for it in range(max_iter): 
            u_old = un.clone() 
            bn = b(un).cpu().numpy().reshape(self.nx ** 2)
            un = spla.spsolve(A, bn)
            un = torch.from_numpy(un).reshape((self.nx, self.nx))
            if torch.norm(un - u_old) < 1e-10:
                break
        print("iter=", it)
        
        u_half = u_half.reshape((self.nx, self.nx))
        vn = (2 / self.dt) * (un - u) - v
        rn = r + b(u_half) / 2 * (un - u)
        
        return un, vn, [u, rn]
            
            
    def linear_energy_conserving_step(self, u, v, last_k, i):
        if i == 1:
            #a1 = -self.grad_Vq(u)
            #un = u + self.dt * v + .5 * self.dt ** 2 * a1
            #vn = v + self.dt * a1
            L = build_D2(self.nx-2, self.ny-2, self.dx, self.dy, dtype=self.dtype).coalesce()
            L_scipy = torch_sparse_to_scipy_sparse(L, ((self.nx) ** 2, (self.nx) ** 2)) 
            I = sp.eye(self.nx * self.nx, format='csr')
            self.lhs_op = (2 / self.dt ** 2) * I - .5 * L_scipy 
            self.iteration_stats = []
            #return un, vn, []

        def cos_diff(un, u): 
            #diff = un - u
            #mask = torch.abs(diff) < 1e-6
            #safe_ratio = (torch.cos(un) - torch.cos(u)) / diff
            #t = -torch.sin((un + u) / 2)
            #return torch.where(mask, t, safe_ratio)
            return -torch.sin((un + u) / 2)

        dx = torch.tensor(self.dx)
        dy = torch.tensor(self.dy)

        def k(u, v):
            return (2 * u / self.dt + 2 * v) / self.dt

        def contraction_rhs(u, v, um):
            cd  = cos_diff(um, u)
            lap = u_xx_yy(self.buf, u, dx, dy)
            kuv = k(u, v) 
            return cd + .5 * lap + kuv

    
        def contract(u, v,):
            max_iter = 10
            #un, vn, [] = u + self.dt * v, v, []
            un, vn, [] = u, v, []
            w = 1
            for it in range(max_iter):
                u_old = un.clone()
                rhs = contraction_rhs(u, v, un)
                rhs = rhs.reshape(self.nx ** 2).cpu().numpy()
                un = spla.spsolve(self.lhs_op, rhs)
                un = torch.from_numpy(un).reshape((self.nx, self.nx))
                vn =  2 * (un - u) / self.dt - v
                ## a better proxy is to find an energy conservation law
                #if it > 2 and torch.norm(self.energy(u, v) - self.energy(un, vn)) < 1e-3: 
                #    break
                if torch.norm(un - u_old) < 1e-6:
                    break
            
            self.iteration_stats.append(it) 
            vn =  2 * (un - u) / self.dt - v 
            return un, vn

        un, vn = contract(u, v,)
        return un, vn, []


    def eec_sparse_step(self, u, v, last_k, i):
        # these integral approximations could get some work
        # currently smoothing out unnecessarily
        def integrate_forces_mid(u_free):
            midpoint = u_free(0.5 * self.dt)
            return -self.dt * self.grad_Vq(midpoint)

        fs = torch.zeros((3, u.shape[0], u.shape[1]), dtype=self.dtype)
        def integrate_forces(u_free):
            points = [0.5 - np.sqrt(15)/10, 0.5, 0.5 + np.sqrt(15)/10]
            weights = [5/18, 4/9, 5/18]
            # important: enforce boundary conditions for integral     
            for k, (t, w) in enumerate(zip(points, weights)):
                #if fs != []: self.apply_bc(u, fs[-1], i)
                f = w * self.grad_Vq(u_free(t * self.dt))     
                #self.apply_bc(u, v, i, self.dt + t)
                fs[k] = f
            force_sum = torch.sum(fs, axis=0) 
            return -self.dt * force_sum

        def integrate_forces_gl(u_free):
            self.wx = torch.tensor([
                [0.1527533871307258, 	-0.0765265211334973],
                [0.1527533871307258,	 0.0765265211334973],
                [0.1491729864726037,	-0.2277858511416451],
                [0.1491729864726037,	 0.2277858511416451],
                [0.1420961093183820,	-0.3737060887154195],
                [0.1420961093183820,	 0.3737060887154195],
                [0.1316886384491766,	-0.5108670019508271],
                [0.1316886384491766,	 0.5108670019508271],
                [0.1181945319615184,	-0.6360536807265150],
                [0.1181945319615184,	 0.6360536807265150],
                [0.1019301198172404,	-0.7463319064601508],
                [0.1019301198172404,	 0.7463319064601508],
                [0.0832767415767048,	-0.8391169718222188],
                [0.0832767415767048,	 0.8391169718222188],
                [0.0626720483341091,	-0.9122344282513259],
                [0.0626720483341091,	 0.9122344282513259],
                [0.0406014298003869,	-0.9639719272779138],
                [0.0406014298003869,	 0.9639719272779138],
                [0.0176140071391521,	-0.9931285991850949],
                [0.0176140071391521,	 0.9931285991850949],
                ], device=self.device)

            weights = self.wx[:, 0]
            points = self.wx[:, 1]
            scaled_points = (points + 1) 
            force_sum = sum(w * self.grad_Vq(u_free(t * self.dt)) for w, t in zip(weights, scaled_points))
            return -self.dt * force_sum

        if i == 1:
            phalf = v 
        else:
            phalf = last_k[0]
 
        u_new = u + self.dt * phalf
        u_free = lambda t: u_new + t * phalf
        phalf = phalf - integrate_forces(u_free)
        self.apply_bc(u, phalf, i)
        v_new = self.grad_Vq(u_new) 
        return u_new, v_new, [phalf]


        # somewhat working version, boundaries have an issue
        #u_free = lambda t: u + t * v
        #force_integral = integrate_forces(u_free)
       
        ## TODO: figure out: according to paper we need factor 2 here?
        #v_new = v -  force_integral
        #u_new = u + self.dt * v_new
       
        #return u_new, v_new, [v_new]


    def etd1_sparse_step(self, u, v, last_k, i):
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])

        gamma = torch.cat([torch.zeros_like(u), -self.m * torch.sin(u)])
        u_vec = self.exp_t_L_approx @ uv +  self.T @ gamma

        #gamma = -self.m * torch.sin(u)
        #correction = torch.cat(
        #        [self.lower_left @ gamma, self.lower_right @ gamma]
        #        )
        #u_vec = self.exp_t_L_approx @ uv + correction
        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        return un, vn, []

    def apply_inverse_lapl(self, u, n):
        u = u.view(n, n)  
        u_hat = torch.fft.fft2(u) 
        u_hat = u_hat / (-self.k2)
        result = torch.fft.ifft2(u_hat).real
        return result.view(n * n, 1).squeeze()

    def etd1_sparse_opt_step(self, u, v, last_k, i):
        """
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])
        uv_linear_half = self.exp_t_L_half_approx @ uv

        gamma = -self.m * torch.sin(u)
        nonlinear_correction = torch.cat(
            [
                self.apply_inverse_lapl(self.lower_right_no_inv @ gamma, self.nx),
                self.lower_left_no_inv @ gamma,
            ]
        )
        uv_nonlinear = uv_linear_half + nonlinear_correction
        uv_final = self.exp_t_L_half_approx @ uv_nonlinear
        un = uv_final[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = uv_final[u.shape[0]:].reshape((self.nx, self.ny))
        return un, vn, []
        """
        
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])

        gamma = -self.m * torch.sin(u)
        # see the structure of the matrix to be inverted to understand structure here
        sine_term = torch.cat(
                    [
                      self.apply_inverse_lapl(self.lower_right_no_inv @ gamma, self.nx),
                      self.lower_left_no_inv @ gamma,
                    ]
                )

        u_vec = self.exp_t_L_approx @ uv + sine_term
        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        return un, vn, []

    def etd2_rk_step(self, u, v, last_k, i):        
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])
        gamma = -self.m * torch.sin(u)
        # see the structure of the matrix to be inverted to understand structure here
        sine_term = torch.cat(
                    [
                      self.apply_inverse_lapl(self.lower_right_no_inv @ gamma, self.nx),
                      self.lower_left_no_inv @ gamma,
                    ]
                )

        u_vec = self.exp_t_L_approx @ uv + sine_term
        if i > 0 and last_k != []: 
            # TODO: see "Exponential Time Differencing for Stiff Systems"
            pass

        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        return un, vn, [sine_term]

    def strang_step(self, u, v, last_k, i):
        #vn = v + self.dt * u_xx_yy(
        #    self.buf, u, torch.tensor(self.dx, dtype=self.dtype), torch.tensor(self.dy, dtype=self.dtype))   
        #un = u + self.dt * vn - .5 * self.dt ** 2 * torch.sin(u)

        #return un, vn, []

        v_half = v + 0.5 * self.dt * u_xx_yy(
            self.buf, u, torch.tensor(self.dx, dtype=self.dtype), torch.tensor(self.dy, dtype=self.dtype))
        u_half = u + 0.5 * self.dt * v_half
        v_full = v_half - self.dt * torch.sin(u_half)
        u_full = u_half + 0.5 * self.dt * v_full
        v_final = v_full + 0.5 * self.dt * u_xx_yy(
            self.buf, u_full, torch.tensor(self.dx, dtype=self.dtype), torch.tensor(self.dy, dtype=self.dtype))
        u_final = u_full + 0.5 * self.dt * v_final

        return u_final, v_final, []

    def etd2_sparse_step(self, u, v, last_k, i):
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])

        gamma = -self.m * torch.sin(u)
        correction = torch.cat(
                [self.lower_left @ gamma, self.lower_right @ gamma]
                )
        u_vec = self.exp_t_L_approx @ uv + correction
        if i > 0 and last_k != []:
            gamma_prev = last_k[0]
            # could be optimized as well
            u_vec += self.T_next @ torch.cat([torch.zeros_like(u), (gamma - gamma_prev)])

        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        return un, vn, [gamma]
        
    def etd1_krylov_step(self, u, v, last_k, i):
        u, v = u.ravel(), v.ravel()
        uv = torch.cat([u, v])
        gamma = torch.cat([torch.zeros_like(u), -self.m * torch.sin(u)])
        u_vec = expm_multiply(self.L, uv, self.dt, compiled=False) + \
                self.L_inv @ (expm_multiply(self.L, gamma, self.dt, compiled=False) - gamma)
        un = u_vec[0:u.shape[0]].reshape((self.nx, self.ny))
        vn = u_vec[u.shape[0]: ].reshape((self.nx, self.ny))
        return un, vn, []

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
            for _ in range(max_iter):
                k1u_old = k1u.clone()
                k1v_old = k1v.clone()
                k2u_old = k2u.clone()
                k2v_old = k2v.clone()
                u1 = u + dt * (a11 * k1u + a12 * k2u)
                u2 = u + dt * (a21 * k1u + a22 * k2u)
                k1v = self.grad_Vq(u1)
                k2v = self.grad_Vq(u2)
                self.apply_bc(k1v, k2v, i)
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
        k1_u = self.dt * v
        #self.apply_bc(k1_u, k1_v, i)

        k2_u = self.dt * (v + 0.5 * k1_v)
        k2_v = self.dt * f(u + 0.5 * self.dt * v)
        #self.apply_bc(k2_u, k2_v, i)

        k3_u = self.dt * (v + 0.5 * k2_v)
        k3_v = self.dt * f(u + 0.5 * self.dt * (v + 0.5 * k1_v))
        #self.apply_bc(k3_u, k3_v, i)

        k4_u = self.dt * (v + k3_v)
        k4_v = self.dt * f(u + self.dt * (v + k3_v))
        #self.apply_bc(k4_u, k4_v, i)
 
        un = u + (1 / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        vn = v + (1 / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return un, vn, []

    def check_CFL(self, v):
        return self.dt <= min(self.dx, self.dy) / v.abs().max()
        
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
        for i, t in get_progress_bar(enumerate(self.tn)):
            if i == 0:
                continue  # we already initialized u0, v0
            u, v, last_k = self.step(u, v, last_k, i)
            with torch.no_grad():
                self.apply_bc(u, v, i,)      

            if i % self.snapshot_frequency == 0:
                self.u[i // self.snapshot_frequency] = u
                self.v[i // self.snapshot_frequency] = v

            if self.enable_energy_control:
                E = self.energy(u, v)
                if E > 2 * E0 or E < .5 * E0:
                    abort = True
                    print("Aborting, energy diverges")
            if abort:
                self.u[(i // self.snapshot_frequency):] = torch.nan
                self.v[(i // self.snapshot_frequency):] = torch.nan
                break

    def reset(self):
        self.u = torch.zeros_like(self.u)
        self.v = torch.zeros_like(self.v)

def animate(X, Y, data, dt, num_snapshots, nt, title):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                data[frame],
                cmap='viridis')
        ax.set_title(f"{title}, t={(frame * dt * (nt / num_snapshots)):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()

def animate_comparison(X, Y, data, dt, num_snapshots, nt,): 
    from matplotlib.animation import FuncAnimation

    k = len(data.keys())
    titles = list(data.keys())
    values = list(data.values())

    fig, axs = plt.subplots(figsize=(20, 20), ncols=k, subplot_kw={"projection":'3d'})
     
    def update(frame):
        for i, method_name in enumerate(titles):

            #if method_name != "stormer-verlet-pseudo":
            #    viz_data = np.abs(data[method_name][frame] - data["stormer-verlet-pseudo"][frame])
            #else:
            #    viz_data = data["stormer-verlet-pseudo"][frame]
            viz_data = data[method_name][frame]

            axs[i].clear()
            axs[i].plot_surface(X, Y,
                    viz_data,
                    #np.sin(viz_data / 2),
                    cmap='viridis')
            axs[i].set_title(f"{method_name}, t={(frame * dt * (nt / num_snapshots)):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()

def animate_comparison_with_kink_solution(X, Y, data, dt, num_snapshots, nt,): 
    from matplotlib.animation import FuncAnimation

    k = len(data.keys()) + 1 
    titles = list(data.keys())
    values = list(data.values())

    fig, axs = plt.subplots(figsize=(20, 20), ncols=k, subplot_kw={"projection":'3d'})

    freq = nt / num_snapshots
     
    def update(frame):
        axs[0].clear()
        axs[0].plot_surface(X, Y,
                analytical_kink_solution(X, Y, freq * dt * frame),
                cmap='viridis')
        axs[0].set_title(f"analytical solution, t={(frame * dt * (nt / num_snapshots)):.2f}")

        for i, method_name in enumerate(titles):
            axs[i+1].clear()
            axs[i+1].plot_surface(X, Y,
                    np.abs(analytical_kink_solution(X, Y, freq * dt * frame)-data[method_name][frame]),
                    cmap='viridis')
            axs[i+1].set_title(f"{method_name}, t={(frame * dt * (nt / num_snapshots)):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()

def plot_phase_diagram_quiver(un, vn, titles): 
    plt.figure(figsize=(8, 6))
    k = len(titles)

    colors = np.random.rand(k, 3)
    for l in range(k):
        du_dt = np.gradient(un[l], axis=0)
        dv_dt = np.gradient(vn[l], axis=0)
        n = un[0].shape[1] // 2 
        
        plt.quiver(un[l][:, n, n], vn[l][:, n, n], du_dt[:, n, n], dv_dt[:, n, n],
                label=titles[l],
                color=colors[l],
                angles='xy', scale_units='xy',  )
 
    plt.xlabel('$u$', fontsize=12)
    plt.ylabel('$v$', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def lyapunov_exponent(un, tn, names):
    l = len(un)

    for k in range(l): 
        dx = (un[k][:, 1:, :] - un[k][:, :-1, :])[:, :, 1:]
        dy = (un[k][:, :, 1:] - un[k][:, :, :-1])[:, 1:, :]
        delta = np.sqrt(dx**2 + dy**2)
        delta_avg = np.mean(delta, axis=(1, 2))
        log_delta = np.log(delta_avg[1:] / delta_avg[0])
        plt.plot(tn, log_delta, label=names[k])
    plt.grid(True)
    plt.legend()
    plt.title("Approximation of Lyapunov exponent")
    plt.show()

def L_infty(u_gt, u):
    return torch.norm((u_gt - u), p=float('inf'), dim=(1,2))

def L_2(u_gt, u):
    return torch.norm((u_gt - u), p=2, dim=(1,2))

def RSME(u_gt, u):
    return L_2(u_gt, u) / torch.sqrt(torch.tensor(u.shape[0], dtype=u.dtype))

def error_norms(ground_truth, un, tn, names):
    k = len(names)
    #norm_names = ["$L_{\infty}$", "$L_2$"]
    #norm_names = ["$L_{\infty}$"]
    norm_names = ["RSME", "$L_{\infty}$", "$L_2$"]
    for l in range(k):
        for i, norm in enumerate([RSME, L_infty, L_2]):
            plt.scatter(tn, norm(ground_truth, un[l]).cpu().numpy(), label=f"{names[l]}, {norm_names[i]}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy(solver, un, vn, names):
    k = len(names)
    assert len(un) == k
    assert len(vn) == k

    for l in range(k):
        es = []
        for i in range(solver.num_snapshots):
            u, v = un[l][i], vn[l][i]
            es.append(solver.energy(u, v).cpu().numpy())
        plt.plot(solver.tn.cpu().numpy()[
                ::solver.snapshot_frequency][1:len(es)],
                es[1:], label=names[l])
        
    plt.grid(True)
    plt.xlabel("T / [1]")
    plt.ylabel("E / [1]")
    plt.legend()
    plt.show()

def plot_energy_with_err(solver, un, vn, names): 
    k = len(names)
    assert len(un) == k
    assert len(vn) == k
    fig, axs = plt.subplots(figsize=(20, 20), ncols=2,)

    for l in range(k):
        es = []
        for i in range(solver.num_snapshots):
            u, v = un[l][i], vn[l][i]
            es.append(solver.energy(u, v).cpu().numpy())
        axs[0].plot(solver.tn.cpu().numpy()[
                ::solver.snapshot_frequency][1:len(es)],
                es[1:], label=names[l])

        axs[0].set_ylabel("E / [1]")
        axs[0].set_xlabel("T / [1]")

        axs[1].plot(solver.tn.cpu().numpy()[
                ::solver.snapshot_frequency][1:len(es)],
                np.log(np.abs((np.array(es) - es[0])))[1:], label=names[l])

        axs[1].set_ylabel("$\\log (E_0 - E(t))$ / [1]")

        
        axs[0].hlines(es[0], xmin=0 + solver.dt, xmax=solver.T, alpha=.5, linestyle='-.')
        for i in range(2):
            axs[i].grid(True)

    plt.suptitle(f"$h={solver.dx:.4f}$, $\\tau={solver.dt:.4f}$")
    plt.legend()

    plt.show()

def plot_density_flux(X, Y, density, flux, name):
    fig, axs = plt.subplots(figsize=(20, 20), ncols=3, subplot_kw={"projection":'3d'})
    axs[0].plot_surface(X, Y, density,
                    cmap='viridis')
    axs[0].set_title("energy density")
    axs[1].plot_surface(X, Y, flux,
                    cmap='viridis')
    axs[1].set_title("flux")

    axs[2].plot_surface(X, Y, torch.abs(flux-density),
                    cmap='viridis')
    axs[2].set_title("diff")
    fig.suptitle(name)
    plt.show()

def animate_density_flux(X, Y, densities, fluxes, name, num_snapshots):
    from matplotlib.animation import FuncAnimation
    fig, axs = plt.subplots(figsize=(20, 20), ncols=3, subplot_kw={"projection":'3d'})
     
    def update(frame):
        for i in range(3): axs[i].clear() 
        axs[0].plot_surface(X, Y, densities[frame],
                        cmap='viridis')
        axs[0].set_title("Hamiltonian density \n$\\frac{1}{2} \left ( u_t^2 + u_x^2 + u_y^2 \\right) + (1 - cos(u))$")
        axs[1].plot_surface(X, Y, fluxes[frame],
                        cmap='viridis')
        axs[1].set_title("Energy flux \n$u \Delta v + \\nabla u \cdot \\nabla v$")

        axs[2].plot_surface(X, Y, np.abs(fluxes[frame]-densities[frame]),
                        cmap='viridis')
        axs[2].set_title("diff") 
        fig.suptitle(f"{name}, t=")
    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()


def convergence_study(methods=['stormer-verlet-pseudo'],):
    initial_u =  circular_ring 
    initial_v = zero_velocity

    T = 3

    L = 4
    grid_range = [30, 40] 
    dxs = [2 * L / (n + 1) for n in grid_range]
    
    data = {}
    nts = [1000]
    for method in methods:
        for nt in nts:
            for i, n in enumerate(grid_range):
                nx = ny = n
                solver = SineGordonIntegrator(-L, L, -L, L, nx,
                                          ny, T, nt, initial_u, initial_v, step_method=method,
                                          #boundary_conditions='homogeneous-neumann',
                                          boundary_conditions='special-kink-bc',
                                          dtype=torch.float64,
                                          snapshot_frequency=10,
                                          c2 = 1,
                                          enable_energy_control=True,
                                          device='cpu')
                solver.evolve()
                E0 = solver.energy(solver.u[0], solver.v[0]).item()
                data_n = {'u': solver.u.cpu().numpy(), 'v': solver.u.cpu().numpy()}
                
                energies = np.array([solver.energy(u, v) for (u, v) in zip(solver.u, solver.v)])
                plt.plot(solver.tn.cpu().numpy()[
                        ::solver.snapshot_frequency][1:len(energies)],
                        np.log(np.abs(energies[1:] - E0)),
                        label=f"{method}: $h = {dxs[i]:.3f}$, $\\tau = {(T/nt):.3f}$")
                #with torch.no_grad():
                #    u, v = solver.u.clone().cpu().numpy(), solver.v.clone().cpu().numpy()
                #    animate(solver.X,
                #            solver.Y, u, solver.dt, solver.num_snapshots, solver.nt,
                #            title=f"{method}: $h = {dxs[i]:.3f}$, $\\tau = {(T/nt):.3f}$")

            
    plt.xlabel("$T$ / [1]")
    plt.ylabel("$\\log(E_0 - E(t))$ / [1]")
    plt.title("Circular ring soliton, no-flux")
    plt.legend()
    plt.grid(True)
    plt.show()



if False:#__name__ == '__main__':
    convergence_study(['gautschi-erkn', 'stormer-verlet-pseudo'])

if __name__ == '__main__':

    L = 3
    nx = ny = 32
    T = 10
    nt = 1000
    # Gautschi integrator has issues with the static breather -- let's check why!
    initial_u = circular_ring# static_breather# bubble # #static_breather #lbf_circular_ring 
    initial_v = zero_velocity

    #initial_u = kink
    #initial_v = kink_start
    
    implemented_methods = {
        #'ETD2-sparse': None,
        #'ETD1-sparse-opt': None,
        #'ETD1-krylov': None,
        #'Strang-split': None,
        #'HBVM': None,
        #'Strang-split': None,
        #'Energy-conserving-1': None,
        #'yoshida-4': None,
        #'gautschi-erkn': None,
        #'crank-nicholson': None,
        #'lattice-boltzmann': None,
        #'etd-spectral': None,
        'stormer-verlet-pseudo': None,
        #'avf-4th-fixed-point': None,
        #'bratsos-implicit': None,
        #'bratsos-corrector': None,
        'linear-DG': None,
        #'locally-energy-conserving-implicit': None,
        #'modified-law': None,
        #'AVF-explicit': None, 
        #'gauss-legendre': None,
        #'candy-neri6': None,
        #'RK4': None,
    }

    u_quiver, v_quiver = [], [] 
    for method in implemented_methods.keys():
        solver = SineGordonIntegrator(-L, L, -L, L, nx,
                                  ny, T, nt, initial_u, initial_v, step_method=method,
                                  boundary_conditions='homogeneous-neumann',
                                  #boundary_conditions='special-kink-bc',
                                  #c2 = lambda X, Y: .1 * torch.exp(-(X ** 2 + Y ** 2)),
                                  #c2 = lambda X, Y: torch.exp(-(1/X**2 + 1/Y**2)),
                                  dtype=torch.float64,
                                  snapshot_frequency=10,
                                  c2 = 1,
                                  enable_energy_control=False,
                                  device='cpu')
        solver.evolve()
        #if solver.iteration_stats:
        #    plt.plot(solver.iteration_stats)
        #    plt.show()
        #    plt.cla()


        #plot_density_flux(solver.X[1:-1,1:-1].cpu().numpy(), solver.Y[1:-1,1:-1].cpu().numpy(),
        #        solver.energy_density(solver.u[-1], solver.v[-1]),
        #        solver.energy_flux(solver.u[-1], solver.v[-1]),
        #        method
        #        )
        with torch.no_grad():
            #animate_density_flux(solver.X[1:-1,1:-1].cpu().numpy(), solver.Y[1:-1,1:-1].cpu().numpy(),
            #        [solver.energy_density(solver.u[i], solver.v[i]).cpu().numpy() for i in range(solver.num_snapshots)],
            #        [solver.energy_flux(solver.u[i], solver.v[i]).cpu().numpy() for i in range(solver.num_snapshots)], 
            #        method, solver.num_snapshots
            #        )

            implemented_methods[method] = solver.u.clone().cpu().numpy()
            u, v = solver.u.clone().cpu().numpy(), solver.v.clone().cpu().numpy()
            u_quiver.append(u)
            v_quiver.append(v)

 

    plot_energy_with_err(solver,
            torch.stack([torch.tensor(u) for u in u_quiver]),
            torch.stack([torch.tensor(v) for v in v_quiver]),
            list(implemented_methods.keys()))

    #gt = torch.stack([analytical_kink_solution(solver.X, solver.Y, t) for t in solver.tn[::solver.snapshot_frequency]])
    #error_norms(gt,
    #        torch.stack([torch.tensor(u) for u in u_quiver]),
    #        solver.tn.cpu().numpy()[::solver.snapshot_frequency],
    #        list(implemented_methods.keys()))

    #plot_phase_diagram_quiver(u_quiver, v_quiver, list(implemented_methods.keys()))
    #lyapunov_exponent(u_quiver, solver.tn.cpu().numpy()[::solver.snapshot_frequency][1:], list(implemented_methods.keys()) )

    #animate_comparison_with_kink_solution(solver.X, solver.Y, implemented_methods, solver.dt,
    #        solver.num_snapshots, solver.nt,)
    animate_comparison(solver.X, solver.Y, implemented_methods, solver.dt,
            solver.num_snapshots, solver.nt,)


    

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
