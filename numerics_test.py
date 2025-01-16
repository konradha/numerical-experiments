import torch
import torch.sparse as sparse
from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def lapl_eigenvalues(nx, ny, dx, dy, tau=None):
    """
        eigenvalues of discrete laplacian
        with gridsize nx x ny
        and spacing dx, dy
    """
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
    return ev


def fac(n):
    s = 1
    if n <= 1: return 1
    for i in range(2, n + 1):
        s *= i
    return s

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

def cosm_multiply(A, v, t, k=30):
    m = A.shape[0]
    beta = torch.norm(v)
    V, H = arnoldi_iteration(A, v/beta, k, t)
    tol = 1e-5
    rk = torch.zeros((k, v.shape[0]))
    for k in range(1, k+1):
        H_np = H.cpu().numpy()
        F_np = np.real(la.cosm(t * H_np[:k, :k]))
        F = torch.from_numpy(F_np)
        w = beta * V[:, :k] @ F[:, 0]
        error = torch.norm(t * H[k, k-1] * F[k-1, 0])
        rk[k-1] = w
        if error < tol:
            return w, rk
    return w, rk



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




# These values remain correct from the paper
THETA_VALUES = torch.tensor([
    5.464021317208652e-08, 2.890823622487534e-04, 4.779406432823473e-03,
    2.025757158516201e-02, 5.269011726519909e-02, 1.017468817136691e-01,
    1.667687913202508e-01, 2.468074880011088e-01, 3.412701939251897e-01,
    4.493398348939874e-01, 5.704925265636802e-01, 7.040554433701242e-01,
    8.493100481817480e-01, 1.005572263543877e+00, 1.172006873999567e+00,
    1.347855579140176e+00, 1.532372545067401e+00, 1.724829240477342e+00,
    1.924513241192994e+00, 2.130723071609624e+00, 2.342776439616025e+00,
    2.560016586507728e+00, 2.781814356459840e+00, 3.007562628094540e+00,
    3.236672697766150e+00
])

class TrigonometricFunction(Enum):
    COS_SIN = "cos.sin"
    COSH_SINH = "cosh.sinh"
    COS_SINC = "cos.sinc"
    COSH_SINCH = "cosh.sinch"
    COS_SINC_SQRT = "cos.sinc.sqrt"
    COSH_SINCH_SQRT = "cosh.sinch.sqrt"

@dataclass
class TrigonometricParameters:
    uses_sqrt: bool
    is_hyperbolic: bool
    uses_shifted_spectrum: bool

def compute_sparse_trace(matrix) -> torch.Tensor:
    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    diagonal_elements = indices[0] == indices[1]
    if not torch.any(diagonal_elements):
        return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)
    return values[diagonal_elements].sum()

# matrix is a sparse tensor
def estimate_matrix_power_norm(matrix, power: int, num_samples: int = 1) -> Tuple[torch.Tensor, int]:
    dimension = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype
    
    def apply_power(vectors: torch.Tensor) -> torch.Tensor:
        result = vectors
        for _ in range(power):
            result = sparse.mm(matrix, result)
        return result
    
    sample_vectors = torch.randn(dimension, num_samples, device=device, dtype=dtype)
    sample_vectors /= torch.norm(sample_vectors, p=1, dim=0, keepdim=True)
    
    previous_estimate = torch.tensor(0.0, device=device, dtype=dtype)
    mv_count = 0
    
    for iteration in range(5):
        powered_vectors = apply_power(sample_vectors)
        mv_count += num_samples * power
        
        vector_norms = torch.norm(powered_vectors, p=1, dim=0)
        current_estimate = torch.max(vector_norms)
        
        if iteration > 0 and (current_estimate - previous_estimate).abs() <= current_estimate * 1e-2:
            break
            
        previous_estimate = current_estimate
        signs = torch.sign(powered_vectors)
        max_indices = torch.argmax(torch.abs(powered_vectors), dim=0)
        
        sample_vectors = torch.zeros_like(powered_vectors)
        sample_vectors[max_indices, torch.arange(num_samples)] = signs[max_indices, torch.arange(num_samples)]
        sample_vectors = apply_power(sample_vectors)
        mv_count += num_samples * power
        
        sample_vectors /= torch.norm(sample_vectors, p=1, dim=0, keepdim=True)
        
    return current_estimate, mv_count

import torch
import torch.sparse as sparse
from enum import Enum
from typing import Tuple
import numpy as np

def create_degree_matrix(alpha_estimates: torch.Tensor, theta_bounds: torch.Tensor,
                      max_degree: int) -> torch.Tensor:
    parameter_count = len(alpha_estimates) + 1
    degree_matrix = torch.zeros(max_degree, parameter_count - 1,
                              device=alpha_estimates.device, dtype=alpha_estimates.dtype)

    for p in range(2, parameter_count + 1):
        start_degree = p * (p-1) - 1
        for degree in range(start_degree, max_degree):
            if degree < max_degree:
                degree_matrix[degree, p-2] = alpha_estimates[p-2] / theta_bounds[degree]

    return degree_matrix

def compute_matrix_function(t: float, matrix, vectors: torch.Tensor, 
                          func_type: str = "cos.sin") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute matrix function action using the Al-Mohy algorithm.
    """
    # Parameters for algorithm
    params = TrigonometricParameters(
        uses_sqrt="sqrt" in func_type,
        is_hyperbolic="h" in func_type,
        uses_shifted_spectrum=func_type in ["cos.sin", "cosh.sinh"]
    )
    
    dimension = matrix.shape[0]
    dtype = matrix.dtype
    device = matrix.device
    
    if params.uses_shifted_spectrum: 
        matrix_mean = compute_sparse_trace(matrix) / dimension
        indices = torch.arange(dimension, device=device)
        identity_indices = torch.stack([indices, indices])
        identity_values = torch.ones(dimension, device=device, dtype=dtype)
        sparse_identity = torch.sparse_coo_tensor(identity_indices, identity_values, matrix.shape)
        shifted_matrix = matrix - matrix_mean * sparse_identity
        t_mean = t * matrix_mean
    else:
        shifted_matrix = matrix
        t_mean = 0
    
    degree_matrix, mv_count, alpha_values, norm_type = select_taylor_degree(
        shifted_matrix, vectors, params.uses_sqrt
    )
    
    min_costs = torch.min(degree_matrix, dim=1)[0]
    optimal_degree = torch.nonzero(min_costs)[0]
    scaling_factor = max(1, int(torch.ceil(min_costs[optimal_degree]).item()))
    
    scaled_time = t / scaling_factor
    intermediate_vectors = vectors.clone()
    
    for step in range(scaling_factor):
        current_vectors = intermediate_vectors.clone()
        base_vectors = current_vectors / 2 if step == scaling_factor - 1 else torch.zeros_like(current_vectors)
        
        for k in range(optimal_degree):
            # Matrix power iteration
            
            #current_vectors = shifted_matrix @ current_vectors
            current_vectors = shifted_matrix @ current_vectors * \
                            (scaled_time**2 / ((2*k+1)*(2*k+2)))
            
            sign = (-1)**(k * (0 if params.is_hyperbolic else 1))
            intermediate_vectors = intermediate_vectors + sign * current_vectors
        
        if params.uses_shifted_spectrum:
            shifted_term = shifted_matrix @ (intermediate_vectors * scaled_time)
            if params.is_hyperbolic:
                intermediate_vectors = torch.cosh(t_mean / scaling_factor) * intermediate_vectors + \
                                    torch.sinh(t_mean / scaling_factor) * shifted_term
            else:
                intermediate_vectors = torch.cos(t_mean / scaling_factor) * intermediate_vectors - \
                                    torch.sinh(t_mean / scaling_factor) * shifted_term
    
    return intermediate_vectors, intermediate_vectors / scaling_factor

def select_taylor_degree(operator, vectors: torch.Tensor,
                        use_sqrt: bool = False, max_degree: int = 25,
                        num_terms: int = 5) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
    """paper Section 3"""
    if num_terms < 2 or max_degree > 25 or max_degree < num_terms * (num_terms - 1):
        raise ValueError("Invalid degree parameters")
    
    power_scaling = 0.5 if use_sqrt else 1.0
    theta_bounds = THETA_VALUES[:max_degree]
    
    operator_norm = sparse.mm(operator.t(), operator).to_dense().diag().max().sqrt()
    scaled_norm = operator_norm ** power_scaling
    operation_count = 0

    k = vectors.shape[1] if vectors.dim() > 1 else 1
    
    cost_bound = 2 * 2 * num_terms * (num_terms + 3) / (max_degree * k) - 1
    
    if scaled_norm <= theta_bounds[max_degree - 1] * cost_bound:
        alpha_estimates = scaled_norm * torch.ones(num_terms - 1, device=operator.device, dtype=operator.dtype)
        return create_degree_matrix(alpha_estimates, theta_bounds, max_degree), operation_count, alpha_estimates, 0
    
    norm_estimates = torch.zeros(num_terms, device=operator.device, dtype=operator.dtype)
    for p in range(num_terms):
        power = int(2 * power_scaling * (p + 1))
        power_norm, matrix_ops = estimate_matrix_power_norm(operator, power)
        operation_count += matrix_ops
        norm_estimates[p] = power_norm ** (1 / (2*p + 2))
    
    alpha_estimates = torch.maximum(norm_estimates[:-1], norm_estimates[1:])
    return create_degree_matrix(alpha_estimates, theta_bounds, max_degree), operation_count, alpha_estimates, 2

def select_parameters(t: float, A: torch.Tensor) -> tuple:
    normA = norm1_sparse(A)
    theta_m = THETA_VALUES
    m = None
    for idx, theta in enumerate(theta_m):
        if normA <= theta:
            m = idx + 1
            break 
    if m is None:
        m = len(theta_m)
    s = max(1, int(np.ceil(normA.cpu().numpy() * abs(t) / theta_m[m-1].cpu().numpy()))) 
    return m, s

def norm1_sparse(A):
    if not A.is_sparse:
        raise ValueError("Input tensor must be sparse")
    A = A.coalesce()
    abs_values = torch.abs(A.values())
    column_sums = torch.zeros(A.size(1), dtype=A.dtype)
    for idx, col in enumerate(A.indices()[1]):
        column_sums[col] += abs_values[idx]
    return torch.max(column_sums)


def funmv(t: float, A: torch.Tensor, b: torch.Tensor, flag: str = "cos.sin") -> tuple:
    shift = flag in ["cos.sin", "cosh.sinh"]
    k0 = 1 if flag in ["cos.sin", "cos.sinc", "cos.sinc.sqrt"] else 0
    nx, ny = A.shape
    assert nx == ny
    n = nx
    Id = torch.sparse.spdiags(torch.ones(nx * ny), offsets=torch.tensor(0), shape=(nx, ny)).coalesce()
    

    if shift:
        mu = compute_sparse_trace(A) / float(n)
        mu = float(mu)     
        A = A - mu * Id
        tmu = t * mu
        
    if t == 0:
        m = 0
        s = 1
    else:
        m, s = select_parameters(t, A)

    undo_inside = False
    undo_outside = False
    if shift and flag == "cos.sin":
        if not np.isreal(tmu):
            cosmu = torch.cos(torch.tensor(tmu/s))
            sinmu = torch.sin(torch.tensor(tmu/s))
            undo_inside = True
        elif abs(tmu) > 0:
            cosmu = torch.cos(torch.tensor(tmu))
            sinmu = torch.sin(torch.tensor(tmu))
            undo_outside = True
        #if abs(tmu) > 0:
        #    cosmu = torch.cos(torch.tensor(tmu/s))
        #    sinmu = torch.sin(torch.tensor(tmu/s)) 
        #    undo_outside = True 
        #    
        #else:
        #    cosmu = torch.cos(torch.tensor(tmu))
        #    sinmu = torch.sin(torch.tensor(tmu)) 
        #    undo_inside = True
             
    T0 = torch.zeros_like(b)
    if s % 2 == 0:
        T0 = b / 2

    U = T0
    T1 = b.clone()

    for i in range(1, s + 2):
        if i == s + 1:
            U = 2 * U
            T1 = U

        V = T1.clone()
        if undo_inside:
            Z = T1.clone()
        B = T1.clone()
        c1 = torch.norm(B, float('inf'))

        for k in range(1, m + 2):
            if flag:
                B = A @ B
            beta = 2*k
            if i <= s:
                gamma = beta - 1
                q = 1.0 / (beta + 1)
            else:
                gamma = beta + 1
                q = gamma
            
            B = A @ B * ((t/s)**2 / (beta * gamma))
            c2 = torch.norm(B, float('inf'))
            V = V + ((-1)**(k*k0)) * B
            if undo_inside:
                Z = Z + (((-1)**(k*k0)) * q * B)

            if c1 + c2 <= torch.norm(V, float('inf')) * 1e-10:
                break
            c1 = c2

        if undo_inside:
            if i <= s:
                V = V * cosmu + A @ (Z * (((-1)**k0) * t * sinmu/s))
            else:
                V = A @ (V * (t * cosmu/s)) + Z * sinmu


        if i == 1:
            T2 = V
        elif i <= s:
            T2 = 2*V - T0

        if i <= s-1 and ((s % 2 == 0) != (i % 2 == 0)):
            U = U + T2

        T0 = T1
        T1 = T2

    C = T2

    if undo_inside:
        S = V
    elif shift:
        S = A @ (V * (t/s))
    else:
        S = V/s

    if undo_outside:
        C = cosmu * C + (((-1)**k0) * sinmu) * S
        S = sinmu * T2 + cosmu * S

    return C, S

if __name__ == '__main__':
    nx = ny = 32
    L = 3

    xn = yn = torch.linspace(-L, L, nx)
    X, Y = torch.meshgrid(xn, yn, indexing='ij')
    dx = dy = 2 * L / (nx + 1)

    u = torch.arctan(torch.exp(X + Y))
    L = build_D2(nx-2, ny-2, dx, dy, torch.float32)
    evs = lapl_eigenvalues(nx , ny , dx, dy)
    k = 100

    cos_fft   = (torch.fft.ifft2(torch.cos(evs) * torch.fft.fft2(u)))
    cos_lfa, r= cosm_multiply(L, u.reshape(nx*ny), 1, k=k)
    cos_lfa   = cos_lfa.reshape((nx, ny))
    cos_dense = la.cosm(L.to_dense().cpu().numpy())@(u.reshape(nx*ny).cpu().numpy())
    cos_dense = torch.from_numpy(cos_dense.reshape((nx, ny)))


    cos_mohy, _ = funmv(1, L, u.reshape(nx*ny), "cos.sin")

    fig, axs = plt.subplots(figsize=(20, 20), ncols=5, subplot_kw={"projection":'3d'})
    axs[0].plot_surface(X, Y, u, cmap='viridis')
    axs[1].plot_surface(X, Y, cos_fft, cmap='viridis')
    axs[2].plot_surface(X, Y, cos_lfa, cmap='viridis')
    axs[3].plot_surface(X, Y, cos_dense, cmap='viridis')
    axs[4].plot_surface(X, Y, cos_mohy.reshape((nx, ny)), cmap='viridis')

    axs[0].set_title('$u$')
    axs[1].set_title('$\mathbf{F}^{-1} [cos(L)\hat u]$')
    axs[2].set_title('Krylov application')
    axs[3].set_title('scipy, dense')
    axs[4].set_title('Al-Mohy')

    plt.show()

