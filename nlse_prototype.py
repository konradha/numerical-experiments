import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import scipy

def torch_sparse_to_scipy_sparse(torch_sparse):
    indices = torch_sparse.coalesce().indices()
    values = torch_sparse.coalesce().values()
    shape = torch_sparse.shape
    return scipy.sparse.coo_matrix(
        (values.cpu().numpy(),
         (indices[0].cpu().numpy(), indices[1].cpu().numpy())),
        shape=shape)

def scipy_sparse_to_torch_sparse(scipy_sparse):
    scipy_sparse = scipy_sparse.tocoo() 
    indices = torch.from_numpy(
        np.vstack((scipy_sparse.row, scipy_sparse.col))
    ).long()
    values = torch.from_numpy(scipy_sparse.data)
    shape = torch.Size(scipy_sparse.shape) 
    return torch.sparse_coo_tensor(indices, values, shape)

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

    indices_main = torch.arange(N, dtype=torch.long)
    indices_off1 = torch.arange(1, N, dtype=torch.long)
    indices_off2 = torch.arange(0, N - 1, dtype=torch.long)

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

def build_D2_periodic(nx, ny, dx, dy, dtype):
    assert nx == ny
    assert dx == dy
    N = (nx + 2) ** 2
    middle_diag = -4 * torch.ones(nx + 2, dtype=dtype)
    diag = torch.cat([middle_diag] * (nx + 2))
    offdiag_pos = torch.ones(N - 1, dtype=dtype)
    inner_outer_identity = torch.ones(N - (nx + 2), dtype=dtype)
    indices_main = torch.arange(N, dtype=torch.long)
    indices_off1 = torch.arange(1, N, dtype=torch.long)
    indices_off2 = torch.arange(0, N - 1, dtype=torch.long)
    row_indices = torch.cat([
        indices_main,
        indices_off1, indices_off2,
        indices_main[:-(nx+2)],
        indices_main[nx+2:]
    ])
    col_indices = torch.cat([
        indices_main,
        indices_off2, indices_off1,
        indices_main[nx+2:],
        indices_main[:-nx-2]
    ])
    values = torch.cat([
        diag,
        offdiag_pos, offdiag_pos,
        inner_outer_identity,
        inner_outer_identity
    ])
    left_edge = torch.arange(0, N, nx + 2, dtype=torch.long)
    right_edge = torch.arange(nx + 1, N, nx + 2, dtype=torch.long)
    row_indices = torch.cat([
        row_indices,
        left_edge, right_edge
    ])
    col_indices = torch.cat([
        col_indices,
        right_edge, left_edge
    ])
    pb_values = torch.ones(2 * len(left_edge), dtype=dtype)
    values = torch.cat([values, pb_values])
    top_edge = torch.arange(0, nx + 2, dtype=torch.long)
    bottom_edge = torch.arange(N - (nx + 2), N, dtype=torch.long)
    row_indices = torch.cat([
        row_indices,
        top_edge, bottom_edge
    ])
    col_indices = torch.cat([
        col_indices,
        bottom_edge, top_edge
    ])
    pb_values = torch.ones(2 * len(top_edge), dtype=dtype)
    values = torch.cat([values, pb_values])
    L = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=(N, N),
        dtype=dtype,
    )
    L *= (1 / dx) ** 2
    return L

def build_D2_radiation(nx, ny, dx, dy, dtype, alpha=1.0):
    assert nx == ny
    assert dx == dy
    N = (nx + 2) ** 2
    middle_diag = -4 * torch.ones(nx + 2, dtype=dtype)
    middle_diag[0] = middle_diag[-1] = -3
    diag = torch.cat([middle_diag] * (nx + 2))
    diag[:nx+2] = -3 
    diag[-nx-2:] = -3  
    diag[::nx+2] = -3 
    diag[nx+1::nx+2] = -3  
    diag[0] = diag[nx+1] = diag[-nx-2] = diag[-1] = -2 

    offdiag_pos = torch.ones(N - 1, dtype=dtype)
    inner_outer_identity = torch.ones(N - (nx + 2), dtype=dtype)

    indices_main = torch.arange(N, dtype=torch.long)
    indices_off1 = torch.arange(1, N, dtype=torch.long)
    indices_off2 = torch.arange(0, N - 1, dtype=torch.long)

    row_indices = torch.cat([
        indices_main,
        indices_off1, indices_off2,
        indices_main[:-(nx+2)],
        indices_main[nx+2:]
    ])

    col_indices = torch.cat([
        indices_main,
        indices_off2, indices_off1,
        indices_main[nx+2:],
        indices_main[:-nx-2]
    ])

    values = torch.cat([
        diag,
        offdiag_pos, offdiag_pos,
        inner_outer_identity,
        inner_outer_identity
    ])

    L = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=(N, N),
        dtype=dtype,
    )
    L *= (1 / dx) ** 2
    return L


def animate(X, Y, data, dt, num_snapshots, nt, title):
    mxl, mxr = x == -Lx /2, x == Lx /2 
    myl, myr = y == -Ly /2, y == Ly /2

    
    
    #prob_density = [
    #        (np.log(np.min(d.cpu().numpy().real ** 2 + d.cpu().numpy().imag ** 2)),
    #            np.log(np.max(d.cpu().numpy().real ** 2 + d.cpu().numpy().imag ** 2)))
    #        for d in data]
    #vmin = min(pd[0] for pd in prob_density)
    #vmax = max(pd[1] for pd in prob_density)
    
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    
    def update(frame):
        ax.clear() 
        #ax.scatter(X, Y, mxl, color='r')    
        #ax.scatter(X, Y, myl, color='b')
        # [10:-10,10:-10]
        ax.plot_surface(X, Y,
                (data[frame].cpu().numpy().real ** 2 + data[frame].cpu().numpy().imag ** 2),
                #data[frame].cpu().numpy().real,
                cmap='jet',)
        #ax.set_zlim(vmin, vmax)
        #ax.set_title(f"{title}, t={(frame * dt * (nt / num_snapshots)):.2f}")

    fps = 3
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    #ani.save("scattering.mp4")
    plt.show()

def u_xx_yy(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1, 1:-1] = (
        (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2) +
        (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
    )
    return uxx_yy

def gaussian_center(x, y, A, sigma, kx, ky, dx, dy):
    un1 =  A * torch.exp(-((x - 3) ** 2 +  (y - 3.) ** 2) / 4 / sigma ** 2) * torch.exp(1j * (x + y))  
    un2 =  A * torch.exp(-((x + 3)** 2 +  (y + 3.) ** 2) / 4 / sigma ** 2) * torch.exp(-1j * (x + y))
    un = un1 + un2
    norm = torch.sqrt(torch.sum(un.real ** 2 + un.imag ** 2))
    return un / norm


def estimate_k(u, dx, dy):
    fx = torch.fft.fft2(u)
    fx = torch.fft.fftshift(fx)
    kx = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(u.shape[0], dx))
    ky = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(u.shape[1], dy))
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2)
    weights = torch.abs(fx)**2
    k_avg = torch.sum(K * weights) / torch.sum(weights)
    return k_avg.real

#def update_laplacian(L, nx, ny, dx, dy, dt, u_torch, u_numpy, i=1):
#    du_dx_left      = torch.mean((u_torch[0, :] - u_torch[1, :])/dx)
#    du_dy_bottom    = torch.mean((u_torch[:, 0] - u_torch[:, 1])/dy)
#    
#    du_dx_right = -torch.mean((u_torch[-1, :] - u_torch[-2, :])/dx)
#    du_dy_top   = -torch.mean((u_torch[:, -1] - u_torch[:, -2])/dy)
#  
#    left_boundary   = np.arange(0, (nx+2)*ny, nx+2)
#    right_boundary  = np.arange(nx+1, (nx+2)*ny, nx+2)
#    bottom_boundary = np.arange(0, nx+2)
#    top_boundary    = np.arange((ny-1)*(nx+2), ny*(nx+2))
#
#    L.data[left_boundary]   *= (1 -  dt*du_dx_left.cpu().numpy())
#    L.data[right_boundary]  *= (1 +  dt*du_dx_right.cpu().numpy())
#    L.data[bottom_boundary] *= (1 -  dt*du_dy_bottom.cpu().numpy())
#    L.data[top_boundary]    *= (1 +  dt*du_dy_top.cpu().numpy())
#
#    if i == 1: 
#        diag_corners = np.array([0, nx+1, (ny-1)*(nx+2), ny*(nx+2)-1])
#        L.data[diag_corners] *= 0.5  
#
#    return L

def update_laplacian_close(L, nx, ny, dx, dy, dt, u_torch, u_numpy, i=1):
    du_dx_left      = (u_torch[0, :] - u_torch[1, :])/dx 
    du_dy_bottom    = (u_torch[:, 0] - u_torch[:, 1])/dy
    du_dx_right     = -(u_torch[-1, :] - u_torch[-2, :])/dx
    du_dy_top       = -(u_torch[:, -1] - u_torch[:, -2])/dy

    left_boundary   = np.arange(0, (nx+2)*ny, nx+2)
    right_boundary  = np.arange(nx+1, (nx+2)*ny, nx+2)
    bottom_boundary = np.arange(0, nx+2)
    top_boundary    = np.arange((ny-1)*(nx+2), ny*(nx+2))

    L.data[left_boundary]   *= (1 - dt*du_dx_left.cpu().numpy()[1:-1])
    L.data[right_boundary]  *= (1 - dt*du_dx_right.cpu().numpy()[1:-1])
    L.data[bottom_boundary] *= (1 - dt*du_dy_bottom.cpu().numpy())
    L.data[top_boundary]    *= (1 - dt*du_dy_top.cpu().numpy())
    
    diag_corners = np.array([0, nx+1, (ny-1)*(nx+2), ny*(nx+2)-1])
    L.data[diag_corners] *= 0.5  

    return L

def update_laplacian(L, nx, ny, dx, dy, dt, u_torch, u_numpy, i=1): 
    left_boundary = np.arange(0, (nx+2)*ny, nx+2)
    right_boundary = np.arange(nx+1, (nx+2)*ny, nx+2)
    bottom_boundary = np.arange(0, nx+2)
    top_boundary = np.arange((ny-1)*(nx+2), ny*(nx+2))
    all_boundaries = np.concatenate([left_boundary, right_boundary, bottom_boundary, top_boundary])
    L.data[all_boundaries] *= 2/3
    return L

def build_D2_pml(nx, ny, dx, dy, dtype, pml_width=10, sigma_max=2.0):
    assert nx == ny
    assert dx == dy
    N = (nx + 2) ** 2 
    sigma_x = torch.zeros((ny+2, nx+2), dtype=torch.complex128)
    sigma_y = torch.zeros((ny+2, nx+2), dtype=torch.complex128)
    
    for i in range(nx+2):
        for j in range(ny+2):
            if i < pml_width:
                d = (pml_width - i)/pml_width
                sigma_x[j,i] = sigma_max * (d**2)
            elif i >= nx+2-pml_width:
                d = (i - (nx+2-pml_width))/pml_width
                sigma_x[j,i] = sigma_max * (d**2) 
            if j < pml_width:
                d = (pml_width - j)/pml_width
                sigma_y[j,i] = sigma_max * (d**2)
            elif j >= ny+2-pml_width:
                d = (j - (ny+2-pml_width))/pml_width
                sigma_y[j,i] = sigma_max * (d**2)

    omega = 60*torch.pi 
    s_x = 1/(1 + 1j*sigma_x/omega)
    s_y = 1/(1 + 1j*sigma_y/omega)
    
    A = s_y/s_x
    B = s_x/s_y  
    C = s_x*s_y

    middle_diag = torch.zeros((ny+2, nx+2), dtype=torch.complex128)
    for j in range(ny+2):
        for i in range(nx+2):
            middle_diag[j,i] = -2*(A[j,i] + B[j,i])

    diag = middle_diag.flatten()
    offdiag_x = torch.zeros(N-1, dtype=torch.complex128) 
    offdiag_y = torch.zeros(N-(nx+2), dtype=torch.complex128)

    for j in range(ny+2):
        for i in range(nx+1):
            idx = j*(nx+2) + i
            offdiag_x[idx] = 0.5*(A[j,i] + A[j,i+1])

    for j in range(ny+1):
        for i in range(nx+2):
            idx = j*(nx+2) + i
            offdiag_y[idx] = 0.5*(B[j,i] + B[j+1,i])
            
    indices_main = torch.arange(N, dtype=torch.long)
    indices_off1 = torch.arange(1, N, dtype=torch.long)
    indices_off2 = torch.arange(0, N-1, dtype=torch.long)
    
    row_indices = torch.cat([
        indices_main,
        indices_off1, indices_off2,
        indices_main[:-(nx+2)],
        indices_main[nx+2:]
    ])
    
    col_indices = torch.cat([
        indices_main,
        indices_off2, indices_off1,
        indices_main[nx+2:],
        indices_main[:-nx-2]
    ])
    
    values = torch.cat([
        diag,
        offdiag_x, offdiag_x,
        offdiag_y, offdiag_y
    ])
    
    L = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=(N, N),
        dtype=torch.complex128
    )
    
    L *= (1/dx)**2
    return L

class NLSEStepper:
    def __init__(self, nx, ny, Lx, Ly, dt, nt, initial_u, snapshot_freq=10, dtype=torch.float64):
        self.dtype = dtype
        self.buf = torch.zeros((nx, ny), dtype=self.dtype)
        self.un_real = torch.zeros((nt // snapshot_freq, nx, ny)) 
        self.un_imag = torch.zeros((nt // snapshot_freq, nx, ny))
        self.un = self.un_real + 1j * self.un_imag

        self.u0 = initial_u.clone()
        self.dt = torch.tensor(dt)

        #self.L = (build_D2(nx-2, ny-2, Lx/(nx-1), Ly/(ny-1), torch.complex128))
        self.L = build_D2_radiation(nx-2, ny-2, Lx/(nx-1), Ly/(ny-1), torch.complex128) 
        
        self.Id = torch.sparse_coo_tensor(
               torch.tensor([[i, i] for i in range(nx * ny)]).T,
               torch.ones(nx *ny, dtype=torch.complex128),
               (nx*ny, nx*ny))
        self.Id_np = torch_sparse_to_scipy_sparse(self.Id)
        
        self.L_lhs = torch_sparse_to_scipy_sparse( -.5 * self.dt * self.L + 1.j * self.Id).tocsr() 

        self.L_split_lhs = torch_sparse_to_scipy_sparse(self.Id + .5j * float(self.dt) * self.L).tocsr()
        self.LU = scipy.sparse.linalg.splu(self.L_split_lhs)
        self.L = torch_sparse_to_scipy_sparse(self.L).tocsr()

         


        self.sf = snapshot_freq

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nt = nt

        self.dx = torch.tensor(Lx / nx)
        self.dy = torch.tensor(Ly / ny)
        

        self.kx = 2*np.pi*torch.fft.fftfreq(nx, self.dx)
        self.ky = 2*np.pi*torch.fft.fftfreq(ny, self.dy)

        self.xn = torch.linspace(-Lx/2, Lx/2, nx)
        self.yn = torch.linspace(-Ly/2, Ly/2, ny)
        self.x, self.y = torch.meshgrid(self.xn, self.yn, indexing='ij')
        self.R = torch.sqrt(self.x ** 2 + self.y ** 2)

    

        """
        Time split

        u_n = H_1(t/2) o H_2(t) o H_1(t/2) u_{n-1}

        """    
    def step_full(self, u, i=1):
        def barrier(x, y):
            # V(x,y) = V₀/2 * (tanh((w - |x + y|)/ε) + 1)
            V0 = 1e1
            L = x.max() - x.min() 
            d = torch.abs(y - x) / L 
            w = .1
            V = V0 * torch.exp(-(d/w)**2) 
            #return V0 * d # hard barrier
            return V # smoothened barrier

        def create_double_slit(x, y, V0=1e10):
            mask = (torch.abs(x) < 0.1).float()
            
            slit1 = ((y < -3.1) | (y > -2.9)).float()
            slit2 = ((y < 2.9) | (y > 3.1)).float()
            
            V = V0 * mask * slit1 * slit2
            return V

        # fully complex evolution
        rho_half = (u.real ** 2 + u.imag ** 2) * u
        us = torch.exp(-.5j * self.dt * rho_half) * u 
        #L = update_laplacian(self.L.tocoo(), self.nx-2, self.ny-2, float(self.dx), float(self.dy),
        #        float(self.dt), us, us.reshape(self.nx ** 2).cpu().numpy(), i)
        L = self.L
        
        uss = us.reshape((self.nx * self.ny)).cpu().numpy()
        usss = scipy.sparse.linalg.expm_multiply(-L * 1j * float(self.dt),  uss)
        
        usss = torch.from_numpy(usss).reshape((self.nx, self.ny))
        rho_half = (usss.real ** 2 + usss.imag ** 2) * usss
        un = torch.exp(-.5j * self.dt * rho_half) * usss
        
        # some work to be done to understand this better
        # need solid transformation to not have any complex values involved
        #print(torch.norm(
        #    (
        #    torch.cos(-.5 * self.dt * rho_half) * usss - 1j * torch.sin(-.5 * self.dt * rho_half) * usss
        #    ) - un))
        return un

    def step_simpler(self, u, i=1.):
        rho_half = (u.real ** 2 + u.imag ** 2) * u
        us = torch.exp(-.5j * self.dt * rho_half) * u 


        rhs = 1.j * us + .5 * self.dt * u_xx_yy(self.buf, us, self.dx, self.dy)
        rhs = rhs.reshape(self.nx * self.ny).cpu().numpy()

        L = update_laplacian(self.L.tocoo(), self.nx-2, self.ny-2, float(self.dx), float(self.dy),
                float(self.dt), us, us.reshape(self.nx ** 2).cpu().numpy(), i)

        self.L = L
        self.L_lhs = -.5 * float(self.dt) * self.L + 1.j * self.Id_np

        us = scipy.sparse.linalg.spsolve(self.L_lhs, rhs)
        us = torch.from_numpy(us).reshape((self.nx, self.ny))

        
        rho_half = (us.real ** 2 + us.imag ** 2) * us
        un = torch.exp(-.5j * self.dt * rho_half) * us 
        return un
        

        
    def step_split(self, u, i):
        # split evolution 
        rho_half = torch.abs(u)
        us = torch.exp(-.5j * self.dt * rho_half) * u 
        
        rhs = us - 5.j * float(self.dt) * u_xx_yy(self.buf, us, self.dx, self.dy)
        rhs = rhs.reshape(self.nx ** 2).cpu().numpy() 
        uss = self.LU.solve(rhs)
        uss = torch.from_numpy(uss).reshape((self.nx, self.ny))
        usss = uss

        rho_half = torch.abs(usss)
        un = torch.exp(-.5j * self.dt * rho_half) * usss
        
        return un


    def neumann_bc(self, u):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

    def radiation_bc(self, u):
        k = estimate_k(u, self.dx, self.dy)
        r_left = self.R[0, :]
        r_right = self.R[-1, :]
        dr_left = (u[1, :] - u[0, :]) / self.dx
        u[0, :] = u[1, :] / (1 + self.dx * (1j * k + 1/(2*r_left)))
        dr_right = (u[-1, :] - u[-2, :]) / self.dx
        u[-1, :] = u[-2, :] / (1 - self.dx * (1j * k + 1/(2*r_right)))
        r_top = self.R[:, 0]
        r_bottom = self.R[:, -1]
        dr_top = (u[:, 1] - u[:, 0]) / self.dy
        u[:, 0] = u[:, 1] / (1 + self.dy * (1j * k + 1/(2*r_top)))
        dr_bottom = (u[:, -1] - u[:, -2]) / self.dy
        u[:, -1] = u[:, -2] / (1 - self.dy * (1j * k + 1/(2*r_bottom)))
        

    def evolve(self):
        u = self.u0
        self.un[0] = u
        for i in tqdm(range(1, self.nt)):
            #u = self.step_simpler(u) 
            #u = self.step_full(u, i)
            u = self.step_split(u, i)
            #plt.imshow(torch.sqrt(u.real ** 2 + u.imag ** 2))
            #plt.show()
            #self.neumann_bc(u)
            #self.radiation_bc(u)
            if i % self.sf == 0:
                
                self.un[i // self.sf] = torch.abs(
                        u.clone() - self.step_full(u, i)
                        )

    def energy(self, u):
        ux = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * self.dx)
        uy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * self.dy)
 
        grad_term = torch.sum(torch.abs(ux)**2 + torch.abs(uy)**2)  
        nonlin_term = -0.5 * torch.sum(torch.abs(u)**4) 
        return (grad_term + nonlin_term) * self.dx * self.dy * .5


def vortex_soliton(x, y, t=0, m=1., mu=1.):
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    f = torch.tanh(r) * torch.exp(-mu*t + 1j*m*theta)
    return f

def dnoidal_wave(x, y, kx, ky, t=0., omega=.6):
    dn = torch.cos(kx*x + 1j * ky*y - omega*t)
    return torch.sqrt(2*(kx**2 + ky ** 2)) * dn

def two_soliton_collision(x, y, t, v1, v2):
    x1 = x - v1*t
    x2 = x - v2*t
    s1 = torch.exp(-(x1**2 + y**2)) * torch.exp(1j*v1*x)
    s2 = torch.exp(-(x2**2 + y**2)) * torch.exp(1j*v2*y)
    return s1 + s2

def sech(x):
    return 1. / torch.cosh(x)

def create_2d_dark_solitons(x, y, soliton_params=None):
    if soliton_params is None:
        soliton_params = [
            {'pos_x': -5.0, 'pos_y': -5.0, 'vel_x': 0.3, 'vel_y': 0.3, 'B': 0.8, 'theta': 0.0},
            {'pos_x': 5.0, 'pos_y': -5.0, 'vel_x': -0.3, 'vel_y': 0.3, 'B': 0.8, 'theta': 2*torch.pi/3},
            {'pos_x': 0.0, 'pos_y': 5.0, 'vel_x': 0.0, 'vel_y': -0.3, 'B': 0.8, 'theta': -2*torch.pi/3}
        ]

    psi = torch.ones_like(x, dtype=torch.complex64)

    for params in soliton_params:
        pos_x, pos_y = params['pos_x'], params['pos_y']
        vel_x, vel_y = params['vel_x'], params['vel_y']
        B = params['B']
        theta = params.get('theta', 0.0)
        theta = torch.tensor(theta)
        x_rot = (x - pos_x) * torch.cos(theta) + (y - pos_y) * torch.sin(theta)
        y_rot = -(x - pos_x) * torch.sin(theta) + (y - pos_y) * torch.cos(theta)

        T = x_rot + vel_x * y_rot
        amplitude = torch.sqrt(1 - B * B * sech(T)**2)
        phase = torch.arcsin(B * torch.tanh(T) / torch.sqrt(1 - B * B * sech(T)**2))
        soliton = amplitude * torch.exp(1j * phase)
        psi *= soliton

    return psi

def create_2d_dark_solitons_diff(x, y, soliton_params = None) -> torch.Tensor:
    if soliton_params is None:
        soliton_params = [
            {'pos_x': -6.0, 'pos_y': 0.0, 'vel_x': 0.5, 'vel_y': 0.0, 'B': 0.5, 'theta': 0.0},
            {'pos_x': 6.0, 'pos_y': 0.0, 'vel_x': -0.5, 'vel_y': 0.0, 'B': 0.5, 'theta': 0.0}
        ]

    psi = torch.ones_like(x, dtype=torch.complex64)

    for params in soliton_params:
        pos_x, pos_y = params['pos_x'], params['pos_y']
        vel_x, vel_y = params['vel_x'], params['vel_y']
        B = params['B']
        theta = params.get('theta', 0.0)
        theta = torch.tensor(theta)

        y_envelope = torch.exp(-0.1 * y**2)

        x_rot = (x - pos_x) * torch.cos(theta) + (y - pos_y) * torch.sin(theta)
        y_rot = -(x - pos_x) * torch.sin(theta) + (y - pos_y) * torch.cos(theta)

        T = x_rot + vel_x * y_rot
        amplitude = torch.sqrt(1 - B * B * sech(T)**2)
        phase = torch.arcsin(B * torch.tanh(T) / torch.sqrt(1 - B * B * sech(T)**2))
        soliton = amplitude * torch.exp(1j * phase)

        # y-confinement
        soliton = 1 + (soliton - 1) * y_envelope[:]
        psi *= soliton

    return psi

def rogue_wave_seed(x, y): 
   base = torch.exp(-(x**2 + y**2)/8.0)
   perturb = 0.1*torch.cos(x/2)*torch.cos(y/2)
   return base*(1.0 + perturb) + torch.tensor(0.j)

def ring_collision(x, y):
   r1 = torch.sqrt((x+5.0)**2 + y**2)
   r2 = torch.sqrt((x-5.0)**2 + y**2)
   width = 1.0
   ring1 = torch.exp(-(r1-2.0)**2/(2*width**2))*(torch.cos(x) - 1j* torch.sin(x))
   ring2 = torch.exp(-(r2-2.0)**2/(2*width**2))*(torch.cos(-x) - 1j* torch.sin(-x))
   return (ring1 + ring2)/2

def initial_slit(x, y, k0=5.0): 
    mask = (x < -5.0).float()
    u0 = mask * torch.exp(-1j * k0 * x)  
    return u0 / torch.sqrt(torch.sum(torch.abs(u0)**2))


nx = 128
ny = 128
Lx = Ly = 20

xn = torch.linspace(-Lx/2, Lx/2, nx)
yn = torch.linspace(-Ly/2, Ly/2, ny)
x, y = torch.meshgrid(xn, yn, indexing='ij')





k0 = .5
theta = torch.tensor(torch.pi/4)
kx = k0*torch.cos(theta)
ky = k0*torch.sin(theta)

sigma = .3
A = 1.0
#u0 = initial_slit(x, y)
#u0 = ring_collision(x, y)
u0 = gaussian_center(x, y, A, sigma, kx, ky, Lx / nx, Ly / ny)
#u0 = dnoidal_wave(x, y, kx, ky)
#u0 = two_soliton_collision(x, y, t=2., v1=3., v2=-3.)
#u0 = create_2d_dark_solitons_diff(x, y)



dt = 1e-3
T = 1.
nt = 1500
num_snapshots = 100 
snap_f = nt // num_snapshots

print((nx/Lx) / dt)

solver = NLSEStepper(nx, ny, Lx, Ly, dt, int(nt), u0, snapshot_freq=snap_f)

solver.evolve()

es = [solver.energy(solver.un[i]) for i in range(1, num_snapshots)]
plt.plot(es)
plt.show()
es = [np.log(np.abs(solver.energy(solver.un[0]) - solver.energy(solver.un[i]))) for i in range(1, num_snapshots)]
plt.plot(es)
plt.show()

#print(solver.un[0], torch.norm(solver.un[0]))
#print(solver.un[1], torch.norm(solver.un[1]))
animate(x, y, solver.un, dt, num_snapshots, nt, "NLSE")

