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

def animate(X, Y, data, dt, num_snapshots, nt, title):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear() 

        
        ax.plot_surface(X, Y,
                #data[frame].cpu().numpy().real ** 2 + data[frame].cpu().numpy().imag ** 2,
                data[frame].cpu().numpy().real,
                cmap='viridis')
        ax.set_title(f"{title}, t={(frame * dt * (nt / num_snapshots)):.2f}")

    fps = 300
    ani = FuncAnimation(fig, update, frames=num_snapshots, interval=num_snapshots / fps, )
    plt.show()

def u_xx_yy(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1, 1:-1] = (
        (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2) +
        (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
    )
    return uxx_yy

def gaussian_center(x, y, A, sigma, kx, ky, dx, dy):
    un =  A * torch.exp(-(x ** 2 +  y ** 2) / 4 / sigma ** 2) * torch.exp(1j * (kx * x + ky * y))  
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

class NLSEStepper:
    def __init__(self, nx, ny, Lx, Ly, dt, nt, initial_u, snapshot_freq=10, dtype=torch.float64):
        self.dtype = dtype
        self.buf = torch.zeros((nx, ny), dtype=self.dtype)
        self.un_real = torch.zeros((nt // snapshot_freq, nx, ny)) 
        self.un_imag = torch.zeros((nt // snapshot_freq, nx, ny))
        self.un = self.un_real + 1j * self.un_imag

        self.u0 = initial_u.clone()

        self.L = torch_sparse_to_scipy_sparse(build_D2(nx-2, ny-2, Lx/nx, Ly/ny, self.dtype))

        self.sf = snapshot_freq

        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nt = nt

        self.dx = torch.tensor(Lx / nx)
        self.dy = torch.tensor(Ly / ny)
        self.dt = torch.tensor(dt)

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
    def step_full(self, u):
        # fully complex evolution
        rho_half = (u.real ** 2 + u.imag ** 2) * u
        us = torch.exp(-.5j * self.dt * rho_half) * u
        iA = -1j * self.L.astype(np.complex128)
        uss = us.reshape((self.nx * self.ny)).cpu().numpy()
        usss = scipy.sparse.linalg.expm_multiply(iA.multiply(self.dt),  uss)
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
        
    def step_split(self, u):
        def sincos_multiply(A, v, t=1.0): 
            exp_plus  = scipy.sparse.linalg.expm_multiply(complex( 1j * t) * A, v)
            exp_minus = scipy.sparse.linalg.expm_multiply(complex(-1j * t) * A,  v)
            sinAv = ((exp_plus - exp_minus)/(2j)).real
            cosAv = ((exp_plus + exp_minus)/2).real
            return sinAv, cosAv

        # split evolution
        v = u.real
        w = u.imag
        rho1 = v ** 2 + w ** 2

        theta = -.5 * self.dt * rho1
        vs = w * torch.cos(theta) - v * torch.sin(theta)
        ws = v * torch.cos(theta) + w * torch.sin(theta)

        #vs = torch.exp(-.5 * self.dt * rho1 * w) * w
        #ws = torch.exp(-.5 * self.dt * rho1 * v) * v 

        vs = vs.reshape((self.nx * self.ny)).cpu().numpy()
        ws = ws.reshape((self.nx * self.ny)).cpu().numpy()
        sin1, cos1 = sincos_multiply(complex(-1j) * self.L, vs, t=self.dt) 
        sin2, cos2 = sincos_multiply(complex(-1j) * self.L, ws, t=self.dt) 
        vss = cos1 - sin2
        wss = sin1 + cos2
        vss = torch.from_numpy(vss).reshape((self.nx, self.ny))
        wss = torch.from_numpy(wss).reshape((self.nx, self.ny))

        rho2 = vss ** 2 + wss ** 2
        theta = -.5 * self.dt * rho1
        vsss = wss * torch.cos(theta) - vss * torch.sin(theta)
        wsss = vss * torch.cos(theta) + wss * torch.sin(theta)

        #vsss = torch.exp(-.5 * self.dt * rho2 * w) * wss 
        #wsss = torch.exp(-.5 * self.dt * rho2 * v) * vss
        un = vsss + 1j * wsss    
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
            u = self.step_full(u) 
            #plt.imshow(torch.sqrt(u.real ** 2 + u.imag ** 2))
            #plt.show()
            self.neumann_bc(u)
            #self.radiation_bc(u)
            if i % self.sf == 0:
                self.un[i // self.sf] = u.clone()

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


nx = ny = 64
Lx = Ly = 20

xn = torch.linspace(-Lx/2, Lx/2, nx)
yn = torch.linspace(-Ly/2, Ly/2, ny)
x, y = torch.meshgrid(xn, yn, indexing='ij')

k0 = .5
theta = torch.tensor(torch.pi/4)
kx = k0*torch.cos(theta)
ky = k0*torch.sin(theta)

sigma = 1.
A = 1.0
u0 = gaussian_center(x, y, A, sigma, kx, ky, Lx / nx, Ly / ny)
#u0 = dnoidal_wave(x, y, kx, ky)
#u0 = two_soliton_collision(x, y, t=2., v1=3., v2=-3.)
#u0 = create_2d_dark_solitons_diff(x, y)



dt = 1e-2
T = 1.
nt = 200
num_snapshots = 100 
snap_f = nt // num_snapshots

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

