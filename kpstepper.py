import torch
import torch.fft as fft
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.special import ellipj
import numpy as np


class KPStepper:
    def __init__(self, Nx, Ny, Lx, Ly, epsilon=1.0, s=1):
        """
        Kadiomtsev-Petiashvili KPI/KPII

        (u_t + e² u_xxx + 3 (u²)_x)_t = -s u_yy

        (reformulate to integro-diff equation then solve using spectral
        method)
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.epsilon = epsilon
        self.s = s

        self.dx = Lx / Nx
        self.dy = Ly / Ny
        
        self.kx = 2.0 * torch.pi * fft.fftfreq(Nx, Lx/Nx)
        self.ky = 2.0 * torch.pi * fft.fftfreq(Ny, Ly/Ny)
        self.KX, self.KY = torch.meshgrid(self.kx, self.ky, indexing='ij')
        
        # L = -ε²(ikx)³ + s(ky²)/(ikx)
        self.L = torch.zeros_like(self.KX, dtype=torch.complex64)
        nonzero_kx = self.KX != 0
        self.L[nonzero_kx] = -epsilon * 1j * self.KX[nonzero_kx]**3 + \
                            s * self.KY[nonzero_kx]**2 / (1j * self.KX[nonzero_kx])
        
        # kx = 0, ky != 0, the solution should decay
        kx_zero = self.KX == 0
        ky_nonzero = self.KY != 0
        self.L[kx_zero & ky_nonzero] = -1.0  # damping

        self.inv_dx_uy_op = torch.zeros_like(self.KX, dtype=torch.complex64)
        self.inv_dx_uy_op[nonzero_kx] = (1j * self.KY[nonzero_kx]) / (1j * self.KX[nonzero_kx])
 
    def linear_step(self, u, dt):
        u_hat = fft.fft2(u)
        u_hat = u_hat * torch.exp(self.L * dt)
        return fft.ifft2(u_hat).real
    
    def nonlinear_step(self, u, dt):
        u_hat = fft.fft2(u*u)
        ux2 = fft.ifft2(-3 * 1j * self.KX * u_hat).real
        return u + dt * ux2
    
    def step(self, u, dt):
        # Strang
        u = self.linear_step(u, dt/2)
        u = self.nonlinear_step(u, dt) 
        u = self.linear_step(u, dt/2) 
        return u

    def energy(self, u):
        """
        H = ∫∫ [½u² + (ε²/4)(∂ₓu)² - (s/2)(∂ₓ⁻¹u_y)²] dx dy
        """
        u_hat = fft.fft2(u)
        ux_hat = 1j * self.KX * u_hat
        ux = fft.ifft2(ux_hat).real
        inv_dx_uy_hat = self.inv_dx_uy_op * u_hat
        inv_dx_uy = fft.ifft2(inv_dx_uy_hat).real
        H_density = 0.5 * torch.abs(u)**2 + \
                   (self.epsilon**2/4.0) * ux**2 - \
                   (self.s/2.0) * inv_dx_uy**2
        return torch.sum(H_density).real * self.dx * self.dy
class DSStepper:
    def __init__(self, Nx, Ny, Lx, Ly, l=1.0, m=1.0, a=1.0, a1=1.0, b=1.0, c=1.0, d=1.0):
        """
        Davey-Stewardson system

        i u_t + l u_xx + m u_yy = b |u|²u + b1 u v_x
        c v_xx + v_yy = d (|u|²)_x

        where v is a field

        """

        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        
        self.l = l  # u_xx coefficient
        self.m = m  # u_yy coefficient
        self.a = a  # cubic term coefficient
        self.a1 = a1  # coupling term coefficient
        self.b = b  # field equation source term
        self.c = c  # v_xx coefficient
        self.d = d  # v_yy coefficient
        

        self.kx = 2.0 * torch.pi * fft.fftfreq(Nx, Lx/Nx)
        self.ky = 2.0 * torch.pi * fft.fftfreq(Ny, Ly/Ny)
        self.KX, self.KY = torch.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2_x = self.KX**2
        self.K2_y = self.KY**2
        
        self.Lu = -1j * (l * self.K2_x + m * self.K2_y)
        
        
        self.Lv = -(c * self.K2_x + d * self.K2_y)
        self.Lv[0,0] = 1.0 
        self.Lv_inv = 1.0 / self.Lv
        self.Lv_inv[0,0] = 0.0
        
    def solve_field(self, u):
        u_squared = torch.abs(u)**2
        rhs = self.b * 1j * self.KX * fft.fft2(u_squared)
        v_hat = self.Lv_inv * rhs
        return fft.ifft2(v_hat).real
        
    def linear_step(self, u, v, dt):
        u_hat = fft.fft2(u)
        u_hat = u_hat * torch.exp(self.Lu * dt)
        return fft.ifft2(u_hat)
    
    def nonlinear_step(self, u, v, dt):
        cubic = self.a * torch.abs(u)**2 * u
        v_x = fft.ifft2(1j * self.KX * fft.fft2(v)).real
        coupling = self.a1 * u * v_x
        
        return u + dt * (-1j) * (cubic + coupling)
    
    def step(self, u, dt): 
        u = self.linear_step(u, None, dt/2)  
        v = self.solve_field(u)
        u = self.nonlinear_step(u, v, dt)
        u = self.linear_step(u, None, dt/2)
        
        return u, v

    def energy(self, u, v): 
        u_hat = fft.fft2(u)
        u_x_hat = 1j * self.KX * u_hat
        u_y_hat = 1j * self.KY * u_hat
        u_x = fft.ifft2(u_x_hat)
        u_y = fft.ifft2(u_y_hat)
        kinetic = self.l * torch.sum(torch.abs(u_x)**2) + self.m * torch.sum(torch.abs(u_y)**2)
        potential = (self.a/2) * torch.sum(torch.abs(u)**4)
        v_hat = fft.fft2(v)
        v_x_hat = 1j * self.KX * v_hat
        v_y_hat = 1j * self.KY * v_hat
        field_energy = (self.c/2) * torch.sum(torch.abs(fft.ifft2(v_x_hat))**2) + \
                      (self.d/2) * torch.sum(torch.abs(fft.ifft2(v_y_hat))**2)

        coupling = self.a1 * torch.sum(torch.abs(u)**2 * fft.ifft2(v_x_hat).real)
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny
        energy = (kinetic + potential + field_energy + coupling) * dx * dy
        return energy.real

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

def ring_soliton(x, y):
    alpha = 4
    r2 = (x-10) ** 2 + (y-10) ** 2
    return alpha * torch.arctan(3 - torch.exp(3 - torch.sqrt(r2)))

def lump_soliton(X, Y):
    """
    lump soliton, rational
    """
    num = 4 * (1 - X**2 + Y**2)
    den = (1 + X**2 + Y**2)**2
    return num/den

def resonant_soliton(X, Y, k1=1.0, k2=1.5, l1=0.5, l2=-0.5):
    """
    Two-soliton resonant solution without autograd
    k1, k2: x-direction wavenumbers
    l1, l2: y-direction wavenumbers
    """
    eta1 = k1*X + l1*Y
    eta2 = k2*X + l2*Y
    
    # interaction coefficient
    A12 = ((k1 - k2)**2 + (l1 - l2)**2) / ((k1 + k2)**2 + (l1 + l2)**2)
    
    exp_eta1 = torch.exp(eta1)
    exp_eta2 = torch.exp(eta2)
    exp_sum = torch.exp(eta1 + eta2)
    F = 1 + exp_eta1 + exp_eta2 + A12*exp_sum 
    dF_dx = k1*exp_eta1 + k2*exp_eta2 + A12*(k1 + k2)*exp_sum
    d2F_dx2 = k1**2*exp_eta1 + k2**2*exp_eta2 + A12*(k1 + k2)**2*exp_sum
    
    # ∂²ᵪ log(F) = (F ∂²ᵪF - (∂ᵪF)²)/F²
    return 2 * (F * d2F_dx2 - dF_dx**2) / F**2



def sech(x):
    return 1 / torch.cosh(x)

def periodic_line_train(X, Y, kappa=1.0, mu=0.0, x0=0.0, Lx=20.0, n_copies=3):
    result = torch.zeros_like(X)
    for n in range(-n_copies, n_copies + 1):
        result += 12 * kappa**2 * sech(kappa * (X + mu*Y - x0 + n*Lx))**2
    return result

def cnoidal_wave(X, Y, amplitude=1.0, k=1.0, m=0.9):
    cn2 = torch.tensor(ellipj(k * X.numpy(), m)[0] ** 2)
    return amplitude * cn2

def quasiperiodic_wave(X, Y, A1=1.0, A2=0.8, n1=1, m1=1, n2=2, m2=1, Lx=20.0, Ly=20.0):
    k1, k2 = 2*np.pi*n1/Lx, 2*np.pi*n2/Lx
    l1, l2 = 2*np.pi*m1/Ly, 2*np.pi*m2/Ly
    return A1*torch.cos(k1*X + l1*Y) + A2*torch.cos(k2*X + l2*Y)

def weierstrass_p_array(X, Y, g2=1.0, g3=1.0, Lx=20.0, Ly=20.0):
    z = X + 1j*Y
    result = torch.zeros_like(X)

    for n in range(-2, 3):
        for m in range(-2, 3):
            if n == 0 and m == 0:
                continue
            w = 2*np.pi*(n/Lx + 1j*m/Ly)
            result += 1/((z - w)**2) - 1/(w**2)

    return -2 * torch.gradient(torch.gradient(torch.log(result.abs()), X)[0], X)[0]

def all_periodic_solutions(X, Y, Lx=20.0, Ly=20.0):
    u0_line = periodic_line_train(X, Y, Lx=Lx)
    u0_cnoidal = cnoidal_wave(X, Y)
    u0_quasi = quasiperiodic_wave(X, Y, Lx=Lx, Ly=Ly)
    u0_weierstrass = weierstrass_p_array(X, Y, Lx=Lx, Ly=Ly)

    return u0_line, u0_cnoidal, u0_quasi, u0_weierstrass



def sample_grf(Nx, Ny, Lx, Ly, length_scale=2.5, variance=1.):
    kx = 2.0 * torch.pi * fft.fftfreq(Nx, Lx/Nx)
    ky = 2.0 * torch.pi * fft.fftfreq(Ny, Ly/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = (KX ** 2 + KY ** 2)
    spectrum = variance * torch.exp(-0.5 ** 2 * length_scale**2 * K2)
    spectrum = spectrum / torch.sum(spectrum)
    white_noise = torch.randn(Nx, Ny, dtype=torch.complex64) + \
                 1j * torch.randn(Nx, Ny, dtype=torch.complex64)

    k = white_noise[0,0].clone()
    white_noise[0,0] = k.real
    field_k = torch.sqrt(spectrum) * white_noise
    field = fft.ifft2(field_k).real

    dx = Lx / Nx
    dy = Ly / Ny

    norm = torch.sqrt(torch.sum(torch.abs(field)**2) * dx * dy)

    return field / norm

def sample_mixture_kpi_grf(Nx, Ny, Lx, Ly, n_wavelets=5, epsilon=1e-10):
    kx = 2.0 * torch.pi * fft.fftfreq(Nx, Lx/Nx)
    ky = 2.0 * torch.pi * fft.fftfreq(Ny, Ly/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    KX_reg = torch.where(abs(KX) < epsilon, epsilon, KX)
    K2 = abs(KX)**3 + (KY**2)/abs(KX_reg)

    wavelet_centers_x = torch.rand(n_wavelets) * Nx
    wavelet_centers_y = torch.rand(n_wavelets) * Ny
    wavelet_scales = torch.rand(n_wavelets) * 0.5 + 0.5

    field_k = torch.zeros((Nx, Ny), dtype=torch.complex64)
    for i in range(n_wavelets):
        phase = torch.exp(-1j * (KX * wavelet_centers_x[i] + KY * wavelet_centers_y[i]))
        morlet = torch.exp(-(K2 * wavelet_scales[i]**2))
        field_k += torch.randn(2)[0] * morlet * phase

    spectrum = 1.0 / (1.0 + K2)**2
    field_k = field_k * torch.sqrt(spectrum)
    k = field_k[0,0].clone()
    field_k[0,0] = k.real
    field = fft.ifft2(field_k).real
    norm = torch.sqrt(torch.sum(torch.abs(field)**2) * (Lx/Nx) * (Ly/Ny))
    return field / norm

def dromion_ds(x, y, alpha=1.0, beta=1.0): 
    u = torch.exp(-alpha*x**2 - beta*y**2)
    v = torch.tanh(x) + torch.tanh(y)
    return u, v

def initialize_ds_periodic(X, Y, Nx, Ny, Lx, Ly, k1=1.0, k2=1.0):
    u = (torch.cos(k1*X) + 1j*torch.sin(k2*Y)) * torch.exp(-(X**2 + Y**2)/16.0)
    u_squared = torch.abs(u)**2
    u_squared_k = fft.fft2(u_squared)
    kx = 2.0 * torch.pi * fft.fftfreq(Nx, Lx/Nx)
    ky = 2.0 * torch.pi * fft.fftfreq(Ny, Ly/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')

    L = -(KX**2 + KY**2)
    L[0,0] = 1.0
    v_k = (1j * KX * u_squared_k) / L
    v_k[0,0] = 0

    v = fft.ifft2(v_k).real
    return u, v


if __name__ == "__main__":
    Nx, Ny = 64, 64
    Lx, Ly = 20.0, 20.0
    stepper = KPStepper(Nx, Ny, Lx, Ly,)
    
    x = torch.linspace(-Lx/2, Lx/2, Nx)
    y = torch.linspace(-Ly/2, Ly/2, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    #u0 = torch.arctan(torch.exp(-((X)**2 + (Y)**2)))
    #u0 = (torch.exp(-((X)**2 + (Y)**2)))
    #u0 = cnoidal_wave(X, Y, .3, .2) # pretty cool wave generated -- breather-like behavior for DS??

    u0 = sample_mixture_kpi_grf(Nx, Ny, Lx, Ly,)  
    #u0 = (X ** 2 + Y ** 2) 
    #u0 = u0 / torch.sqrt(torch.sum(torch.abs(X ** 2 + Y ** 2)**2) * (Lx/Nx) * (Ly/Ny))
    #u0, v0 = initialize_ds_periodic(X, Y, Nx, Ny, Lx, Ly,)
    #v0 =  torch.zeros_like(u0)
 
    dt = 1e-2
    u = u0
    nt = 1000
    T = dt * nt
    sf = 10
    un = np.zeros((nt//sf, Nx, Ny))
    x_es = []
    es = []
    e0 = stepper.energy(u0)
    for n in tqdm(range(nt)): 
        #if abs(e) >= 2 * abs(e0):
        #    break  
        u = stepper.step(u, dt) 
        e = stepper.energy(u,)
        if n % sf == 0:
            x_es.append(n)
            es.append(e)
            un[n//sf, :, :] = u.real.cpu().numpy()
    
    plt.plot(dt * np.array(x_es), es)
    plt.show()
    animate(X, Y, un, dt, nt//sf, nt, "KP-I")

