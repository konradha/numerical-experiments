import numpy as np
import matplotlib.pyplot as plt

def neumann_bc(u):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

def lapl_eigenvalues(nx, ny, dx, dy, tau=None):
    # findable from REU paper using google search
    ev = np.zeros((nx, ny))
    Nx = int(1 / dx)
    Ny = int(1 / dy)
    Lx = nx * dx
    Ly = ny * dy
    norm = dx * dy
    if tau is None: tau = 1
    for i in range(-nx//2, nx//2):
        for j in range(-ny//2, ny // 2):
            #ev[i, j] = -(4 /nx/ny) * (
            #        np.sin((np.pi * i / nx / 2)) ** 2 + np.sin((np.pi * j / ny / 2)) ** 2
            #        )
            ev[i, j] = -4 * np.pi ** 2 / (nx * ny * dx * dy) * (i ** 2 + j ** 2)
            #ev[i, j] = -4 * (i ** 2 + j ** 2)
    return ev

def u_xx_yy_9(buf, a, dx, dy):
    uxx_yy = buf
    uxx_yy[1:-1,1:-1] = (1/(6*dx*dy)) * (
        4*(a[1:-1,2:] + a[1:-1,:-2] + a[2:,1:-1] + a[:-2,1:-1]) +
        a[2:,2:] + a[:-2,:-2] + a[2:,:-2] + a[:-2,2:] -
        20*a[1:-1,1:-1])
    return uxx_yy

def fourier_laplacian(u, Lx, Ly):
    nx, ny = u.shape
    u_ext = np.zeros((2*ny, 2*nx))
    u_ext[:ny, :nx] = u
    u_ext[:ny, nx:] = np.fliplr(u)
    u_ext[ny:, :nx] = np.flipud(u)
    u_ext[ny:, nx:] = np.flipud(np.fliplr(u))

    dx = 2*Lx/nx
    dy = 2*Ly/ny

    kx = np.fft.fftfreq(2*nx, ) * nx * np.pi/Lx 
    ky = np.fft.fftfreq(2*ny, ) * ny * np.pi/Ly

    KX, KY = np.meshgrid(kx, ky)

    u_hat = np.fft.fft2(u_ext)

    lap_u_hat = -(KX**2 + KY**2) * u_hat
    lap_u_ext = np.real(np.fft.ifft2(lap_u_hat))
    lap_u = lap_u_ext[:ny, :nx]
    return lap_u


def f1(x, y):
    return np.exp(-(X**2 + Y **2) / 2)

def f2(x, y):
    return 4 * np.arctan(np.exp((x + y)))

if __name__ == '__main__':
    nx = ny = 25
    Lx = Ly = 4

    dx = 2 * Lx / (nx - 1)
    dy = 2 * Ly / (ny - 1)

    lapl_ev = lapl_eigenvalues(nx, ny, dx, dy)

    xn, yn = np.linspace(-Lx, Lx, nx), np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(xn, yn)

    for f in [f1, f2]:
        u = f(X, Y)

        fig, axs = plt.subplots(figsize=(20, 20), ncols=4, subplot_kw={"projection":'3d'})
        face1 = u_xx_yy_9(np.zeros_like(u), u, dx, dy)
        face2 = np.abs(face1 - fourier_laplacian(u, Lx, Ly))
        face3 = np.abs(u_xx_yy_9(np.zeros_like(u), u, dx, dy) - np.fft.ifft2(lapl_ev * np.fft.fft2(u)))
        face4 = fourier_laplacian(u, Lx, Ly)

        for f in [face1, face2, face3, face4]:
            f = neumann_bc(f)

        #z_min = np.min([np.min(f) for f in (face1, face2, face3, face4)])
        #z_max = np.max([np.max(f) for f in (face1, face2, face3, face4)])
        
        axs[0].plot_surface(X, Y, face1, cmap='viridis')
        axs[1].plot_surface(X, Y, face2, cmap='viridis')
        axs[2].plot_surface(X, Y, face3, cmap='viridis')
        axs[3].plot_surface(X, Y, face4, cmap='viridis')
        axs[0].set_title("Higher order stencil Laplacian")
        axs[1].set_title("Difference (mirror extension)")
        axs[2].set_title("Difference (hand-rolled) (abs)")
        axs[3].set_title("Mirror extension")
        #for i in range(2):
        #    axs[i].set_zlim(z_min, z_max)

        plt.show()

    # TODO: mirror extension to have better values!



