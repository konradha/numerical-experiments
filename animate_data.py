import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import argv

def animate(X, Y, data, nt,):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                data[frame],
                cmap='viridis')
    fps = 300
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    plt.show()


fname = str(argv[1])
L = float(argv[2])
nx = int(argv[3])
ny = int(argv[4])

data = np.load(fname)
nt = data.shape[0]
assert data.shape[1] == nx and data.shape[2] == ny

xn = np.linspace(-L, L, nx)
yn = np.linspace(-L, L, ny)
X, Y = np.meshgrid(xn, yn)

animate(X, Y, data, nt)
