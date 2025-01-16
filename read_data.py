import numpy as np
import matplotlib.pyplot as plt

from sys import argv

fname1 = str(argv[1])
fname2 = str(argv[2])

data1 = np.load(fname1)
data2 = np.load(fname2)


L = 3
nx, ny = data1.shape
assert nx, ny == data2.shape

xn = np.linspace(-L, L, nx)
yn = np.linspace(-L, L, ny)
X, Y = np.meshgrid(xn, yn)
fig, axs = plt.subplots(figsize=(20, 20), ncols=2, subplot_kw={"projection":'3d'})
axs[0].plot_surface(X, Y, data1, cmap='viridis')
axs[1].plot_surface(X, Y, data2, cmap='viridis')
plt.show()
