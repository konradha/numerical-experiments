import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import splu, LinearOperator, eigs
from scipy.integrate import solve_ivp
from rbf.sputils import expand_rows, expand_cols
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes


def analytical(x, y, t):
    return 4 * np.arctan(np.exp(x + y - t))

L = 5.
vert = np.array([[-L, -L], [L, -L], [L, L], [-L, L]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

Nt = 400
T = 10.
times = np.linspace(0.0, T, Nt)
spacing = 5.e-1
stencil_size = 35
order = 4
# the RBF choice might have to be changed
phi = 'phs5'

nodes, groups, normals = poisson_disc_nodes(
    spacing,
    (vert, smp),
    boundary_groups={'all': range(len(smp))},
    boundary_groups_with_ghosts=['all'])
n = nodes.shape[0]

groups['interior+boundary:all'] = np.hstack((groups['interior'], groups['boundary:all']))

def f(x, y):
    ## "static breather-like"
    #omega = .95
    #return 4 * np.arctan(np.sin(omega * x) / np.cosh(omega * y))

    return 4 * np.arctan(np.exp(x + y))

    ### "circular" elliptic Jacobi function -- easily yields instabilities!
    #from scipy.special import ellipj
    #m = .5
    #u = x + y
    #sn, cn, dn, ph = ellipj(u, m)
    #return np.array(sn)

    ## ring soliton
    #R = 1.001
    ## stability assertion
    #assert R > 1 and R ** 2 < 2 * (2 * L) ** 2
    #return 4 * np.arctan((x ** 2 + y ** 2 - R ** 2) / (2 * R))




def g(x, y):
    return np.zeros_like(x)

u_init = np.zeros((n,))
u_init[groups['interior+boundary:all']] = f(nodes[groups['interior+boundary:all'], 0],
                                            nodes[groups['interior+boundary:all'], 1])
v_init = np.zeros((n,))
v_init[groups['interior+boundary:all']] = g(nodes[groups['interior+boundary:all'], 0],
                                            nodes[groups['interior+boundary:all'], 1])
z_init = np.hstack((u_init, v_init))

# interpolation matrix
B_disp = weight_matrix(
    x=nodes[groups['interior+boundary:all']],
    p=nodes,
    n=1,
    diffs=(0, 0))

# BC matrix (hopefully)
B_neumann = weight_matrix(
    x=nodes[groups['boundary:all']],
    p=nodes,
    n=stencil_size,
    diffs=[(1, 0), (0, 1)],
    coeffs=[normals[groups['boundary:all'], 0],
            normals[groups['boundary:all'], 1]],
    phi=phi,
    order=order)

B = expand_rows(B_disp, groups['interior+boundary:all'], n)
B += expand_rows(B_neumann, groups['ghosts:all'], n)
B = B.tocsc()
Bsolver = splu(B)

D = weight_matrix(
    x=nodes[groups['interior+boundary:all']],
    p=nodes,
    n=stencil_size,
    diffs=[(2, 0), (0, 2)],
    phi=phi,
    order=order)
D = expand_rows(D, groups['interior+boundary:all'], n)
D = D.tocsc()

global_t = 0

def state_derivative(t, z):
    u, v = z.reshape((2, -1)) 
    global_t = t 
    us = Bsolver.solve(u)
    return np.hstack([v, D.dot(us) - np.sin(us)])
 

print('Performing time integration...')
soln = solve_ivp(
    fun=state_derivative,
    t_span=[times[0], times[-1]],
    y0=z_init,
    method='RK45',
    t_eval=times)
print('Done')

def calculate_energy(u, v, nx, ny, dx, dy):
    f = 2
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (f * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (f * dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux ** 2
    uy2 = uy ** 2
    ut2 = ut ** 2
    cos = 2 * (1 - np.cos(u[1:-1, 1:-1]))
    integrand = np.sum(ux2 + uy2 + ut2 + cos)
    # simple trapeziodal rule
    return 0.5 * integrand * dx * dy


N = 36
xgrid, ygrid = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
xy = np.array([xgrid.flatten(), ygrid.flatten()]).T

I = weight_matrix(
    x=xy, 
    p=nodes[groups['interior+boundary:all']], 
    n=stencil_size, 
    diffs=(0, 0), 
    phi=phi,
    order=order)
I = expand_cols(I, groups['interior+boundary:all'], n)

tn, un = np.array(soln['t']), np.array(soln['y'])

data = un

es = []
nx = ny = N
dx = dy = 2 * L / N
u_solutions = []
v_solutions = []
for i, t in enumerate(tn):
    u, v = data[:, i].reshape((2, -1)) 
    u = I.dot(u).reshape((nx, ny))
    v = I.dot(v).reshape((nx, ny))
    u_solutions.append(u)
    v_solutions.append(v)

    #fig, ax = plt.subplots(figsize=(20, 20), ncols=2, subplot_kw={"projection":'3d'}) 
    #ax[0].plot_surface(xgrid, ygrid, u, cmap='viridis')
    #ax[1].plot_surface(xgrid, ygrid, v, cmap='viridis')
    #plt.show()
     
    es.append(
        calculate_energy(u, v, nx, ny, dx, dy)
        )
plt.plot(es)
plt.show()


fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

u, v = un[:, 0].reshape((2, -1))
u_xy = I.dot(u).reshape((N, N))
surf = ax.plot_surface(xgrid, ygrid, u_xy, cmap='viridis',)


def update(frame):
    u, v = un[:, frame].reshape((2, -1))
    u_xy = I.dot(u).reshape((N, N))
    ax.clear()
    ax.plot_surface(xgrid, ygrid, u_xy, cmap='viridis')

fps = 30
ani = FuncAnimation(fig, update, frames=Nt, interval= Nt / fps,)
plt.show()


u_solutions = np.array(u_solutions)
v_solutions = np.array(v_solutions)
with open('rbf-testdata.npy', 'wb') as f:
    np.save(f, u_solutions)
    np.save(f, v_solutions)
