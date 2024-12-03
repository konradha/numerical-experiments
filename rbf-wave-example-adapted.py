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

L = 4.
vert = np.array([[-L, -L], [L, -L], [L, L], [-L, L]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

times = np.linspace(0.0, 4., 400)
spacing = 5.e-1
stencil_size = 35
order = 4
phi = 'phs5'

nodes, groups, normals = poisson_disc_nodes(
    spacing,
    (vert, smp),
    boundary_groups={'all': range(len(smp))},
    boundary_groups_with_ghosts=['all'])
n = nodes.shape[0]

def f(x, y):
    return 4 * np.arctan(np.exp(x + y))

def g(x, y):
    return -4 * np.exp(x + y) / (1 + np.exp(2*x + 2*y))

groups['interior+boundary:all'] = np.hstack((groups['interior'], groups['boundary:all']))

u_init = np.zeros((n,))
u_init[groups['interior+boundary:all']] = f(nodes[groups['interior+boundary:all'], 0],
                                            nodes[groups['interior+boundary:all'], 1])
v_init = np.zeros((n,))
v_init[groups['interior+boundary:all']] = g(nodes[groups['interior+boundary:all'], 0],
                                            nodes[groups['interior+boundary:all'], 1])
z_init = np.hstack((u_init, v_init))

def bc_left(y, t):
    return -4 * np.exp(-L + y) / (np.exp(2*t) + np.exp(-2*L + 2*y))

def bc_right(y, t):
    return -4 * np.exp(L + y) / (np.exp(2*t) + np.exp(2*L + 2*y))

def bc_bottom(x, t):
    return -4 * np.exp(x - L) / (np.exp(2*t) + np.exp(2*x - 2*L))

def bc_top(x, t):
    return -4 * np.exp(x + L) / (np.exp(2*t) + np.exp(2*x + 2*L))

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

left_boundary_mask   = np.nonzero(nodes[groups['boundary:all']][:, 0] == -L)
right_boundary_mask  = np.nonzero(nodes[groups['boundary:all']][:, 0] == L)
top_boundary_mask    = np.nonzero(nodes[groups['boundary:all']][:, 1] == L)
bottom_boundary_mask = np.nonzero(nodes[groups['boundary:all']][:, 1] == -L)

def state_derivative(t, z):
    print("calculating derivative at", t)
    u, v = z.reshape((2, -1))
        
    global_t = t 

    us = Bsolver.solve(u)
    return np.hstack([v, D.dot(us) - np.sin(us)])

 
    #u_boundary = np.where(
    #    normals[groups['boundary:all'], 0] != 0,
    #    np.where(
    #        nodes[groups['boundary:all'], 0] < 0, 
    #        bc_left(nodes[groups['boundary:all'], 1], t),
    #        bc_right(nodes[groups['boundary:all'], 1], t)
    #    ),
    #    np.where(
    #        nodes[groups['boundary:all'], 1] < 0,
    #        bc_bottom(nodes[groups['boundary:all'], 0], t),
    #        bc_top(nodes[groups['boundary:all'], 0], t)
    #    )
    #)
    #u[groups['boundary:all']] = u_boundary

    #u_solved = Bsolver.solve(u)

    ##u_solved[left_boundary_mask] = analytical(
    ##        -L * np.ones_like(nodes[left_boundary_mask][:, 1]), nodes[left_boundary_mask][:, 1], t)
    ##u_solved[right_boundary_mask] = analytical(
    ##        L * np.ones_like(nodes[right_boundary_mask][:, 1]), nodes[right_boundary_mask][:, 1], t)

    ##u_solved[top_boundary_mask] = analytical(
    ##        nodes[top_boundary_mask][:, 0], L * np.ones_like(nodes[top_boundary_mask][:, 0]), t)
    ##u_solved[bottom_boundary_mask] = analytical(
    ##        nodes[bottom_boundary_mask][:, 0], -L * np.ones_like(nodes[bottom_boundary_mask][:, 0]), t)
    # 
    #return np.hstack([v, D.dot(u_solved) - np.sin(u_solved)])


print('Performing time integration...')
soln = solve_ivp(
    fun=state_derivative,
    t_span=[times[0], times[-1]],
    y0=z_init,
    method='RK45',
    t_eval=times)
print('Done')

xgrid, ygrid = np.meshgrid(np.linspace(-L, L, 100), np.linspace(-L, L, 100))
xy = np.array([xgrid.flatten(), ygrid.flatten()]).T

I = weight_matrix(
    x=xy, 
    p=nodes[groups['interior+boundary:all']], 
    n=stencil_size, 
    diffs=(0, 0), 
    phi=phi,
    order=order)
I = expand_cols(I, groups['interior+boundary:all'], n)


fig = plt.figure(figsize=(10, 8),)
ax = fig.add_subplot(111, projection='3d')
p = ax.plot_surface(xgrid, ygrid, f(xgrid, ygrid), cmap='viridis')



def update(index):
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')# fig.add_subplot(111, projection= '3d')

    u, v = soln.y[:, index].reshape((2, -1))
    u_xy = I.dot(u).reshape((100, 100))

    u_analytical = analytical(xgrid, ygrid, global_t) 

    for s in smp:
        ax.plot(vert[s, 0], vert[s, 1], 'k-')

    p = ax.plot_surface(xgrid, ygrid,
            #u_xy,
            np.abs(u_xy - u_analytical),
            cmap='viridis')

    ax.set_title(f'sine-Gordon, t={times[index]:.2f}')
    ax.set_xlim(-L-0.5, L+0.5)     
    ax.set_ylim(-L-0.5, L+0.5)     
    ax.grid(ls=':', color='k')   
    ax.set_aspect('equal')
    ax.set_zlim(-0.05, 7.1 + 0.05)
    fig.colorbar(p)
    fig.tight_layout()

    return

ani = FuncAnimation(
    fig=fig, 
    func=update, 
    frames=range(0, len(times), 1), 
    repeat=True,
    blit=False)
    
ani.save('sine_gordon_simulation.gif', writer='pillow', fps=30)
plt.show()
