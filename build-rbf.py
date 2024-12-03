import rbf
from rbf.pde.nodes import poisson_disc_nodes
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows, expand_cols

from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

import numpy as np
import matplotlib.pyplot as plt

def u0(x, y):
    return 4 * np.arctan(np.exp(x + y))

def v0(x, y):
    return -4 * np.exp(x + y) / (1. + np.exp(2 * x + 2 * y))

Lx = Ly = 7
vert = np.array([[-Lx, -Ly], [Lx, -Ly], [Lx, Ly], [-Lx, Ly]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

# radial spacing between nodes
spacing = 5e-1

dt = 1e-1

# shape param
eps = 0.3/spacing

# don't need normals for now (3rd return value)
nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
N = nodes.shape[0]


#plt.scatter(vert[:, 0], vert[:, 1])
#plt.scatter(nodes[groups['interior']][:, 0],
#            nodes[groups['interior']][:, 1], color="red")
#plt.scatter(nodes[groups['boundary:all']][:, 0],
#            nodes[groups['boundary:all']][:, 1], color="black")
#plt.show()



n_points = 64
xn, yn = np.linspace(-Lx, Lx, n_points), np.linspace(-Ly, Ly, n_points) 
X, Y = np.meshgrid(xn, yn)
points = np.vstack([X.ravel(), Y.ravel()]).T
m = 35

x_boundary_points = np.nonzero(
                        np.logical_or(X[:, 0] == -Lx, X[:, 0] == Lx)
                    )
y_boundary_points = np.nonzero(
                        np.logical_or(Y[:, 1] == -Ly, Y[:, 1] == Ly)
                    )

groups['all'] = np.array(
        list(groups['interior']) + list(groups['boundary:all']))

groups['boundary:x'] = np.nonzero(
                                np.logical_or(
                                nodes[:, 0] == Lx,
                                nodes[:, 0] == -Lx
                                )
                            )[0]
groups['boundary:y'] = np.nonzero(
                                np.logical_or(
                                nodes[:, 1] == Ly,
                                nodes[:, 1] == -Ly
                                )
                            )[0]



Id = weight_matrix(
        x = points,
        p = nodes[groups['all']],
        diffs = (0, 0), # no derivative, we want the identity
        n = m,
        )

# make it an actual solution vector which we project onto
# our RBF collection in our domain
u_start = u0(nodes[:, 0], nodes[:, 1])

Id = expand_cols(Id, groups['all'], N)


fig, axs = plt.subplots(figsize=(20, 20), nrows=1, ncols=1,
        subplot_kw={"projection":'3d'})
 
#L = weight_matrix(
#        x = points,
#        p = nodes[groups['all']],
#        diffs = [(2, 0), (0, 2)], # Laplacian
#        n = m,
#        )
#
#L = expand_cols(L, groups['all'], N)
u_xy = Id.dot(u_start)

axs.plot_surface(X, Y, u_xy.reshape((n_points, n_points)), cmap='viridis')
plt.show()


A = np.zeros((N, N))
points_x_boundary_mask = np.logical_or(
                                points[:, 0] == Lx,
                                points[:, 0] == -Lx
                                )
                            
points_y_boundary_mask = np.logical_or(
                                points[:, 1] == Ly,
                                points[:, 1] == -Ly
                                )
points_interior_mask = np.logical_or(~points_x_boundary_mask,
                            ~points_y_boundary_mask
                            )
points_interior = np.nonzero(points_interior_mask)[0]
points_x_boundary = np.nonzero(points_x_boundary_mask)[0]
points_y_boundary = np.nonzero(points_y_boundary_mask)[0]

#A1 = weight_matrix( 
#        x = points[points_interior],
#        p = nodes[groups['interior']],
#        diffs = [(2, 0), (0, 2)], # Laplacian
#        n = m,
#        )
#Ax = weight_matrix(
#        x = points[points_x_boundary],
#        p = nodes[groups['boundary:x']],
#        diffs = (1, 0),
#        n = m + 2,
#        )
#
#Ay = weight_matrix(
#        x = points[points_y_boundary],
#        p = nodes[groups['boundary:y']],
#        diffs = (0, 1),
#        n = m + 2,
#        )


A1 = weight_matrix(
        x = nodes[groups['interior']],
        p = nodes,
        diffs = [(2, 0), (0, 2)], # Laplacian
        n = m,
        ) 

Ax = weight_matrix(
        x = nodes[groups['boundary:x']],
        # x-boundary
        p = nodes,
        diffs = (1, 0), # d_x
        n = 2,
        )

Ay = weight_matrix(
        x = nodes[groups['boundary:y']],
        # y-boundary
        p = nodes,
        diffs = (0, 1), # d_y
        n = 2,
        ) 

#A1    = expand_cols(A1, groups['interior'],   N)
#Ax    = expand_cols(Ax, groups['boundary:x'], N)
#Ay    = expand_cols(Ay, groups['boundary:y'], N)


A1    = expand_rows(A1, groups['interior'],   N)
Ax    = expand_rows(Ax, groups['boundary:x'], N)
Ay    = expand_rows(Ay, groups['boundary:y'], N)

print(A1.shape, Ax.shape, Ay.shape)

#print(A1.nnz, Ax.nnz, Ay.nnz)

#for i in range(2):
#    if i == 0: s = "rows"
#    else: s = "cols"
#    zero_rows_A1 = np.where(A1.getnnz(axis=i) == 0)[0]
#    print(f"A1 Zero {s}:", zero_rows_A1)
#
#    zero_rows_Ax = np.where(Ax.getnnz(axis=i) == 0)[0]
#    print(f"Ax Zero {s}:", zero_rows_Ax)
#
#    zero_rows_Ay = np.where(Ay.getnnz(axis=i) == 0)[0]
#    print(f"Ay Zero {s}:", zero_rows_Ay)


A = A1 + Ax + Ay

u_sol = spsolve(A, u_start)

#fig, axs = plt.subplots(figsize=(20, 20), nrows=1, ncols=1,)
#axs.scatter(nodes[:, 0], nodes[:, 1], c=u_sol, cmap='viridis')
#plt.show()





#Ax = expand_cols(Ax, groups['boundary:x'], N)
#Ay = expand_cols(Ay, groups['boundary:y'], N)
#
#A = A1 + Ax + Ay

# expand and add matrices to have the full work operator


#B = -dt ** 2 * A + Id(interior) + A_boundary
#C = .5 * dt ** 2 * A + 2 * Id(interior) # boundary operator 0





# TODO:
# - inner operator of A is Laplacian
# - boundary operator of A is depending on node:
#   If on x-boundary: diffs = (1, 0)
#   If on y-boundary: diffs = (0, 1)

# Then: Fill matrices: A, B, C, D

# Then we should be able to get to u_1

# then we should be able to get to u_n

# and run error analysis (should be feasible within a few hours)


# Then: figure out why the ADI-HOC method is still off
# (piece of paper)


# then write message to Sid with some summary of progress on these fronts
# saying that the C++ part is not yet touched but it's important to
# see how people usually solve this problem!

