import util

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from copy import deepcopy
import itertools
from typing import Dict, Optional, Union
import numpy.typing as npt


class RBFMapper:
    def __init__(self, points: np.array,
            y_min: float, y_max: float, x_min: float, x_max: float,
            nx: int, ny: int,
            boundary_condition: str = "neumann-like",
            m_neighbors: int = 5):
        valid_boundary = ["neumann-like"]

        self.points = points
        self.xmin, self.xmax = x_min, x_max
        self.ymin, self.ymax = y_min, y_max
        self.m = m_neighbors

        self.nx, self.ny = nx, ny
        self.n = nx * ny

        if boundary_condition not in valid_boundary: raise NotImplemented 
        self.boundary_condition = boundary_condition
 
        self.boundary_x = self._compute_x_boundary_indices()
        self.boundary_y = self._compute_y_boundary_indices()
        self.interior_idx = self._compute_interior_indices()
        self.corners = self._mark_corners()

        [self.m_neighbors_distances, self.m_nearest] = self._compute_neighbors()


    def _compute_interior_indices(self):
        self._interior_mask = np.logical_and(
                ~self._x_boundary_mask,
                ~self._y_boundary_mask)
        return self.points[self._interior_mask]

    def _compute_x_boundary_indices(self):
        self._x_boundary_mask = np.logical_or(
                np.isclose(self.points[:, 0], xmin, atol=1e-5),
                np.isclose(self.points[:, 0], xmax, atol=1e-5)
                )
        return self.points[self._x_boundary_mask]

    def _compute_y_boundary_indices(self):
        self._y_boundary_mask = np.logical_or(
                np.isclose(self.points[:, 1], ymin, atol=1e-5),
                np.isclose(self.points[:, 1], ymax, atol=1e-5)
                )
        return self.points[self._y_boundary_mask]

    def _mark_corners(self):
        both_x_and_y_boundary = np.logical_and(self._y_boundary_mask,
                        self._x_boundary_mask)
        self._corner_mask = both_x_and_y_boundary 
        # these belong to both boundaries so we'll have to do 
        # an averaging here and be careful about the boundary conditions here
        # self._x_boundary_mask[both_x_and_y_boundary] = False
        # self._y_boundary_mask[both_x_and_y_boundary] = False
        return self.points[both_x_and_y_boundary]

    def _compute_neighbors(self):
        from sklearn.neighbors import KDTree
        kd = KDTree(self.points, leaf_size=2 * self.m, metric='euclidean')
        self._kd = kd
        return kd.query(self.points, k = self.m, return_distance=True)

    def plot_domain(self):
        # testing function to see how points are distributed on rectangular domain
        plt.scatter(self.points[self._x_boundary_mask][:, 0], self.points[self._x_boundary_mask][:, 1])
        plt.scatter(self.points[self._y_boundary_mask][:, 0], self.points[self._y_boundary_mask][:, 1])
        plt.scatter(self.points[self._interior_mask][:, 0], self.points[self._interior_mask][:, 1])
        plt.scatter(self.points[self._corner_mask][:, 0], self.points[self._corner_mask][:, 1], color="red")
        plt.show()

    def plot_domain_and_m_neighbors(self, n=3):
        # testing function to see if we get the right nearest neighbors on rectangular domain
        plt.scatter(self.points[self._x_boundary_mask][:, 0], self.points[self._x_boundary_mask][:, 1])
        plt.scatter(self.points[self._y_boundary_mask][:, 0], self.points[self._y_boundary_mask][:, 1])
        plt.scatter(self.points[self._corner_mask][:, 0], self.points[self._corner_mask][:, 1], color="grey")

        rand_idx = np.random.randint(0, self.n, n) 
        for ni in rand_idx:
            plt.scatter(self.points[self.m_nearest[ni], 0], self.points[self.m_nearest[ni], 1], color="black")
            plt.scatter(self.points[ni, 0], self.points[ni, 1], color="red")
        plt.show()

    def plot_phi(self, L):
        xn = np.linspace(-L, L, 1000)
        yn = np.linspace(-L, L, 1000)
        X, Y = np.meshgrid(xn, yn)
        fig, axs = plt.subplots(figsize=(20, 20), ncols=3,
                subplot_kw={"projection":'3d'}) 
        for k in range(1, 4):
            axs[k-1].plot_surface(X, Y,
                    self.Phi(X, k),
                    cmap='viridis')
            axs[k-1].set_title(f"kappa = {k}")
        plt.show()

    def plot_m_n_matrix_structure(self):
        idx_set = np.nan * np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.m):
                idx_set[i, self.m_nearest[i]] = 1
        idx_set[idx_set == 0] = np.nan  
        plt.imshow(idx_set)
        plt.show()

    def phi(self, r, kappa: float = 2.):
        return r ** (2 * kappa) * np.log(r + 1e-10)

    def Phi(self, r, kappa: float = 2.):
        # \Delta Phi = phi
        f1 = (1 / (4 * (kappa + 1) ** 2))  
        f2 = - (1 / (4 * (kappa + 1) ** 3))  
        return (f1 * np.log(r + 1e-10) - f2) *  r ** (2 * kappa + 2)

    def Phi_r(self, r, kappa: float = 2.): 
        f1 = (1 / (4 * (kappa + 1) ** 2))  
        f2 = - (1 / (4 * (kappa + 1) ** 3))
        return (2 * kappa + 2) * (
                (1 / (2 * kappa + 2) + np.log(r + 1e-10)) * f1 - f2) * r ** (2 * kappa + 1) 

    def Phi_x(self, x, r, kappa: float = 2.):
        return (x / r) * self.Phi_r(r, kappa)

    def Phi_y(self, y, r, kappa: float = 2.):
        return (y / r) * self.Phi_r(r, kappa)

    #def P_eval_at_basis(self, x, y):
    #    # just go for monomials at first
    #    return np.array([
    #        1, x, y,
    #        #x * y,
    #        #x ** 2, y ** 2
    #        ])
    
    def _build_local_domain_matrix(self, n: int,
            points_in_domain: Optional[Union[int, npt.NDArray]],
            fun_to_evaluate):
        """
            n: index of point to calculate the local matrix for
        """
        if isinstance(points_in_domain, int):
            points_in_domain = self.points[self.m_nearest[n]]

        assert points_in_domain.shape[0] == self.m

        assert 0 <= n < self.n
        local_distances = self.m_neighbors_distances[n] 
        local_points    = self.points[self.m_nearest[n]]

        # hardcoding this -- we want the identity
        diffs = np.asarray([[0, 0], [0, 0]], dtype=int)

        assert self.m == len(local_points)
        m = self.m
        # fast shorthand to build up distance matrix with m neighbors
        points_diffs = local_points[:, np.newaxis, :] - local_points[np.newaxis, :, :] 
        # L2-norm for distances
        points_dists = np.sqrt(np.sum(points_diffs ** 2, axis=-1))

        Phi_mm = self.Phi(points_dists)

        def _max_poly_order(size, dim):
            order = -1
            while util.monomial_count(order + 1, dim) <= size:
                order += 1
            return order - 1

        # 2d space here assumed
        max_order = _max_poly_order(self.m, 2)

        order = diffs.sum(axis=1).max()
        order = min(order, max_order)

        ndim = self.m
        power = util.monomial_powers(order, ndim)
        P_ms = util.mvmonos(local_points, power)
       
        mat = np.block([
                [Phi_mm, P_ms],
                [P_ms.T, np.zeros((P_ms.shape[1], P_ms.shape[1]))]
                ])

        
        #rhs = np.concatenate([points_in_domain, np.zeros(P_ms.shape[1])]) 
        rhs = np.hstack([fun_to_evaluate(points_in_domain), np.zeros(P_ms.shape[1])])
        return np.linalg.solve(mat, rhs)

    def weights(self, base_points, eval_points, diffs,):
        pass
        

    def local_system(self, n: int, local_values: npt.NDArray):
        Phi_mm, P_ms = self._build_local_domain_matrix(n)
        m = self.m
        s = P_ms.shape[1]

        Y_ms = np.block([
                [Phi_mm, P_ms],
                [P_ms.T, np.zeros((s, s))]
            ])
        rhs = np.hstack([local_values, np.zeros((s,))])
        coeffs = np.linalg.solve(Y_ms, rhs)
        return coeffs


    def build_global_map(self, rhs):
        n = self.n
        A = lil_matrix((n, n)) 

        #for i in range(n):
        #    local_i = 
 
        
def const_fun(x):
    assert x.shape[1] == 2 
    const = 10
    return const * np.ones(x.shape[0])

if __name__ == '__main__':
    plotting = False
    L = 7.
    nx = ny = 32
    xmin = ymin= -L
    ymax = xmax = L

    xn, yn = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xn, yn)

    points = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
    
    mapper = RBFMapper(points, ymin, ymax, xmin, xmax, nx, ny) 
    if plotting:
        mapper.plot_domain_and_m_neighbors()
        mapper.plot_phi(L)
        mapper.plot_m_n_matrix_structure()
   
    #print(mapper.local_system(1, np.array([1, 0, 0, 0, 0]))) 

    #print(mapper._build_local_domain_matrix(1, 1, const_fun))
    
