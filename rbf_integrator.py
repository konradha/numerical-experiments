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


class RBFIntegrator:
    def __init__(self, Lx_min, Lx_max, Ly_min, Ly_max, nt, T,
            spacing=5e-1, stencil_size=35, order=1, phi='phs4', method='RK45'): 

        self.Lx_min, self.Lx_max = Lx_min, Lx_max
        self.Ly_min, self.Ly_max = Ly_min, Ly_max
         
        vert = np.array([
            [Lx_min, Ly_min], [Lx_max, Ly_min],
            [Lx_max, Ly_max], [Lx_min, Ly_max]])
        smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        self.nt = nt
        self.T = T
        self.tn = np.linspace(0., T, nt)

        # m, num nearest neighbors to interpolate
        self.stencil_size = stencil_size
        # spacing between nodes
        self.spacing = spacing
        self.order = order
        self.phi = phi # might change to 'gtps'

        nodes, groups, normals = poisson_disc_nodes(
            spacing,
            (vert, smp),
            boundary_groups={'all': range(len(smp))},
            boundary_groups_with_ghosts=['all'])
        n = nodes.shape[0]    
        groups['interior+boundary:all'] = np.hstack(
                (groups['interior'], groups['boundary:all']))

        self.groups = groups
        self.nodes = nodes 
        self.normals = normals
        self.n = n
        
        # interpolation matrix
        B_disp = weight_matrix(
            x=nodes[groups['interior+boundary:all']],
            p=nodes,
            n=1,
            diffs=(0, 0))
        self.B_disp = B_disp

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
        self.B_neumann = B_neumann

        B = expand_rows(B_disp, groups['interior+boundary:all'], n)
        B += expand_rows(B_neumann, groups['ghosts:all'], n)
        self.B = B.tocsc()
        self.Bsolver = splu(B)

        D = weight_matrix(
            x=nodes[groups['interior+boundary:all']],
            p=nodes,
            n=stencil_size,
            diffs=[(2, 0), (0, 2)],
            phi=phi,
            order=order)
        D = expand_rows(D, groups['interior+boundary:all'], n)
        self.D = D.tocsc()

        self.global_t = 0.

        self.method = method
    # inital function
    def f(self, x, y):
        ## "static breather-like"
        #omega = .1
        #return 4 * np.arctan(np.sin(omega * x) / np.cosh(omega * y))

        #return 4 * np.arctan(np.exp(x + y))

        ### "circular" elliptic Jacobi function -- easily yields instabilities!
        #from scipy.special import ellipj
        #m = .5
        #u = x + y
        #sn, cn, dn, ph = ellipj(u, m)
        #return np.array(sn)

        ## ring soliton
        ##R = 1.001
        #R =  .5
        ## stability assertion
        ##assert R > 1 and R ** 2 < 2 * (2 * L) ** 2
        #return 4 * np.arctan((x ** 2 + y ** 2 - R ** 2) / (2 * R))

        R = .5
        return 4 * np.arctan(((x - 5.) ** 2 + (y - 5.) ** 2 - R ** 2) / (2 * R))


    # initial velocity
    def g(self, x, y):
        return np.zeros_like(x)


    def evolve(self):
        n = self.n
        u_init = np.zeros((n,))
        u_init[self.groups['interior+boundary:all']] = self.f(self.nodes[
                                                            self.groups['interior+boundary:all'], 0],
                                                        self.nodes[
                                                            self.groups['interior+boundary:all'], 1])
        v_init = np.zeros((n,))
        v_init[self.groups['interior+boundary:all']] = self.g(self.nodes[self.groups['interior+boundary:all'], 0],
                                                            self.nodes[self.groups['interior+boundary:all'], 1])
        z_init = np.hstack((u_init, v_init))

        def state_derivative(t, z):
            u, v = z.reshape((2, -1)) 
            self.global_t = t # bookkeeping
            
            us = self.Bsolver.solve(u)
            return np.hstack([v, self.D.dot(us) - np.sin(us)])
         
        
        self.solutions = solve_ivp(
            fun=state_derivative,
            t_span=[self.tn[0], self.tn[-1]],
            y0=z_init,
            method=self.method,
            #method='Radau',
            #method='BDF',
            t_eval=self.tn)

    def get_solution(self, N_interpolation_points=36):
        N = N_interpolation_points
        xgrid, ygrid = np.meshgrid(np.linspace(self.Lx_min, self.Lx_max, N), np.linspace(self.Ly_min, self.Ly_max, N))

        xy = np.array([xgrid.flatten(), ygrid.flatten()]).T

        I = weight_matrix(
            x=xy, 
            p=self.nodes[self.groups['interior+boundary:all']], 
            n=self.stencil_size, 
            diffs=(0, 0), 
            phi=self.phi,
            order=self.order)
        I = expand_cols(I, self.groups['interior+boundary:all'], self.n)

        tn, un = np.array(self.solutions['t']), np.array(self.solutions['y']) 

        data = un
        un_ret, vn_ret = [], []
        for i, t in enumerate(tn):
            u, v = data[:, i].reshape((2, -1)) 
            u = I.dot(u).reshape((N, N))
            v = I.dot(v).reshape((N, N))
            un_ret.append(u)
            vn_ret.append(v)
        
        return xgrid, ygrid, np.array(un_ret), np.array(vn_ret)
       

if __name__ == '__main__':
    nt = 500
    integrator = RBFIntegrator(-10, 10, -10, 10, nt, 10.)
    integrator.evolve()
    xgrid, ygrid, un, vn = integrator.get_solution()
 
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.plot_surface(xgrid, ygrid, un[frame], cmap='viridis')

    fps = 30
    ani = FuncAnimation(fig, update, frames=nt, interval= nt / fps,)
    plt.show()

