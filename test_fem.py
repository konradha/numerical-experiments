import numpy as np
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tqdm import tqdm


class SimpleMesh:
    def __init__(self, Lx, Ly, nx, ny):
        x = np.linspace(-Lx, Lx, nx)
        y = np.linspace(-Ly, Ly, ny)
        
        X, Y = np.meshgrid(x, y)
        self.nodes = np.column_stack([X.ravel(), Y.ravel()])
        
        from scipy.spatial import Delaunay
        
        tri = Delaunay(self.nodes)
        self.elements = tri.simplices
        
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
    
    def plot(self):
        plt.figure(figsize=(10,8))
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.elements)
        plt.plot(self.nodes[:,0], self.nodes[:,1], 'o')
        plt.show()
    
    def local_element_matrices(self, element_index):
        elem_nodes = self.nodes[self.elements[element_index]]
       
        # simple P1 element
        x1, y1 = elem_nodes[0]
        x2, y2 = elem_nodes[1]
        x3, y3 = elem_nodes[2]
        area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        local_mass = np.eye(3) * (area / 12)
        
        return local_mass, area

class SineGordonFEM:
    def __init__(self, mesh, c, m):
        self.nodes = mesh.nodes
        self.elements = mesh.elements
        self.c = c if not callable(c) else c(nodes)
        self.m = m

        self.boundary_nodes = self._find_boundary_nodes()
        self.interior_nodes = np.setdiff1d(np.arange(len(self.nodes)), self.boundary_nodes)
       
        self.M = self.assemble_mass_matrix().tocsc()
        self.K = self.assemble_stiffness_matrix().tocsc() 
        self.M_inv = spalin.splu(self.M)

        self._enforce_neumann_bc()


        self.u_old = None

    def _find_boundary_nodes(self,):
        nx = ny = int(np.sqrt(self.nodes[:, 0].shape[0]))
        boundary = []
        boundary.extend(range(nx))
        boundary.extend(range((ny-1)*nx, ny*nx)) 
        for i in range(1, ny-1):
            boundary.append(i * nx)
            boundary.append(i * nx + nx - 1)
        return np.unique(boundary)

    def apply_periodic_bc(self):
        nx, ny = int(np.sqrt(len(self.nodes))), int(np.sqrt(len(self.nodes)))

        for i in range(ny):
            left_node = i * nx
            right_node = (i + 1) * nx - 1
            self.M[left_node, :] += self.M[right_node, :]
            self.M[:, left_node] += self.M[:, right_node]
            self.M[right_node, :] = 0
            self.M[:, right_node] = 0
            self.M[right_node, left_node] = 1

            self.K[left_node, :] += self.K[right_node, :]
            self.K[:, left_node] += self.K[:, right_node]
            self.K[right_node, :] = 0
            self.K[:, right_node] = 0
            self.K[right_node, left_node] = 1

        for j in range(nx):
            bottom_node = j
            top_node = (ny - 1) * nx + j
            self.M[bottom_node, :] += self.M[top_node, :]
            self.M[:, bottom_node] += self.M[:, top_node]
            self.M[top_node, :] = 0
            self.M[:, top_node] = 0
            self.M[top_node, bottom_node] = 1

            self.K[bottom_node, :] += self.K[top_node, :]
            self.K[:, bottom_node] += self.K[:, top_node]
            self.K[top_node, :] = 0
            self.K[:, top_node] = 0
            self.K[top_node, bottom_node] = 1

        self.M_inv = spalin.splu(self.M.tocsc())


    def _enforce_neumann_bc(self,): 
        # assembly should be consistent with homogeneous Neumann BC
        pass
         
    def assemble_mass_matrix(self):
        def compute_local_mass_matrix(element_nodes):
            x1, y1 = element_nodes[0]
            x2, y2 = element_nodes[1]
            x3, y3 = element_nodes[2]
            area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            local_mass = (area / 12) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])
            return local_mass

        # P1 assembly
        M = sparse.lil_matrix((len(self.nodes), len(self.nodes)))
        for element in self.elements:
            local_nodes = [self.nodes[i] for i in element]
            local_mass = compute_local_mass_matrix(local_nodes)
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    M[global_i, global_j] += local_mass[i,j]
        return M.tocsr()
    
    def assemble_stiffness_matrix(self):
        def compute_local_stiffness_matrix(element_nodes, c_coeff):
            x1, y1 = element_nodes[0]
            x2, y2 = element_nodes[1]
            x3, y3 = element_nodes[2]
            area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

            b1 = (y2 - y3) / (2 * area)
            c1 = (x3 - x2) / (2 * area)
            b2 = (y3 - y1) / (2 * area)
            c2 = (x1 - x3) / (2 * area)
            b3 = (y1 - y2) / (2 * area)
            c3 = (x2 - x1) / (2 * area)

            local_stiffness = c_coeff * area * np.array([
                [b1*b1 + c1*c1, b1*b2 + c1*c2, b1*b3 + c1*c3],
                [b2*b1 + c2*c1, b2*b2 + c2*c2, b2*b3 + c2*c3],
                [b3*b1 + c3*c1, b3*b2 + c3*c2, b3*b3 + c3*c3]
            ])

            return local_stiffness

        K = sparse.lil_matrix((len(self.nodes), len(self.nodes)))
        for element in self.elements:
            local_nodes = [self.nodes[i] for i in element]
            local_stiffness = compute_local_stiffness_matrix(local_nodes, self.c)
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    K[global_i, global_j] += local_stiffness[i,j]
        return K.tocsr()
    
    def nonlinear_force(self, u):
        return self.m * np.sin(u)
    
    def symplectic_step(self, u, v, dt):
        # self.K ~ -\Delta 
        # hence self.K @ u + self.nonlinear_force(u) ~ -\Delta u + m sin(u)
        # ie. it should work out fine with the given signs
        v_half = v - 0.5 * dt * (self.K @ u + self.nonlinear_force(u))
        u_new = u + dt * self.M_inv.solve(v_half) 
        v_new = v_half - 0.5 * dt * (self.K @ u_new + self.nonlinear_force(u_new)) 
        return u_new, v_new
 

def initial_u(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    #return 4 * np.arctan(np.exp(x + y))
    return np.arctan(np.exp(-(x ** 2 + y ** 2) / 2))

def initial_v(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    return np.zeros_like(x)

def plot_surface(mesh, u):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        u,                
        cmap='viridis',   
        edgecolor='none', 
        alpha=0.8         
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

def animate_evolution(mesh, un, M, dt=1e-2):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(
        mesh.nodes[:, 0], 
        mesh.nodes[:, 1], 
        M @ un[0],                
        cmap='viridis',   
        edgecolor='none', 
        alpha=0.8         
    )
   
    def update(frame):
        ax.clear()
        surf = ax.plot_trisurf(
            mesh.nodes[:, 0], 
            mesh.nodes[:, 1], 
            M @ un[frame],                
            cmap='viridis',   
            edgecolor='none', 
            alpha=0.8         
        ) 
        ax.set_title(f"t={(dt * frame):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=un.shape[0], interval= un.shape[0] / fps, )
    plt.show()

def animate_evolution_fd(nodes, un, dt=1e-2):
    N = nodes[:, 0].shape[0]
    n = int(np.sqrt(N))
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        nodes[:, 0].reshape((n, n)), 
        nodes[:, 1].reshape((n, n)), 
        un[0],                
        cmap='viridis',   
        alpha=0.8         
    )
   
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(
            nodes[:, 0].reshape((n, n)), 
            nodes[:, 1].reshape((n, n)), 
            un[frame],                
            cmap='viridis',   
            edgecolor='none', 
            alpha=0.8         
        ) 
        ax.set_title(f"t={(dt * frame):.2f}")
    fps = 300
    ani = FuncAnimation(fig, update, frames=un.shape[0], interval= un.shape[0] / fps, )
    plt.show()

def energy(u, v, M, K, m,):
    kinetic   = .5 * v.T @ M @ v
    potential = .5 * u.T @ K @ u + m * np.sum(np.ones_like(u) - np.cos(u)) 
    total     = kinetic + potential
    return total

def calculate_fd(u, v, nx, ny, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    ut = v[1:-1, 1:-1]
    ux2 = ux ** 2
    uy2 = uy ** 2
    ut2 = ut ** 2
    cos = 2 * (1 - np.cos(u[1:-1, 1:-1]))
    integrand = np.sum(ux2 + uy2 + ut2 + cos)
    # simple trapeziodal rule
    return 0.5 * integrand * dx * dy


if __name__ == '__main__':
    mesh = SimpleMesh(Lx=25, Ly=25, nx=30, ny=30)
    # c = m = 1
    solver = SineGordonFEM(mesh, 1, 1) 
    u0 =  initial_u(solver.nodes)
    v0 =  initial_v(solver.nodes)

    u_evol, v_evol = [], []
    nt = 10000
    dt = 1e-2
    T = dt * nt
    for i in tqdm(range(nt)):
        if i == 0:
            un, vn = solver.symplectic_step(u0, v0, dt)
        else:
            un, vn = solver.symplectic_step(un, vn, dt)
        u_evol.append(un)
        v_evol.append(vn)

    u_evol = np.array(u_evol)
    v_evol = np.array(v_evol)

    #animate_evolution(mesh, u_evol[::10], solver.M, dt * 10)
    
    u_evol_sv = []
    def lapl(u, Lx, Ly):
        def u_yy(a):
            dy = 2 * Ly / (a.shape[1] - 1)
            uyy = np.zeros_like(a)
            uyy[1:-1, 1:-1] = (a[1:-1, 2:] + a[1:-1, :-2] - 2 * a[1:-1, 1:-1]) / (dy ** 2)
            return uyy

        def u_xx(a):
            dx = 2 * Lx / (a.shape[0] - 1)
            uxx = np.zeros_like(a)
            uxx[1:-1, 1:-1] = (a[2:, 1:-1] + a[:-2, 1:-1] - 2 * a[1:-1, 1:-1]) / (dx ** 2)
            return uxx

        return u_xx(u) + u_yy(u)


    def step(ub, ubb, dt):
        op = lapl(ub, 25, 25) - np.sin(ub)
        u_new = 2 * ub - ubb + dt ** 2 * op
        return u_new


    N = 53
    xn = np.linspace(-25, 25, N)
    yn = np.linspace(-25, 25, N)
    X, Y = np.meshgrid(xn, yn)
    nodes = np.vstack([X.reshape(N * N), Y.reshape(N * N)]).T
    u0 = initial_u(nodes).reshape((N, N))

    u1 = u0 + .5 * dt ** 2 * (lapl(u0, 25, 25) - np.sin(u0))
    ubb = u0
    ub = u1
    u_evol_sv = [u0, u1]
    for i in tqdm(range(nt)):
        if i == 0 or i == 1:
            continue
        un = step(ub, ubb, dt)
        u_evol_sv.append(un)
        ubb = ub
        ub = un

    v_sv = [np.zeros_like(u0)]
    for i in tqdm(range(1, nt - 1)):
        v_sv.append((u_evol_sv[i + 1] - u_evol_sv[i - 1]) / (2 * dt)) 
    v_sv.append(v_sv[-1])
    u_evol_sv = np.array(u_evol_sv)  
    v_sv = np.array(v_sv)
    print(f"{4 * np.prod(v_sv.shape)/(1<<30)} GB for v_sv")

    es_fem, es_fdm = [], []

    for t in tqdm(range(nt)):
        es_fem.append(energy(u_evol[t], v_evol[t], solver.M, solver.K, solver.m))
        es_fdm.append(calculate_fd(u_evol_sv[t], v_sv[t], N, N, 2 * 25 / N, 2 * 25 / N))
    plt.plot(np.linspace(0, T, nt), es_fem, label="E: FEM, symplectic")
    plt.plot(np.linspace(0, T, nt), es_fdm, label="E: FDM, stormer-verlet")
    plt.legend()
    plt.show()

    #animate_evolution_fd(nodes, u_evol_sv[::10], dt * 10)

    for i in range(nt):
        u_evol[i] = solver.M @ u_evol[i]
    

    for i in range(0, nt, 1000):
        fig, axs = plt.subplots(ncols=2, subplot_kw={"projection":'3d'})
        axs[0].plot_trisurf(
            mesh.nodes[:, 0], 
            mesh.nodes[:, 1], 
            u_evol[i],                
            cmap='viridis',   
            edgecolor='none',          
        )
        axs[1].plot_surface(
            nodes[:, 0].reshape((N, N)), 
            nodes[:, 1].reshape((N, N)), 
            u_evol_sv[i],                
            cmap='viridis',   
            edgecolor='none',          
        )
        axs[0].set_title(f"FEM symplectic {i=}")
        axs[1].set_title(f"FD Stormer-Verlet {i=}")
        plt.show()
