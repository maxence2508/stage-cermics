from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy as np
import pandas as pd
from time import time

def create_Q(w: np.ndarray) -> np.ndarray:
    """Create the matrix Q defined in Theorem 2.13 of the
    manuscript.
    """
    N = len(w)
    Q = np.zeros((N,N))
    for i in range(N):
        Q[i,0:(i+1)] = w[0:(i+1)]
        if i < N-1:
            Q[i,i+1] = -np.sum(w[0:(i+1)])
    return Q

class liftedRHeston:
    """Class for the construction of a lifted rough Heston model
    transformed to have the canonical state space R_+^N.
    """
    def __init__(self, N:int, theta:float, lam:float, \
        nu:float, x:np.ndarray, w:np.ndarray, v_0:np.ndarray) -> None:
        """Set up the parameters for a lifted rough Heston model in the
        transformation given in Theorem 2.13 of the manuscript.

        Parameters
        ----------
        N : int
            Dimension of the lifted model. Currently only supports N=2,3.
        theta : float
            Parameter of rough Heston.
        lam : float
            Parameter of rough Heston.
        x : np.ndarray
            Notes of the quadrature formula, i.e., speeds of mean reversion.
            A vector of dimension N.
        w : np.ndarray
            Weights of the quadrature formula. A vector of dimension N.
        v_0 : np.ndarray
            Initial values of the lifted rough Heston model before the
            domain tranformation. A vector of dimension N.
        """
        assert N == 2 or N == 3, f"{N = } is not currently supported."
        self.N = N
        self.theta = theta
        self.lam = lam
        self.nu = nu
        self.w = w
        self.w_bar = np.sum(w)
        self.Q = create_Q(w)
        self.x = x
        self.G = self.Q @ np.diag(x) @ np.linalg.inv(self.Q)
        self.v_0 = v_0
        self.z_0 = self.Q @ v_0
    
class explicitExample:
    """An explicit example for the PDE.
    
    The solution is 1 + beta*t + sum(alpha * z**2)."""
    def __init__(self, liftedrheston:liftedRHeston, alpha:np.ndarray, beta:float, T:float) -> None:
        """Generate the explicit example.

        Parameters
        ----------
        liftedrheston : liftedRHeston
            The underlying lifted RHeston model as well as the domain transformation.
        alpha : np.ndarray
            Parameters of the explicit solution, a vector of dimension N.
        beta : float
            Parameter of the explicit solution.
        T : float
            Terminal time.
        """
        self.model = liftedrheston
        self.alpha = alpha
        self.beta = beta
        self.T = T
        # Check that the dimension is 2 or 3
        assert self.model.N == len(alpha), \
            f"Inconsistency in dimensions: {self.model.N} != {len(alpha)}."
        assert self.model.N in {2,3}, \
            f"N = {self.model.N}, but only N = 2 or 3 are supported."
    
    def u_exp(self, t:float, z:np.ndarray) -> float|np.ndarray:
        """Evaluate the explicit solution.

        Parameters
        ----------
        t : float
            Time.
        z : np.ndarray
            Space variable.

        Returns
        -------
        float|np.ndarray
            Value of the PDE's solution.
        """
        # Note that the kind of stupid form is dictated by my understanding of
        # FeNICSx. In other words: this works, and other version I tried did not.
        value = 1 + self.beta * t
        for i in range(self.model.N):
            value += self.alpha[i] * z[i]**2
        return value

    def f(self, z:np.ndarray) -> float|np.ndarray:
        """Source term for the explicit solution.

        Parameters
        ----------
        z : np.ndarray
            Space variable.

        Returns
        -------
        float|np.ndarray
            Value of the source term.
        """
        N = self.model.N
        temp = 0
        for i in range(self.model.N):
            for j in range(self.model.N):
                temp += self.alpha[i] * z[i] * self.model.G[i,j] * (z[j] - self.model.z_0[j])
        return self.beta - 2 * temp + \
            2 * self.model.w_bar * self.alpha[N-1] * (self.model.theta - self.model.lam * z[N-1]) * z[N-1] +\
                 self.model.nu**2 * self.model.w_bar**2 * self.alpha[N-1] * z[N-1]

class numericalParameters:
    """Numerical parameters."""
    def __init__(self, num_steps:int, x_mesh_size:int, L_low:float|np.ndarray,
                  L_up:float|np.ndarray) -> None:
        """Set the numerical parameters.

        Parameters
        ----------
        num_steps : int
            Number of time steps.
        x_mesh_size : int
            Paramter controlling the mesh in space.
        L_low : float or np.array
            The domain is chosen to be [L_low, L_up]^N.
        L_up : float or np.array
            The domain is chosen to be [L_low, L_up]^N.
        """
        self.num_steps = num_steps
        self.x_mesh_size = x_mesh_size
        self.L_low = L_low
        self.L_up = L_up

def solve_PDE(model:liftedRHeston, example:explicitExample, numPar:numericalParameters) -> tuple[float,float]:
    """Compute the numerical solution of the PDE and return 
    the L^2 and L^infty errors compared to the explicit solution.

    Parameters
    ----------
    model : liftedRHeston
        Specification of the transformed lifted rough Heston model.
    example : explicitExample
        Specification of the explicit solution and the source term.
    numPar : numericalParameters
        Specification of the numerical parameters.

    Returns
    -------
    tuple[float,float]
        L^2 error and L^infty error.
    """
    t = example.T  # Start time for the backward PDE
    dt = example.T / numPar.num_steps  # Time step size
    N = model.N

    if not isinstance(numPar.L_low, np.ndarray):
        L_low = np.full(N, numPar.L_low)
    else:
        assert len(numPar.L_low) == N
        L_low = numPar.L_low
    
    if not isinstance(numPar.L_up, np.ndarray):
        L_up = np.full(N, numPar.L_up)
    else:
        assert len(numPar.L_up) == N
        L_up = numPar.L_up

    # Set up the domain and the abstract function space
    assert N == 2 or N == 3, f"{N = } is currently not supported."
    if N == 2:
        domain = mesh.create_rectangle(MPI.COMM_WORLD,
            [L_low, L_up], 
            [numPar.x_mesh_size]*N, mesh.CellType.triangle)
    else:
        p0 = L_low
        p1 = L_up
        domain = mesh.create_box(MPI.COMM_WORLD, [p0, p1], 
            [numPar.x_mesh_size]*N, cell_type=mesh.CellType.hexahedron)

    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define the Dirichlet boundary condition. Note that this is time-dependent,
    # hence, needs to be updated at each time step.
    u_D = fem.Function(V)
    u_D.interpolate(lambda z: example.u_exp(t, z))
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

    # Set up the terminal condition.
    u_old = fem.Function(V)
    u_old.interpolate(lambda z: example.u_exp(t, z))

    # Set up the source term
    f = fem.Function(V)
    f.interpolate(example.f)

    # Set up the variational form.
    z = ufl.SpatialCoordinate(domain)
    dz = ufl.dx # to not mix x and z variable names

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # define auxiliary functions; the problem parameters are implicit arguments
    def drift(u, z):
        #assert N == 2
        #temp = ufl.Dx(u, 0) * (G[0,0] * (z[0] - z_0[0]) + G[0,1] * (z[1] - z_0[1])) + ufl.Dx(u, 1) * (G[1,0] * (z[0] - z_0[0]) + G[1,1] * (z[1] - z_0[1]))
        #return -dt * temp + dt * w_bar * (theta - lam * z[N-1]) * ufl.Dx(u, N-1)
        temp = 0
        for i in range(N):
            for j in range(N):
                temp += ufl.Dx(u, i) * model.G[i,j] * (z[j] - model.z_0[j])
        return -dt * temp + dt * model.w_bar * (model.theta - model.lam * z[N-1]) * ufl.Dx(u, N-1)


    F = (-u + drift(u,z) - 0.5 * dt * model.nu**2 * model.w_bar**2 * ufl.Dx(u, N-1)) * v * dz \
        - 0.5 * dt * model.nu**2 * model.w_bar**2 * z[N-1] * ufl.Dx(u, N-1) * ufl.Dx(v, N-1) * dz \
        + (u_old - dt * f) * v * dz

    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = create_vector(L)
    
    uh = fem.Function(V)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    for n in range(numPar.num_steps):
        # Update Diriclet boundary condition
        t -= dt
        u_D.interpolate(lambda z: example.u_exp(t, z))

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L)

        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Solve linear problem
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # Update solution at previous time step (u_n)
        u_old.x.array[:] = uh.x.array

    # Compute L2 error and error at nodes
    V_ex = fem.functionspace(domain, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(lambda z: example.u_exp(0, z))
    error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))

    # Compute values at mesh vertices
    error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
    
    return error_L2, error_max

def main():
    theta = 0.8
    lam = 1.2 # lambda
    nu = 0.7
    N = 2
    x = np.array([0.1, 3.5]) #np.array([0.1, 3.5, 4.1])
    v_0 = np.array([0.2, 0.3]) #np.array([0.2, 0.3, 0.4])
    w = np.array([0.4, 1.8]) #np.array([0.4, 1.8, 2.1])
    model = liftedRHeston(N, theta, lam, nu, x, w, v_0)

    alpha = np.array([3, 4]) #np.array([3, 4, 1.9])
    beta = 1.6
    T = 2
    example = explicitExample(model, alpha, beta, T)

    num_steps = [4, 8, 16, 32, 64, 128, 256, 512] # Number of time steps
    x_mesh_size = [4, 8, 16, 32, 64, 128, 256, 512] # mesh-size in the spatial domain in each coordinate
    L_low = np.array([-0.5, -0.5])
    L_up = np.array([3.5, 3.5])

    error_L2 = np.empty(len(num_steps))
    error_max = np.empty(len(num_steps))
    run_times = np.empty(len(num_steps))

    print(f"Run the solver on the domain [{L_low}, {L_up}]^{N}.")

    for i in range(len(num_steps)):
        print(f"num_steps = {num_steps[i]}, x_mesh_size = {x_mesh_size[i]}")
        numPar = numericalParameters(num_steps[i], x_mesh_size[i], L_low, L_up)
        start_time = time()
        ret = solve_PDE(model, example, numPar)
        run_times[i] = time() - start_time
        error_L2[i] = ret[0]
        error_max[i] = ret[1]

        print(f"Run time: {run_times[i]}")
        print(f"L2-error: {error_L2[i]:.2e}")
        print(f"Maximal error: {error_max[i]:.2e}")

    results = pd.DataFrame({'num-steps' : num_steps,
                            'x-mesh-size' : x_mesh_size,
                            'L2-error' : error_L2, 
                            'max-error' : error_max, 
                            'run-times' : run_times})

    fname = f"./results_l{L_low}_L{L_up}_N{N}.txt"
    results.to_csv(fname, sep=' ')
    #np.savetxt(fname, error_L2)
    #np.savetxt(fname, error_max)

if __name__ == '__main__':
    main()
