import firedrake as fd

class HelmholtzProblem:
    """Defines a finite-element approximation of a Helmholtz problem.

    Defines a finite-element approximation of the Helmholtz equation with heterogeneous coefficients, gives the ability to define a preconditioner, and provides the methods to solve the approximation (using GMRES) and analyse the converence a little.
    """

    def __init__(self, mesh, V, k, A, n, f, g, boundary_condition_type="Impedance", aP=None):
        """Creates an instance of HelmholtzProblem.

        mesh - a mesh object created by fd.Mesh (or one of Firedrake's utitlity mesh functions)

        V - a FunctionSpace defined on mesh

        k - a positive float

        A - A (function that generates a?) UFL expression for the 'diffusion coefficient'. Output should be a spatially heterogeneous symmetric 2x2 matrix

        n - A (function that generates a?) UFL expression for the 'squared slowness'. Output should be a spatially heterogeneous real

        boundary_condition_type - whether to use an Impedance boundary condition, or another type. (Currently only Impedance boundary conditions are supported.)

        f - A UFL expression for the right-hand side of the Helmholtz PDE

        g - Either a UFL expression for the right-hand side of an impedance boundary condition (if boundary_condition_type = "Impedance") or None

        Syntax for A, n, f, and g: x = fd.SpatialCoordinate(mesh), nu = FacetNormal(mesh)
        
        aP - Either an instance of HelmholtzProblem with the same mesh, V, and boundary_condition_type; or None.

        Attributes:

        mesh - as above

        V - as above

        k - as above

        A - as above

        n - as above

        f - as above

        g - as above

        aP - as above

        u_h - a Firedrake Function holding the numerical solution of the PDE (equals the zero function if solve() has not been called)

        GMRES_its - int holding the number of GMRES iterations it took for the solver to converge (equals None if solve() has not been called)
        """

        if not(isinstance(mesh,fd.mesh.MeshGeometry)):
            raise UserInputError("Input argument 'mesh' is not a Firedrake Mesh.")

        elif not(isinstance(V,fd.functionspaceimpl.WithGeometry)):
            raise UserInputError("Input argument 'V' is not a Firedrake FunctionSpace.")

        elif V.mesh() != mesh:
            raise UserInputError("Function space 'V' is not defined on 'mesh'.")

        elif not(isinstance(k,float)):
            raise UserInputError("Wavenumber k should be a float.")

        elif k <= 0:
            raise UserInputError("Wavenumber k should be positive.")
        
        elif boundary_condition_type != "Impedance":
            raise UserInputError("Only Impedance boundary conditions currently implemented.")

        elif boundary_condition_type == "Impedance" and g == None:
            raise UserInputError("Impedance boundary data g must be defined.")

        elif aP != None:

            if not(isinstance(aP,HelmholtzProblem)):
                raise UserInputError("Input argument aP must be an instance of HelmholtzProblem.")

            elif aP.mesh != mesh:
                raise UserInputError("Preconditioner aP must be defined on the mesh 'mesh'.")

            elif aP.V != V:
                raise UserInputError("Preconditioner aP must be define the Function Space 'V'.")


        self.mesh = mesh

        self.V = V

        self.k = k
        
        self.A = A

        self.n = n

        self.f = f

        self.g = g

        self.aP = aP

        # Define numerical solution
        self.u_h = fd.Function(V)

        self.GMRES_its = None
        
        # Set up finite-element problem
        
        # Define trial and test functions on the space
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        x = fd.SpatialCoordinate(mesh)

        nu = fd.FacetNormal(mesh)

        # Define sesquilinear form and antilinear functional
        a = (fd.inner(A * fd.grad(u), fd.grad(v)) - k**2 * fd.inner(n * u,v)) * fd.dx - (1j* k * fd.inner(u,v)) * fd.ds
        L =  fd.inner(g,v)*fd.ds

        # Define problem and solver (following code courtesy of Lawrence Mitchell, via Slack)
        if aP == None:
            a_pre = None
            solver_parameters={"ksp_type": "gmres",
                               "mat_type": "aij",
                               "ksp_norm_type": "unpreconditioned"
                               }
        else:
            a_pre = (inner(aP.A * grad(u), grad(v)) - aP.k**2 * inner(aP.n * u,v)) * dx - (1j* aP.k * inner(u,v)) * ds
            solver_parameters={"ksp_type": "gmres",
                               "mat_type": "aij",
                               "pmat_type": "aij",
                               "snes_lag_preconditioner": -1,
                               "pc_type": "lu",
                               "ksp_reuse_preconditioner": True,
                               "ksp_norm_type": "unpreconditioned"
                               }
            
        problem = fd.LinearVariationalProblem(a, L, self.u_h, aP=a_pre, constant_jacobian=False)
        self.solver = fd.LinearVariationalSolver(problem, solver_parameter = solver_parameters)

    def solve(self):
        """Solves the Helmholtz Problem, and creates attributes of the solution and the number of GMRES iterations. Warning - can take a while!"""

        self.solver.solve()

        self.GMRES_its = self.solver.snes.ksp.getIterationNumber()
    
class UserInputError(Exception):
    """Error raised when the user fails to supply correct inputs.
    
    Attributes:
        message - Error message explaining the error.
    """

    def __init__(self,message):

        self.message = message

    
