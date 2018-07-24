import firedrake as fd

class HelmholtzProblem:
    """Defines a finite-element approximation of a Helmholtz problem.

    Defines a finite-element approximation of the Helmholtz equation with heterogeneous coefficients, gives the ability to define a preconditioner, and provides the methods to solve the approximation (using GMRES) and analyse the converence a little.
    """

    def __init__(self, mesh, V, A, n, boundary_condition_type="Impedance", f, g, aP=None):
        """Creates an instance of HelmholtzProblem.

        mesh - a mesh object created by fd.Mesh (or one of Firedrake's utitlity mesh functions)

        V - a FunctionSpace defined on mesh

        A - A (function that generates a?) UFL expression for the 'diffusion coefficient'. Output should be a spatially heterogeneous symmetric 2x2 matrix

        n - A (function that generates a?) UFL expression for the 'squared slowness'. Output should be a spatially heterogeneous real

        boundary_condition_type - whether to use an Impedance boundary condition, or another type. (Currently only Impedance boundary conditions are supported.)

        f - A UFL expression for the right-hand side of the Helmholtz PDE

        g - A UFL expression for the right-hand side of an impedance boundary condition (if boundary_condition_type = "Impedance") or None

        aP - An instance of HelmholtzProblem with the same mesh, V, and boundary_condition_type; or None.
        """

        # The following is wrong - I should instead raise and exception or show an error message - as in https://wiki.python.org/moin/UsingAssertionsEffectively
        assert isinstance(mesh,fd.Mesh)

        assert isinstance(V,fd.FunctionSpace)

        assert V.mesh() == mesh

        assert boundary_condition_type == "Impedance", 


