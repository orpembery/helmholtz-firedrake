import firedrake as fd
import numpy as np

class HelmholtzProblem(object):
    """Defines a finite-element approximation of a Helmholtz problem.

    Defines a finite-element approximation of the Helmholtz equation
    with heterogeneous coefficients, gives the ability to define a
    preconditioner, and provides the methods to solve the approximation
    (using GMRES) and analyse the converence a little.

    Modifiying coefficients/forcing functions can be done one of two
    ways:

    1) Using the defined 'set_' methods built into the class. This will
    cause Firedrake to re-compile new C code, and therefore is slower.

    2) If the coefficients/forcing functions are partly defined using
    Firedrake Constants, these Constants can be updated directly using
    their assign() methods. This does not cause Firedrake to recompile,
    and is faster.

    Do NOT modify the attributes directly to change the coefficients
    (when this happens, the problem must be re-initialised).

    Attributes defined:
       
        u_h - a Firedrake Function holding the numerical solution of the
        PDE (equals the zero function if solve() has not been called).

        GMRES_its - int holding the number of GMRES iterations it took
        for the solver to converge (equals -1 if solve() has not been
        called).

    Methods defined:

        set_prop (where prop is one of k, A, n, A_pre, n_pre, f, g) -
        sets the corresponding attribute and re-initialises the problem,

        solve - solves the PDE using (preconditioned) GMRES and updates
        the attributes u_h and GMRES_its with the solution and the
        number of GMRES iterations, respectively.
    """

    def __init__(self, k, V,
                 A=fd.as_matrix([[1.0,0.0],[0.0,1.0]]),n=1.0,
                 A_pre=None,n_pre=None,
                 f=1.0,g=0.0):
        """Creates an instance of HelmholtzProblem.

        Parameters:

        k - a positive float

        V - a Firedrake FunctionSpace defined on a mesh

        A - A UFL (possibly containing Firedrake Constants) expression
        for the 'diffusion coefficient'. Output should be a spatially
        heterogeneous symmetric 2x2 matrix.

        n - A UFL expression (possibly containing Firedrake Constants)
        for the 'squared slowness'. Output should be a spatially
        heterogeneous real.

        A_pre - None, or a UFL expression (possibly containing Firedrake
        Constants) for the 'diffusion coefficient' for the
        preconditioning problem. Output should be a spatially
        heterogeneous symmetric 2x2 matrix.

        n_pre - None, or a UFL expression (possibly containing Firedrake
        Constants) for the 'squared slowness' for the preconditioning
        problem. Output should be a spatially heterogeneous real.

        f - A UFL expression for the right-hand side of the Helmholtz
        PDE.

        g - A UFL expression for the right-hand side of the impedance
        boundary condition.
        """
        self._initialised = False
        
        self.set_k(k)

        self.set_A(A)

        self.set_n(n)

        self.set_A_pre(A_pre)

        self.set_n_pre(n_pre)

        self.set_f(f)

        self.set_g(g)

        self._V = V

        self.GMRES_its = -1
        """int - number of GMRES iterations. Initialised as -1."""

        self.u_h = fd.Function(self._V)
        """Firedrake function - initialised as zero function."""

    def solve(self):
        """
        Solve the Helmholtz problem.

        Solves the Helmholtz Problem, and creates attributes of the
        solution and the number of GMRES iterations. Warning - can take
        a while!"""

        if not(self._initialised):
            self._initialise_problem()
            
        self._solver.solve()

        assert isinstance(self._solver.snes.ksp.getIterationNumber(),int)
       
        self.GMRES_its = self._solver.snes.ksp.getIterationNumber()

    def _initialise_problem(self):
        """
        Set up the Firedrake machinery for solving the problem.
        """

        # Define trial and test functions on the space
        self._u = fd.TrialFunction(self._V)

        self._v = fd.TestFunction(self._V)

        # Define sesquilinear form and antilinear functional
        self._a = self._define_form(self._A,self._n)
        
        self._set_L()
        
        self._set_pre()

        # Define problem and solver (following code courtesy of Lawrence
        # Mitchell, via Slack)
        problem = fd.LinearVariationalProblem(
                      self._a, self._L, self.u_h,
                      aP=self._a_pre, constant_jacobian=False)
        
        self._solver = fd.LinearVariationalSolver(
                           problem, solver_parameters = self._solver_parameters)
        
        self._initialised = True

    def set_k(self,k):
        """Sets the wavenumber k."""

        self._k = k

        if self._initialised:
            self._initialise_problem()

    def _define_form(self,A,n):
        """Defines a Helmholtz sesquilinear form."""

        return (fd.inner(A * fd.grad(self._u), fd.grad(self._v))\
                - self._k**2 * fd.inner(n * self._u,self._v)) * fd.dx\
                - (1j* self._k * fd.inner(self._u,self._v)) * fd.ds
                    
    def set_A(self,A):
        """Set the 'diffusion coefficient' A."""

        self._A = A

        if self._initialised:
            self._initialise_problem()
        

    def set_n(self,n):
        """Set the 'squared slowness' n."""

        self._n = n

        if self._initialised:
            self._initialise_problem()
            
    def set_A_pre(self,A_pre):
        """
        Set A for the preconditioning problem.
        """

        self._A_pre = A_pre

        if self._initialised:
            self._initialise_problem()

    def set_n_pre(self,n_pre):
        """
        Set the 'squared slowness' for the preconditioning problem.
        """

        self._n_pre = n_pre

        if self._initialised:
            self._initialise_problem()

    def set_f(self,f):
        """Set the 'domain forcing function' f."""

        self._f = f

        if self._initialised:
            self._initialise_problem()
        
    def set_g(self,g):
        """Set the 'impedance boundary forcing function' g."""

        self._g = g

        if self._initialised:
            self._initialise_problem()

    def _set_pre(self):
        """
        Set the preconditioning bilinear form and the solver parameters.
        """


        if self._A_pre == None or self._n_pre == None:
            self._a_pre = None
            self._solver_parameters={"ksp_type": "gmres",
                                     "mat_type": "aij",
                                     "ksp_norm_type": "unpreconditioned"
                                     }
        else:
            self._a_pre = self._define_form(self._A_pre,self._n_pre)
                     
            self._solver_parameters={"ksp_type": "gmres",
                                     "mat_type": "aij",
                                     "pmat_type": "aij",
                                     "snes_lag_preconditioner": -1,
                                     "pc_type": "lu",
                                     "ksp_reuse_preconditioner": True,
                                     "ksp_norm_type": "unpreconditioned"
                                     } 
        
    def _set_L(self):
        """Set the right-hand side of the weak form.

        A little bit hacky, because Firedrake/UFL complains if f or g =
        0.0.
        """

        x = fd.SpatialCoordinate(self._V.mesh())
        
        if self._f == 0.0:
            self.set_f(x[0]-x[0])

        if self._g == 0.0:
            self.set_g(x[0]-x[0])

        self._L =  fd.inner(self._f,self._v)*fd.dx\
                   + fd.inner(self._g,self._v)*fd.ds

class StochasticHelmholtzProblem(HelmholtzProblem):
    """Defines a stochastic Helmholtz finite-element problem.

    All attributes and methods are as in HelmholtzProblem, except for the following additions:

    Attributes:

    A_stoch - see parameters of init.

    n_stoch - see parameters of init

    Methods:

    sample - samples (using the sample methods of A_stoch and n_stoch)
    the coefficients A and n, and updates them.
    """

    def __init__(self, k, V, A_stoch, n_stoch, **kwargs):
        """Creates an instance of StochasticHelmholtzProblem.

        All parameters are as in HelmholtzProblem, apart from:

        - A_stoch - a instance of a class with the following
          attributes/methods:

            Attributes: coeff - a realisation of the type given by A in
            HelmholtzProblem. Must be implemented using Numpy.

            Methods: sample - randomly updates coeff

            This specification is implementation-agnostic, but it's
            best to implement this using Firedrake Constants (so that
            sample() simply assign()s the values of the Constants), as
            then the form doesn't need to be recompiled for each new
            realisation.

        - n_stoch - a instance of a class with the following
          attributes/methods:

            Attributes: coeff - a realisation of the type given by n in
            HelmholtzProblem. Must be implemented using Numpy.

            Methods: sample - randomly updates coeff

            Same comment about implementation as for A_gen holds.

        - **kwargs takes a dictionary whose keys are some subset of
            [A_pre,n_pre,f,g], where these satisfy the requirements
            in HelmholtzProblem.
        """

        self._A_sample = A_stoch.sample

        self._n_sample = n_stoch.sample
              
        super().__init__(k, V, A=A_stoch.coeff, n=n_stoch.coeff,
                         **kwargs)

    def sample(self):
        """Samples the coefficients A and n."""

        self._A_sample()

        self._n_sample()
