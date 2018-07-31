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

    1) Using the 'set_' methods built into the class. This will cause
    Firedrake to re-compile new C code, and therefore is slower.

    2) If the coefficients/forcing functions are partly defined using
    Firedrake Constants, these Constants can be updated directly using
    their assign() methods. This does not cause Firedrake to recompile,
    and is faster.

    Do NOT modify the attributes directly to change the coefficients
    (when this happens, the problem must be re-initialised).
    """

    def __init__(self, k, V,
                 A=fd.as_matrix([[1.0,0.0],[0.0,1.0]]),n=1.0,
                 A_pre=None,n_pre=None,
                 f=1.0,g=0.0):
        """Creates an instance of HelmholtzProblem.

        Arguments:

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

        
        Attributes defined:

        u_h - a Firedrake Function holding the numerical solution of the
        PDE (equals the zero function if solve() has not been called).

        GMRES_its - int holding the number of GMRES iterations it took
        for the solver to converge (equals -1 if solve() has not been
        called).
        """
        self._set_initialised(False)
        
        self.set_k(k)

        self.set_A(A)

        self.set_n(n)

        self.set_A_pre(A_pre)

        self.set_n_pre(n_pre)

        self.set_f(f)

        self.set_g(g)

        self.set_V(V)

        self._set_GMRES_its()

        self._initialise_u_h()

    def solve(self):
        """
        Solves the Helmholtz Problem, and creates attributes of the
        solution and the number of GMRES iterations. Warning - can take
        a while!"""

        if not(self._initialised):
            self._initialise_problem()
       
        self._solver.solve()

        assert isinstance(self._solver.snes.ksp.getIterationNumber(),int)
        
        self._set_GMRES_its(self._solver.snes.ksp.getIterationNumber())

    def _initialise_problem(self):
        """
        Sets up all the TrialFunction, TestFunction etc. machinery
        for solving the Helmholtz problem.
        """

        # Define trial and test functions on the space
        self._u = fd.TrialFunction(self._V)
        self._v = fd.TestFunction(self._V)

        # Define sesquilinear form and antilinear functional
        self._a = (fd.inner(self._A * fd.grad(self._u), fd.grad(self._v))\
                   - self._k**2 * fd.inner(self._n * self._u,self._v)) * fd.dx\
                   - (1j* self._k * fd.inner(self._u,self._v)) * fd.ds
        self._set_L()
        
        self._set_pre()

        # Define problem and solver (following code courtesy of Lawrence
        # Mitchell, via Slack)
        problem = fd.LinearVariationalProblem(
                      self._a, self._L, self.u_h,
                      aP=self._a_pre, constant_jacobian=False)
        
        self._solver = fd.LinearVariationalSolver(
                           problem, solver_parameter = self._solver_parameters)

        self._set_initialised(True)

    def set_k(self,k):

        """Sets the wavenumber k."""

        self._k = k

        if self._initialised:
            self._initialise_problem()
        
    def set_A(self,A):
        """Sets the 'diffusion coefficient' A."""

        self._A = A

        if self._initialised:
            self._initialise_problem()
        

    def set_n(self,n):
        """Sets the 'squared slowness' n."""

        self._n = n

        if self._initialised:
            self._initialise_problem()
            
    def set_A_pre(self,A_pre):
        """
        Sets the 'diffusion coefficient' for the preconditioning
        problem.
        """

        self._A_pre = A_pre

        if self._initialised:
            self._initialise_problem()

    def set_n_pre(self,n_pre):
        """
        Sets the 'squared slowness' for the preconditioning problem.
        """

        self._n_pre = n_pre

        if self._initialised:
            self._initialise_problem()

    def set_f(self,f):
        """Sets the 'domain forcing function' f."""

        self._f = f

        if self._initialised:
            self._initialise_problem()
        
    def set_g(self,g):
        """Sets the 'impedance boundary forcing function' g."""

        self._g = g

        if self._initialised:
            self._initialise_problem()

    def _set_GMRES_its(self,GMRES_its=-1):
        """
        Sets the number of GMRES iterations needed to solve the
        Helmholtz problem.
        """
        self.GMRES_its = GMRES_its

    def _set_u_h(self,u_h):
        """Sets the finite-element solution of the Helmholtz problem."""

        self.u_h = u_h

    def set_V(self,V):
        """Sets the function space."""

        self._V = V


    def _set_initialised(self,TF):
        """Says whether the Helmholtz Problem has been initialised."""

        self._initialised = TF

    def _initialise_u_h(self):
        """Initialises the Function to hold the solution."""

        self.u_h = fd.Function(self._V)

    def _set_pre(self):
        """
        Sets the preconditioning bilinear form and the solver
        parameters.
        """


        if self._A_pre == None and self._n_pre == None:
            self._a_pre = None
            self._solver_parameters={"ksp_type": "gmres",
                                     "mat_type": "aij",
                                     "ksp_norm_type": "unpreconditioned"
                                     }
        else:
            self._a_pre =\
                (fd.inner(self._A_pre * fd.grad(self._u),fd.grad(self._v))\
                - self._k**2 * fd.inner(self._n_pre * self._u,self._v))* fd.dx\
                - (1j* self.k * fd.inner(self._u,self._v)) * fd.ds
                     
            self._solver_parameters={"ksp_type": "gmres",
                                     "mat_type": "aij",
                                     "pmat_type": "aij",
                                     "snes_lag_preconditioner": -1,
                                     "pc_type": "lu",
                                     "ksp_reuse_preconditioner": True,
                                     "ksp_norm_type": "unpreconditioned"
                                     } 
        
    def _set_L(self):
        """Sets the right-hand side of the weak form. A little bit
        hacky, because Firedrake/UFL complains if f or g = 0.0.
        """

        x = fd.SpatialCoordinate(self._V.mesh())
        
        if self._f == 0.0:
            self.set_f(x[0]-x[0])

        if self._g == 0.0:
            self.set_g(x[0]-x[0])

        self._L =  fd.inner(self._f,self._v)*fd.dx\
                   + fd.inner(self._g,self._v)*fd.ds

        















        

class StochasticHelmholtzProblem(HelmholtzProblem):
    """Defines a stochastic Helmholtz finite-element problem."""

    def __init__(self, k, V, A_stoch, n_stoch, seed=1, **kwargs):
        """Creates an instance of StochasticHelmholtzProblem.

        All arguments are as in HelmholtzProblem, apart from:

        - A_stoch - a instance of a class with the following
          attributes/methods:

            Attributes: A - a realisation of the type given by A in
            HelmholtzProblem. Must be implemented using Numpy.

            Methods: sample - randomly updates A

            (This is specification is implementation-agnostic, but it's
            best to implement this using Firedrake Constants, as then
            the form doesn't need to be recompiled for each new
            realisation.)

        - n_stoch - a instance of a class with the following
          attributes/methods:

            Attributes: n - a realisation of the type given by n in
            HelmholtzProblem. Must be implemented using Numpy.

            Methods: sample - randomly updates n

            (Same comment about implementation as for A_gen holds.)

        - **kwargs takes a dictionary whose keys are some subset of
            {A_pre,n_pre,f,g}, where these satisfy the requirements
            in HelmholtzProblem.
        """

        self._set_A_sample(A_stoch.sample)

        self._set_n_sample(n_stoch.sample)

        self.set_seed(seed)
               
        super().__init__(k, V, A=A_stoch.A, n=n_stoch.n,
                         **kwargs)

    def sample(self):
        """Samples the coefficients A and n."""

        self._A_sample()

        self._n_sample()

    def _set_A_sample(self,method):
        """Sets the method for sampling A."""

        self._A_sample = method

    def _set_n_sample(self,method):
        """Sets the method for sampling n."""

        self._n_sample = method

    def set_seed(self,seed):
        """Sets the random seed."""
        self._seed = seed
    

class HelmholtzNotImplementedError(Exception):
    """Error raised when a given feature isn't implemented in
    helmholtz_firedrake yet.
    
    Attributes:
        message - Error message explaining the error.
    """

    def __init__(self,message):

        self.message = message
