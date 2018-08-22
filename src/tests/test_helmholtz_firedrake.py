import helmholtz.problems as hh
import firedrake as fd
import numpy as np
import pytest
from nearby_preconditioning.experiments import PiecewiseConstantCoeffGenerator

def test_HelmholtzProblem_init_simple():
    """Test a simple setup."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 2.0
    g = 1.1
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._A_pre == A_pre
    assert prob._n_pre == n_pre
    assert prob._f == f
    assert prob._g == g
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0

def test_HelmholtzProblem_init_f_zero():
    """Test a simple setup with f = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 0.0
    g = 1.1
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._A_pre == A_pre
    assert prob._n_pre == n_pre
    # Currently not testing f, as the code sets f = x[0]-x[0], as that
    # doesn't crash Firedrake
    assert prob._g == g
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0

def test_HelmholtzProblem_init_g_zero():
    """Test a simple setup with g = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 2.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._A_pre == A_pre
    assert prob._n_pre == n_pre
    assert prob._f == f
    # Currently not testing g, as the code sets g = x[0]-x[0], as that
    # doesn't crash Firedrake
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0

def test_HelmholtzProblem_init_f_g_zero():
    """Test a simple setup with f = g = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 0.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._A_pre == A_pre
    assert prob._n_pre == n_pre
    # Currently not testing f and g, as the code sets f = g = x[0]-x[0],
    # as that doesn't crash Firedrake
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0

def test_HelmholtzProblem_init_no_pc():
    """Test a simple setup with no preconditioner."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = None
    n_pre = None
    f = 1.0
    g = 1.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._A_pre == A_pre
    assert prob._n_pre == n_pre
    assert prob._f == f
    assert prob._g == g
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0

def test_HelmholtzProblem_init_one_pc_none():
    """Test a simple setup with one preconditioning coeff as None."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = None
    n_pre = 1.0
    f = 1.0
    g = 1.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    prob._initialise_problem()

    assert prob._k == k
    assert prob._V == V
    assert prob._A == A
    assert prob._n == n
    assert prob._a_pre == None
    assert prob._f == f
    assert prob._g == g
    assert prob.GMRES_its == -1
    assert prob.u_h.vector().sum() == 0.0
    
@pytest.mark.slow  
def test_HelmholtzProblem_solver_convergence():
    """Test that the solver is converging at the correct rate."""
    k_range = [10.0,12.0,20.0,30.0,40.0]#[10.0,12.0]#[20.0,40.0]
    num_levels = 2
    tolerance = 0.05
    log_err_L2 = np.empty((num_levels,len(k_range)))
    log_err_H1 = np.empty((num_levels,len(k_range)))

    for ii_k in range(len(k_range)):
        k = k_range[ii_k]
        print(k)
        num_points = np.ceil(np.sqrt(2.0) * k**(1.5)) * 2.0**np.arange(float(num_levels))

        log_h = np.log(np.sqrt(2.0) * 1.0 / num_points)
        for ii_points in range(num_levels):
            print(ii_points)
            # Coarsest grid has h ~ k^{-1.5}, and then do uniform refinement
            mesh = fd.UnitSquareMesh(num_points[ii_points],num_points[ii_points])
            V = fd.FunctionSpace(mesh, "CG", 1)

            x = fd.SpatialCoordinate(mesh)
            nu = fd.FacetNormal(mesh)
            d = fd.as_vector([1.0/np.sqrt(2.0),1.0/np.sqrt(2.0)])
            exact_soln = fd.exp(1j * k * fd.dot(x,d))
            f = 0.0
            g = 1j*k*fd.exp(1j*k*fd.dot(x,d))*(fd.dot(d,nu)-1)

            prob = hh.HelmholtzProblem(k,V,f=f,g=g)

            prob.solve()

            log_err_L2[ii_points,ii_k] = np.log(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="L2"))
            log_err_H1[ii_points,ii_k] = np.log(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="H1"))

        #plt.plot(log_h,log_err_L2[:,ii_k])

        #plt.plot(log_h,log_err_H1[:,ii_k])

        #plt.show()

        fit_L2 = np.polyfit(log_h,log_err_L2[:,ii_k],deg=1)[0]

        fit_H1 = np.polyfit(log_h,log_err_H1[:,ii_k],deg=1)[0]

        print(fit_L2)

        print(fit_H1)

        assert abs(fit_L2 - 2.0) <= tolerance

        assert abs(fit_H1 - 1.0) <= tolerance
                
def test_HelmholtzProblem_solver_exact_pc():
    """Test solver converges in 1 GMRES iteration with exact precon.."""

    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    prob = hh.HelmholtzProblem(k,V)

    prob.set_A_pre(prob._A)

    assert prob._A_pre == prob._A

    prob.set_n_pre(prob._n)

    assert prob._n_pre == prob._n

    prob.solve()
    
#    assert prob._a_pre == prob._a

    assert prob.GMRES_its == 1

    
            

    
def test_HelmholtzProblem_set_k():
    """Test that set_k assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    k = 15.0

    prob._initialise_problem()
    
    prob.set_k(k)

    assert prob._k == k
    # At the moment, not testing that the new form is correct, as I
    # don't know how to.
    
def test_HelmholtzProblem_set_A():
    """Test that set_A assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    
    prob.set_A(A)

    assert prob._A == A
    # At the moment, not testing that the new form is correct, as I
    # don't know how to.

def test_HelmholtzProblem_set_n():
    """Test that set_n assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    n = 1.1
    
    prob.set_n(n)

    assert prob._n == n
    # At the moment, not testing that the new form is correct, as I
    # don't know how to.

def test_HelmholtzProblem_set_pre():
    """Test that set_A_pre and set_n_pre assign and re-initialise."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    A_pre = fd.as_matrix([[0.9,0.2],[0.2,0.8]])

    n_pre = 1.1
    
    prob.set_A_pre(A_pre)

    prob.set_n_pre(n_pre)

    assert prob._A_pre == A_pre
    # At the moment, not testing that the new form is correct, as I
    # don't know how to.
    
def test_HelmholtzProblem_set_f():
    """Test that set_f assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    f = 1.1
    
    prob.set_f(f)

    assert prob._f == f
    # At the moment, not testing that the new rhs is correct, as I
    # don't know how to.

def test_HelmholtzProblem_set_g():
    """Test that set_g assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    g = 1.1
    
    prob.set_g(g)

    assert prob._g == g
    # At the moment, not testing that the new rhs is correct, as I
    # don't know how to.
    
def test_StochasticHelmholtzProblem_sample():
    """Test that sample routine works correctly."""

    class DeterministicCoeff(object):
        """Generates 'random' coefficients in a known way.

        Coefficients are of the type required in
        StochasticHelmholtzProblem, but matrix-valued coefficients take
        the values [[1.0,0.0],[0.0,1.0/n]] and scalar-valued
        coefficients take the value 1.0 = 1.0/n, where n >= 1 is the
        number of times the coefficient has been sampled.
        """

        def __init__(self,type):
            """Initialises testing class."""
            self._counter = 1

            self._changes = fd.Constant(1.0)

            self._type = type

            if type == "matrix":
                self.coeff = fd.as_matrix([[1.0,0.0],[0.0,1.0/self._changes]])

            if type == "scalar":
                self.coeff = 1 + 1.0/self._changes

        def sample(self):
            """Update coefficient."""

            self._counter += 1
            
            self._changes.assign(float(self._counter))

    A_test = DeterministicCoeff("matrix")

    n_test = DeterministicCoeff("scalar")

    k = 20.0
    
    mesh = fd.UnitSquareMesh(100,100)

    V = fd.FunctionSpace(mesh, "CG", 1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_test,n_test)

    num_samples = 999
    
    for ii in range(num_samples):
        prob.sample()

    # The following test isn't perfect, but it shows that the `sample'
    # method for StochasticHelmholtzProblem is calling the `sample'
    # method of the coefficients correctly. (It doesn't show if the
    # embedding in the form is correct.)
    assert n_test._changes.values() == float(num_samples + 1)
    assert A_test._changes.values() == float(num_samples + 1)

def test_force_solver_params():
    """Test that forcing an LU solver works."""

    mesh = fd.UnitSquareMesh(100,100)

    V = fd.FunctionSpace(mesh,"CG",1)
    
    k = 20.0

    prob = hh.HelmholtzProblem(k,V)

    prob.force_lu()

    assert prob._solver_parameters["ksp_type"] == "preonly"

    assert prob._solver_parameters["pc_type"] == "lu"

def test_force_solver_params():
    """Test that 'unforcing' an LU solver works."""

    mesh = fd.UnitSquareMesh(100,100)

    V = fd.FunctionSpace(mesh,"CG",1)
    
    k = 20.0

    prob = hh.HelmholtzProblem(k,V)

    prob.force_lu()

    prob.unforce_lu()

    assert prob._solver_parameters["ksp_type"] != "preonly"

def test_A_stoch_none():
    """Test that setting not setting A_stoch doesn't misbehave."""

    k = 10.0

    mesh = fd.UnitSquareMesh(100,100)

    V = fd.FunctionSpace(mesh, "CG", 1)

    n_stoch = PiecewiseConstantCoeffGenerator(mesh,2,0.1,1.0,[1])

    prob = hh.StochasticHelmholtzProblem(k,V,n_stoch=n_stoch)

def test_n_stoch_none():
    """Test that not setting n_stoch doesn't misbehave."""

    k = 10.0

    mesh = fd.UnitSquareMesh(100,100)

    V = fd.FunctionSpace(mesh, "CG", 1)

    A_stoch = PiecewiseConstantCoeffGenerator(
        mesh,2,0.1,fd.as_matrix([[1.0,0.0],[0.0,1.0]]),[2,2])

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=A_stoch)
