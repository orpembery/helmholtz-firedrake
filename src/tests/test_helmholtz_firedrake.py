import helmholtz_firedrake as hh
import firedrake as fd

def test_HelmholtzProblem_init_simple():
    """Test a simple setup."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 2.0
    g = 1.1
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [
        prob._k,prob._V,prob._A,prob._n,prob._A_pre,prob._n_pre,
        prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,A_pre,n_pre,f,g,-1,0.0]

def test_HelmholtzProblem_init_f_zero():
    """Test a simple setup with f = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 0.0
    g = 1.1
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [prob._k,prob._V,prob._A,prob._n,prob._A_pre,prob._n_pre,
            prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,A_pre,n_pre,f,g,-1,0.0]

def test_HelmholtzProblem_init_g_zero():
    """Test a simple setup with g = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 2.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [prob._k,prob._V,prob._A,prob._n,prob._A_pre,prob._n_pre,
            prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,A_pre,n_pre,f,g,-1,0.0]

def test_HelmholtzProblem_init_f_g_zero():
    """Test a simple setup with f = g = 0."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = A
    n_pre = n
    f = 0.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [prob._k,prob._V,prob._A,prob._n,prob._A_pre,prob._n_pre
            ,prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,A_pre,n_pre,f,g,-1,0.0]

def test_HelmholtzProblem_init_f_g_zero():
    """Test a simple setup with no preconditioner."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = None
    n_pre = None
    f = 0.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [prob._k,prob._V,prob._A,prob._n,prob._A_pre,prob._n_pre
            ,prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,A_pre,n_pre,f,g,-1,0.0]

def test_HelmholtzProblem_init_f_g_zero():
    """Test a simple setup with only one preconditioning coeff as None."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    n = 1.1
    A_pre = None
    n_pre = 1.0
    f = 0.0
    g = 0.0
    prob = hh.HelmholtzProblem(k,V,A=A,n=n,A_pre=A_pre,n_pre=n_pre,f=f,g=g)

    assert [prob._k,prob._V,prob._A,prob._n,prob._a_pre,
            prob._f,prob._g,prob.GMRES_its,prob._u_h.vector().sum()]
    == [k,V,A,n,None,f,g,-1,0.0]

# heterogeneous A,n,A_pre,n_pre,f,g - maybe?
    
# test solve - multiple k and h?-------------------------------------------------

def test_HelmholtzProblem_solver_convergence():
    """Test that the solver is converging at the correct rate."""
    k_range = [20.0,40.0]
    num_levels = 2
    tolerance = 0.05
    
    err_L2 = numpy.empty((len(k_range),num_levels))
    err_H1 = numpy.empty((len(k_range),num_levels))

    fit_L2 = np.empty(len(k_range))
    fit_H1 = np.empty(len(k_range))
    
    for ii_k in range(len(k_range)):
        k = k_range(ii_k)
        num_points = np.ceil(np.sqrt(2.0) * k**(1.5)) * 2.0**arange(float(num_levels))
        for ii_points in range(num_levels):

            # Coarsest grid has h ~ k^{-1.5}, and then do uniform refinement
            mesh = fd.UnitSquareMesh(num_points[ii_points],num_points[ii_points])
            V = FunctionSpace(mesh, "CG", 1)

            # True solution is a plane wave
            x = fd.SpatialCoordinate(mesh)
            nu = fd.FacetNormal(mesh)
            exact_soln = fd.exp(1j * k * fd.dot(x,d))
            f = 0.0
            g = 1j*k*exp(1j*k*dot(x,d))*(dot(d,nu)-1)

            prob = hh.HelmholtzProblem(k,V,f=f,g=g)
            
            prob.solve()
            
            err_L2[ii_k,ii_points] = fd.norms.errornorm(exact_soln,prob.u_h,type="L2")
            err_H1[ii_k,ii_points] = fd.norms.errornorm(exact_soln,prob.u_h,type="H1")

            h = np.sqrt(2.0) * 1.0 / num_points

            fit_L2[ii_k] = np.polyfit(h,err_L2,deg=1)[0]

            fit_H1[ii_k] = np.polyfit(h,err_H1,deg=1)[0]

    assert all(abs(fit_L2 - 2.0) <= tolerance) and all(abs(fit_H1 - 1.0) <= tolerance)
                
def test_HelmholtzProblem_solver_exact_pc():
    """Test that solver converges in 1 GMREs iteration with exact preconditioner."""

    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    prob = hh.HelmholtzProblem(k,V)

    prob.set_A_pre(prob._A)

    prob.set_n_pre(prob._n)

    prob.solve()

    assert prob.GMRES_its == 1

    
            

    
def test_HelmholtzProblem_set_k():
    """Test that set_k assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    k = 15.0
    
    prob.set_k(k)

    assert [prob._k,prob._a]
    == [k,(fd.inner(prob._A * fd.grad(prob._u), fd.grad(prob._v))\
           - k**2 * fd.inner(prob._n * prob._u,prob._v)) * fd.dx\
        - (1j* prob._k * fd.inner(prob._u,prob._v)) * fd.ds]

def test_HelmholtzProblem_set_A():
    """Test that set_A assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    A = fd.as_matrix([[0.9,0.2],[0.2,0.8]])
    
    prob.set_A(A)

    assert [prob._A,prob._a]
    == [A,(fd.inner(A * fd.grad(prob._u), fd.grad(prob._v))\
           - prob._k**2 * fd.inner(prob._n * prob._u,prob._v)) * fd.dx\
        - (1j* prob._k * fd.inner(prob._u,prob._v)) * fd.ds]

def test_HelmholtzProblem_set_n():
    """Test that set_n assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    n = 1.1
    
    prob.set_n(n)

    assert [prob._n,prob._a]
    == [n,(fd.inner(prob._A * fd.grad(prob._u), fd.grad(prob._v))\
           - prob._k**2 * fd.inner(n * prob._u,prob._v)) * fd.dx\
        - (1j* prob._k * fd.inner(prob._u,prob._v)) * fd.ds]


def test_HelmholtzProblem_set_pre():
    """Test that set_A_pre and set_n_pre assign and re-initialise."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    A_pre = fd.as_matrix([[0.9,0.2],[0.2,0.8]])

    n_pre = 1.1
    
    prob.set_A_pre(A_pre)

    prob.set_n_pre(n_pre)

    assert [prob._A_pre,prob._n_pre,prob._solver.problem.JP]
    == [A_pre,n_pre,(fd.inner(A_pre * fd.grad(prob._u), fd.grad(prob._v))\
                     - prob._k**2 * fd.inner(n_pre * prob._u,prob._v)) * fd.dx\
        - (1j* prob._k * fd.inner(prob._u,prob._v)) * fd.ds]

def test_HelmholtzProblem_set_f():
    """Test that set_f assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    f = 1.1
    
    prob.set_f(f)

    assert [prob._f,prob._L] == [f,fd.inner(f,prob._v)*fd.dx\
                   + fd.inner(prob._g,prob._v)*fd.ds]

    def test_HelmholtzProblem_set_g():
    """Test that set_g assigns and re-initialises."""
    mesh = fd.UnitSquareMesh(100,100)
    V = FunctionSpace(mesh, "CG", 1)
    k = 20.0
    
    prob = hh.HelmholtzProblem(k,V)

    g = 1.1
    
    prob.set_g(g)

    assert [prob._g,prob._L] == [g,fd.inner(prob._f,prob._v)*fd.dx\
                   + fd.inner(g,prob._v)*fd.ds]
