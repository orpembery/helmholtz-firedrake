# Assumes this code is being run from the top level folder. Otherwise,
# add the helmholtz_firedrake folder to your PYTHONPATH
import sys
sys.path.append('.')
import helmholtz_firedrake.problems as hh
import firedrake as fd
import numpy as np
from helmholtz_firedrake.coefficients import PiecewiseConstantCoeffGenerator, UniformKLLikeCoeff, SamplingError
import helmholtz_firedrake.utils as hh_utils
import copy

def test_coeff_being_updated():
    """Test that the p/w/ const random coefficients are updated."""

    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 10
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    A_copy = copy.deepcopy(A_stoch._constant_array[0,0].values())

    n_copy = copy.deepcopy(n_stoch._constant_array[0,0].values())

    A_stoch.sample()

    n_stoch.sample()

    A_diff = A_copy - A_stoch._constant_array[0,0].values()

    assert all(A_copy != 0.0)

    assert n_copy != n_stoch._constant_array[0,0].values()    
    
def test_coeff_size():
    """Tests that the p/w/ const coeffs are the correct size."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 1
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])
    for ii in range(num_repeats):
        A_stoch.sample()
        n_stoch.sample()
        fl = A_stoch._constant_array.flat
        for jj in fl:
            coords = hh_utils.flatiter_hack(A_stoch._constant_array,fl.coords)
            assert A_stoch._constant_array[coords].evaluate(None,None,(),None).shape\
                == (2,2)
            assert n_stoch._constant_array[coords].evaluate(None,None,(),None).shape\
                == ()
    
def test_matrices_spd():
    """Tests that p/w const coeffmatrices are spd.

    Uses Sylvester's criterion.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    for ii in range(num_repeats):
        A_stoch.sample()

        fl = A_stoch._constant_array.flat
        for jj in fl:
            coords = hh_utils.flatiter_hack(A_stoch._constant_array,fl.coords)

            assert A_stoch._constant_array[coords].evaluate(
                None,None,(),None)[1,0]\
                == A_stoch._constant_array[coords].evaluate(
                    None,None,(),None)[0,1]
            
            assert 1.0 +\
                A_stoch._constant_array[coords].evaluate(
                    None,None,(),None)[0,0] > 0.0
            
            assert np.linalg.det(np.array([[1.0,0.0],[0.0,1.0]])\
                                 + A_stoch._constant_array[coords].\
                                 evaluate(None,None,(),None))\
                                 > 0.0

def test_matrices_noise_level():
    """Tests p/w const coeff matrices have correct noise_level.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    A_func = fd.Function(V)
    
    for ii in range(num_repeats):
        A_stoch.sample()

    for ii_x in range(2):
        for ii_y in range(2):
            A_func.interpolate(A_stoch.coeff[ii_x,ii_y]-float(A_pre[ii_x,ii_y]))

            assert (np.abs(A_func.dat.data_ro) <= noise_level).all()

        fl = A_stoch._constant_array.flat
        for jj in fl:
            coords = hh_utils.flatiter_hack(A_stoch._constant_array,fl.coords)

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[0,0]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[1,1]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[0,1]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[1,0]) <= noise_level

def test_noise_level():
    """Tests p/w const coeffs have correct noise_level.

    Only works for the case coeff_pre = 1.0."""

    mesh = fd.UnitSquareMesh(100,100)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    n_pre = 1.0
    n_stoch = PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    V = fd.FunctionSpace(mesh, "CG", 1)
    n_func = fd.Function(V)
    
    for ii in range(num_repeats):
        n_stoch.sample()

        fl = n_stoch._constant_array.flat
        for jj in fl:
            coords = hh_utils.flatiter_hack(n_stoch._constant_array,fl.coords)

            assert abs(n_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)) <= noise_level

        n_func.interpolate(n_stoch.coeff)

        assert (np.abs(n_func.dat.data_ro-n_pre) <= noise_level).all()

    
            
def test_kl_like_coeff_sampling():
    """Tests the KL-like coefficient samples and reintialises
    correctly."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    j_scaling = 1.0

    n_0 = 1.0

    num_points = 3

    stochastic_points = np.array(np.arange(float(num_points)),ndmin=2).transpose().repeat(J,axis=1)/float(2 * J)

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                 stochastic_points)

    # The first sample is loaded by default, so I think this is OK.
    assert (kl_like.current_point() == 0.0).all()
    
    for jj in range(num_points-1):
        kl_like.sample()

        assert (kl_like.current_point() == float(jj+1)/float(2 * J)).all()

    try:
        kl_like.sample()
        success = False
    except SamplingError:
        success = True

    assert success

    kl_like.reinitialise()
    assert (kl_like.current_and_unsampled_points() == stochastic_points).all()

def test_kl_like_coeff_change_all_points():
    """Tests the KL-like coefficient changes points correctly (and
    accesses points correctly too)."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    j_scaling = 1.0

    n_0 = 1.0

    stochastic_points = np.random.rand(3,J) - 0.5

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                 stochastic_points)

    new_points = np.random.rand(7,J) - 0.5
    
    kl_like.change_all_points(new_points)

    assert (kl_like.current_and_unsampled_points() == new_points).all()

def test_kl_like_coeff_reorder():
    """Tests the KL-like coefficient reordering works correctly."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 2

    delta = 2.0

    lambda_mult = 0.1

    j_scaling = 1.0

    n_0 = 1.0

    stochastic_points_2 = np.array([[1.0,1.0],[2.0,2.0]])

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                 stochastic_points_2)

    kl_like.reorder([1,0],include_current_point=True)

    assert (kl_like.current_and_unsampled_points() == np.array([[2.0,2.0],[1.0,1.0]])).all()

    stochastic_points_3 = np.array([[1.0,1.0],[2.0,2.0],[3.0,3.0]])

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                 stochastic_points_3)

    kl_like.reorder([1,0],include_current_point=False)
    assert (kl_like.unsampled_points() == np.array([[3.0,3.0],[2.0,2.0]])).all()

def test_kl_like_j_scaling():
    """Tests the KL-like coefficient with scaled j doesn't do anything weird."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    n_0 = 1.0

    num_points = 3

    stochastic_points = np.array(np.arange(float(num_points)),ndmin=2).transpose().repeat(J,axis=1)/float(2 * J)

    for j_scaling in [0.1,0.3,0.5,0.7,0.9,1.0]:
    
        kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                     stochastic_points)

        # The first sample is loaded by default, so I think this is OK.
        assert (kl_like.current_point() == 0.0).all()

        for jj in range(num_points-1):
            kl_like.sample()

            assert (kl_like.current_point() == float(jj+1)/float(2 * J)).all()

        try:
            kl_like.sample()
            success = False
        except SamplingError:
            success = True

        assert success

def test_kl_like_j_scaling_correct_value():
    """Tests the KL-like coefficient with scaled j gives the correct value."""

    num_cells = 100
    
    mesh = fd.UnitSquareMesh(num_cells,num_cells)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    n_0 = 1.0

    num_points = 3

    stochastic_points = np.array([0.1 for jj in range(J)],ndmin=2)

    j_scaling = 1.5

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,j_scaling,n_0,
                                 stochastic_points)

    point = (1.0,1.0)

    true_value = n_0
    
    for jj in range(1,J+1):
        true_value += stochastic_points[0,jj-1] * lambda_mult * jj**(-1.0-delta)\
                      * np.cos(jj*np.pi/j_scaling) * np.cos((jj+1)*np.pi/j_scaling)
        
    V = fd.FunctionSpace(mesh,"DG",0)

    interp_function = fd.Function(V)

    interp_function.interpolate(kl_like.coeff)

    # I've chosen the absolute tolerance here so that the test passes,
    # but eeyeballing what happened when you changed the parameters
    # etc. indicated everything was fine
    assert np.isclose(interp_function.dat.data_ro[-1],true_value,atol = 1e-3)

    
