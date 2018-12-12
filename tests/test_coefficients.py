import helmholtz_firedrake.problems as hh
import firedrake as fd
import numpy as np
import helmholtz_firedrake.coefficients as hh_coeff
import helmholtz_firedrake.utils as hh_utils
import copy

def test_coeff_being_updated():
    """Test that the random coefficients are actually updated."""

    k = 20.0
    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 10
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,n_pre,[1])

    A_copy = copy.deepcopy(A_stoch._constant_array[0,0].values())

    n_copy = copy.deepcopy(n_stoch._constant_array[0,0].values())

    A_stoch.sample()

    n_stoch.sample()

    A_diff = A_copy - A_stoch._constant_array[0,0].values()

    assert all(A_copy != 0.0)

    assert n_copy != n_stoch._constant_array[0,0].values()    
    
def test_coeff_size():
    """Tests that the coeffs generated are the correct size."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 1
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    n_pre = 1.0
    n_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
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
    """Tests that the matrices are spd, using Sylvester's criterion.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
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
    """Tests that the matrices have correct noise_level.

    Only works for the case coeff_pre = I."""

    mesh = fd.UnitSquareMesh(100,100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    num_pieces = 12
    noise_level = 0.1
    num_repeats = 100
    
    A_pre = fd.as_matrix(np.array([[1.0,0.0],[0.0,1.0]]))
    A_stoch = hh_coeff.PiecewiseConstantCoeffGenerator(mesh,num_pieces,
                                                   noise_level,A_pre,[2,2])

    for ii in range(num_repeats):
        A_stoch.sample()

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

def test_lognormal_L1_exponential_runs():
    """Test the lognormal RF whose log has L1 exponential covariance."""

    mesh = fd.UnitSquareMesh(10,10)

    mean = 0.0

    sigma = 1.0

    l = (0.1,0.15)

    J = 10

    hh_coeff.LognormalL1ExponentalCovarianceCoeff(mesh,mean,sigma,l,J)
