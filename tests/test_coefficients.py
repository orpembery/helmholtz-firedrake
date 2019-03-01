import helmholtz_firedrake.problems as hh
import firedrake as fd
import numpy as np
from helmholtz_firedrake.coefficients import PiecewiseConstantCoeffGenerator, UniformKLLikeCoeff, SamplingError
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
    """Tests that the coeffs generated are the correct size."""

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
    """Tests that the matrices are spd, using Sylvester's criterion.

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
    """Tests that the matrices have correct noise_level.

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

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[0,0]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[1,1]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[0,1]) <= noise_level

            assert abs(A_stoch._constant_array[coords]\
                       .evaluate(None,None,(),None)[1,0]) <= noise_level

def test_kl_like_coeff():
    """Tests the KL-like coefficient."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    n_0 = 1.0

    given_points = PointsHolder(np.random.rand(10,J) - 0.5)

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,given_points)

    # The first sample is loaded by default in the current code, so I
    # think this is OK.
    for jj in range(J-1):
        print(jj)
        kl_like.sample()

        kl_like.coeff

    try:
        kl_like.sample()
    except SamplingError:
        pass

    kl_like.reinitialise()

def test_kl_like_coeff_changing_externally():
    """Tests the KL-like coefficient."""

    mesh = fd.UnitSquareMesh(10,10)

    J = 10

    delta = 2.0

    lambda_mult = 0.1

    n_0 = 1.0

    given_points = PointsHolder(np.random.rand(3,J) - 0.5)

    kl_like = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,given_points)



    kl_like.sample()

    kl_like.stochastic_points.points = kl_like.stochastic_points.points[-1:,:]
    
    try:
        kl_like.sample()
    except SamplingError:
        pass

    assert kl_like.stochastic_points.points.shape[0] == 0



# The following is just to enable the tests to take place

class PointsHolder(object):

    def __init__(self,points):

        self.points = points
