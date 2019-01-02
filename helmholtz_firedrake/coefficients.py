import firedrake as fd
import numpy as np
import helmholtz_firedrake.utils as utils
from copy import deepcopy

class PiecewiseConstantCoeffGenerator(object):
    """Does the work of A_stoch and n_stoch in
    StochasticHelmholtzProblem for the case of piecewise continuous on
    some grid.

    Floats are Unif(noise_level,noise_level) perturbations
    of coeff_pre (see init).

    Matrices (call them A) are generated so that if coeff_pre is the 2x2
    identity, then the entrywise L^\infty norm of A-I is <= noise_level
    almost surely, and the matrices are s.p.d..

    The method for construction can almost certainly be made better, but
    is a bit of a hack until they've sorted complex so that I can do
    things the way Julian Andrei (Kiel) does.

    Attribute:

    coeff - a UFL expression for a piecewise constant (scalar- or
    matrix-valued) coefficient, implemented using Firedrake Constants.

    Method:

    sample - randomly updates coeff by randomly sampling around the
    known background given by the input argument coeff_pre. Samples have
    (entrywise) L^\infty norm <= noise_level almost surely.
    """

    def __init__(self,mesh,num_pieces,noise_level,coeff_pre,coeff_dims):
        """Initialises a piecewise-constant random coefficient, where
        the coefficient is peicewise-constant on a num_pieces x
        num_pieces grid.

        Parameters:

        mesh - a Firedrake mesh object.

        num_pieces - int - the number of `pieces' in each direction for
        the piecewise-constant coefficients. Empirically must be <= 13,
        or get a recursion error.

        noise_level - positive float - the level of the random noise.

        coeff_pre - a UFL expression with dimension coeff_dims - the
        background around which we take piecewise-constant random
        perturbations.

        coeff_dims - a list, either [1] or [2,2] - the `dimension' of
        the coefficient (i.e., either scalar-valued or 2x2 matrix
        valued).
        """

        self._coeff_dims = coeff_dims
        """Dimension of the coefficient - [1] or [2,2]."""

        self._num_pieces = num_pieces
        """Number of pieces in each direction\
        for random coefficients."""

        self._noise_level = noise_level
        """Magnitude of random noise."""
                                                  
        self._coeff_initialise(mesh,coeff_pre)
        """Create realisation of random coefficient."""

        self.sample()
        

    def _list_extract(self,values_list,x_coord,y_coord,coord_length):
        """If values_list were put into a coord_length x
        coord_length array, extracts the item at position
        (x_coord,y_coord).
        
        Parameters:
        
        values_list - list of length coord_length**2.

        x_coord, y_coord - int in range(coord_length).

        coord_length - int.
        """
        return values_list[x_coord + y_coord * coord_length]

    def _coeff_initialise(self,mesh,coeff_pre):
        """Initialises self.coeff equal to coeff_pre, but sets up
        Firedrake Constant structure to allow for sampling.

        Parameters:

        mesh - a Firedrake mesh.

        coeff_pre - see init.
        """

        if self._coeff_dims == [2,2]\
                and coeff_pre != fd.as_matrix([[1.0,0.0],[0.0,1.0]]):

            warnings.warn("coeff_pre is not the identity. There is no\
            guarantee that the randomly-generated matrices are\
            positive-definite, or have the correct amount of noise.")

        d = mesh.geometric_dimension()
            
        # Bit of a hack, set up num_pieces
        num_pieces_list = []
        [num_pieces_list.append(self._num_pieces) for ii in range(d)]
            
        # Set up all the Firedrake Constants:
        self._constant_array = np.empty(num_pieces_list,dtype=object)
        ca_flat = self._constant_array.flat
        
        for ii in ca_flat:
            coords = utils.flatiter_hack(self._constant_array,ca_flat.coords)
            
            if self._coeff_dims == [2,2]:
                self._constant_array[coords] = fd.Constant(
                    np.array([[0.0,0.0],[0.0,0.0]]),domain=mesh)
            elif self._coeff_dims == [1]:
                self._constant_array[coords] = fd.Constant(0.0,domain=mesh)

        # You can't reinitialise a flatiter (I think)
        del(ca_flat)
                
        # Form coeff by looping over all the subdomains
        x = fd.SpatialCoordinate(mesh)

        self.coeff = coeff_pre

        ca_flat = self._constant_array.flat
        
        for ii in ca_flat:

            coords = utils.flatiter_hack(self._constant_array,ca_flat.coords)

            coords_t = np.array(coords,dtype=float).transpose()
            
            loc_array = np.vstack((coords_t,coords_t + 1.0))\
                        /float(self._num_pieces)
            
            self.coeff += utils.nd_indicator(
                x,self._constant_array[coords],loc_array)

    def sample(self):
        """Samples the coefficient coeff."""
        
        ca_flat = self._constant_array.flat

        for ii in ca_flat:
            coords = utils.flatiter_hack(self._constant_array,ca_flat.coords)
            self._constant_array[coords].assign(self._generate_matrix_coeff())

    def _generate_matrix_coeff(self):
        """Generates a realisation of the random coefficient.

        For matrices, uses the fact that for real matrices,
        [[1+a,b],[b,1+c]] is positive-definite iff 1+a>0 and b**2 <
        (1+a)*(1+c).

        Let L denote self._noise_level. Matrices are of the form A =
        coeff_pre + [[a,b],[b,c]], where a,c ~ Unif(-L,0) and, letting
        b_bound = min{L,sqrt((1+a)*(1+c))}, b ~ Unif(-b_bound,b_bound)
        with a,b,c independent of each other. This construction
        guarantees that if coeff_pre = I, then A is s.p.d. and the
        entrywise L^\infty norm of A-I is bounded by self._noise_level
        almost surely.
        """

        if self._coeff_dims == [1]:
            coeff = self._noise_level*(
                2.0 * np.random.random_sample(self._coeff_dims) - 1.0)
        elif self._coeff_dims == [2,2]:
            a = self._noise_level * np.random.random_sample(1)
            c = self._noise_level * np.random.random_sample(1)
            b_bound = min(self._noise_level,np.sqrt((1.0+a)*(1.0+c)))
            b = b_bound * (2.0 * np.random.random_sample(1) - 1.0)

            coeff = np.array([[a,b],[b,c]])
        return coeff


class GammaConstantCoeffGenerator(object):
    """Does the work of n_stoch in StochasticHelmholtzProblem for the
    case of a constant but gamma-distributed refractive index.

    Attribute:

    coeff - a Firedrake Constant containing the value of the
    scalar-valued coefficients.

    Method:

    sample - randomly updates coeff by randomly sampling from coeff_base
    + gamma(rate).
    """

    def __init__(self,shape,scale,coeff_lower_bound):
        """Initialises a constant, gamma-distributed constant.

        Parameters:

        shape - the shape of the gamma distribution (commonly called k).

        scale - the scale of the gamma distribution (commonly
        called theta).

        coeff_lower_bound - the lower bound for the coefficient.

        The mean of the gamma distribution is shape * scale and the
        variance is shape * scale**2.
        """

        self._coeff_rand = fd.Constant(0.0)

        self._coeff_lower_bound = coeff_lower_bound

        self.coeff = self._coeff_lower_bound + self._coeff_rand
        """Spatially homogeneous, but random coefficient."""

        self._shape = shape
        
        self._scale = scale
        
        self.sample()

    def sample(self):
        """Samples the exponentially-distributed coefficient coeff."""

        self._coeff_rand.assign(np.random.gamma(self._shape,self._scale))

class UniformKLLikeCoeff(object):
    """A coefficient given by a KL-like expansion, with uniform R.V.s.

    The coefficient can be used in 2- or 3-D in a
    StochasticHelmholtzProblem.

    Initially the coefficient is set up so that the y_j (see init
    documentation) are all NaN. To set the values of y_j iteratively
    (iterating along a provided list of values - see init documentation)
    use the sample() method.

    Attributes:

    coeff - a UFL-implemented coeff, suitable for using in a
    helmholtz-firedrake StochasticHelmholtzProblem.

    stochastic_points - a numpy array of width J (see init
    documentation) containing the values of y_j (for many realisations
    of the vector y). Can be changed between calls to sample.

    Methods:

    sample - as in the requirements for StochasticHelmholtzProblem

    reinitialise - Restores the list of 'stochastic points' to the
    original list, and initialises the coefficients in series expansion
    to NaN.

    """

    def __init__(self,mesh,J,delta,lambda_mult,n_0,given_points):
        """Initialises the coefficient.

        The coefficient is of the form

        \[ n(y,x) = n_0 + \sum_{j=1}^J \sqrt{\lambda_j} y_j \psi_j,\]

        where the $\psi_j$ are given by 
        \[ \psi_j(x) = \cos(j\pi x[0]) \cos((j+1)\pi x[1])...
                           \cos((j+2)\pi x[2])\]
        (where the third term is neglected in 2-D),
        the $\sqrt{\lambda_j}$ are chosen so that 
        \[\sqrt{\lambda_j} = lambda_mult j^{-1-\delta}\]
        and the $y_j$ are in $[-1/2,1/2]$.

        Parameters:

        mesh - Firedrake Mesh - on which the coefficient will be
        defined.

        J - positive int - the number of terms in the expansion.

        delta - positive real - controls the convergence rate of the
        series (see above).

        n_0 - float or UFL expression on mesh - the mean of the
        expansion.

        stochastic_points - numpy ndarray of floats, of width J and some
        length. Each row gives the points y in 'stochastic space' at
        which the coefficient will be evaluated.

        """
        self._J = J
        
        self._stochastic_points_copy = deepcopy(given_points)

        self.reinitialise()

        self._mesh = mesh
        
        self._x = fd.SpatialCoordinate(self._mesh)

        # A bit of fiddling through this, because we want to start
        # indexing at 1
        
        self._psij = np.array([fd.cos(float(jj+1) * np.pi * self._x[0])
                               * fd.cos(float(jj+2) * np.pi * self._x[1])
                               for jj in range(self._J)])

        if mesh.geometric_dimension() == 3:
          for jj in range(self._J):
              self._psij[jj] = self._psij[jj]\
                               * fd.cos(float(jj+3) * np.pi * self._x[2])

        self._sqrt_lambda = np.array([lambda_mult * float(jj+1)**(-1.0-delta)
                                      for jj in range(self._J)])

        self.coeff = n_0

        for jj in range(self._J):
            self.coeff += self._sqrt_lambda[jj]\
                     * self._stochastic_points_constants[jj] * self._psij[jj]

    def sample(self):
        """Samples the coefficient, selects the next 'stochastic point'.

        If all the stochastic points have been sampled, returns a
        SamplingError.

        """
        if self.stochastic_points.shape[0] == 0:
            raise SamplingError

        else:
            for jj in range(self._J):
                self._stochastic_points_constants[jj].assign(
                    self.stochastic_points[0,jj])

            self.stochastic_points = self.stochastic_points[1:,:]

    def reinitialise(self):
        """Restores all stochastic points, and resets the Constants."""

        self.stochastic_points = deepcopy(self._stochastic_points_copy)

        # Update Constants if they already exist, create them if not.
        try:
            for jj in range(self._J):
                self._stochastic_points_constants[jj].assign(np.nan)
        except AttributeError:
            self._stochastic_points_constants = np.array([fd.Constant(np.nan)
                                                          for ii in range(self._J)])
                   
class SamplingError(Exception):
    """Error raised when all points have been sampled."""

    def __init__(self):
        print("All stochastic points have been sampled. Reinitalise the coefficient.")
        
