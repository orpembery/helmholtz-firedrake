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

        with np.nditer(self._constant_array,flags=['refs_ok'],op_flags=['writeonly']) as array_it:
            
            for const in array_it:

                if self._coeff_dims == [2,2]:
                    const[...] = fd.Constant(np.array([[0.0,0.0],[0.0,0.0]]),domain=mesh)
                elif self._coeff_dims == [1]:
                    const[...] = fd.Constant(0.0,domain=mesh)
                
        # Form coeff by looping over all the subdomains
        x = fd.SpatialCoordinate(mesh)

        self.coeff = coeff_pre

        array_it = np.nditer(self._constant_array,flags=['refs_ok','multi_index'])

        while not array_it.finished:
        
            const =  array_it[0]
            
            loc_array = np.array((array_it.multi_index,1+np.array(array_it.multi_index))
                                 ,dtype='float').T/float(self._num_pieces)
            
            self.coeff += utils.nd_indicator(x,const,loc_array)

            array_it.iternext()

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


class ExponentialConstantCoeffGenerator(object):
    """Does the work of n_stoch in StochasticHelmholtzProblem for the
    case of a constant but 1+exponential-distributed refractive index.

    Attribute:

    coeff - a Firedrake Constant containing the value of the
    scalar-valued coefficients.

    Method:

    sample - randomly updates coeff by randomly sampling from coeff_base
    + gamma(rate).
    """

    def __init__(self,scale):
        """Initialises a constant, 1+exponential-distributed constant.

        Parameters:
        scale - the scale of the exponential distribution

        The mean of the exponential distribution is scale and the
        variance is scale**2.
        """

        self._coeff_rand = fd.Constant(0.0)

        self.coeff = 1.0 + self._coeff_rand
        """Spatially homogeneous, but random coefficient."""
        
        self._scale = scale
        
        self.sample()

    def sample(self):
        """Samples the exponentially-distributed coefficient coeff."""

        self._coeff_rand.assign(np.random.exponential(self._scale))

class UniformKLLikeCoeff(object):
    """A coefficient given by a KL-like expansion, with uniform R.V.s.

    The coefficient can be used in 2- or 3-D in a
    StochasticHelmholtzProblem.

    Initially the coefficient is set up so that the y_j are the first
    row of given_points (see __init__ documentation). To set the values
    of y_j iteratively (iterating along a provided list of values - see
    init documentation) use the sample() method.

    Attributes:

    coeff - a UFL-implemented coeff, suitable for using in a
    helmholtz-firedrake StochasticHelmholtzProblem.

    Methods:

    sample - as in the requirements for StochasticHelmholtzProblem.

    current_point - gives the 'stochastic coefficients' y_j
    corresponding to the current realisation of the coefficient.

    unsampled_points - gives the 'stochastic coefficients' y_j
    corresponding to all the realisations of the coefficient that have
    not yet been sampled.

    current_and_unsampled_points - gives the 'stochastic coefficients'
    y_j corresponding to both the current realisation of the coefficient
    and all the realisations of the coefficient that have not yet been
    sampled.

    reorder - allows the user to reorder the points that have not yet
    been sampled, so that they are sampled in a different order.

    reinitialise - Resets the coefficient, as if it has just been
    reinitialised.

    change_all_points - allows the user to completely change the
    'stochastic coefficients' y_j.

    """

    def __init__(self,mesh,J,delta,lambda_mult,j_scaling,n_0,stochastic_points):
        """Initialises the coefficient.

        The coefficient is of the form

        \[ n(y,x) = n_0 + \sum_{j=1}^J \sqrt{\lambda_j} y_j \psi_j,\]

        where the $\psi_j$ are given by 
        \[ \psi_j(x) = \cos(j\pi/j_scaling x[0]) \cos((j+1)\pi/j_scaling x[1])...
                           \cos((j+2)\pi/j_scaling x[2])\]
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

        lambda_mult - positive float - controls the absolute value of
        the series (see above).

        j_scaling - positive float - scales the oscillations in the
        psi_j (see above).

        n_0 - float or UFL expression on mesh - the mean of the
        expansion.

        stochastic_points - A numpy ndarray of floats, of width J and some
        length. Each row gives the points y in 'stochastic space' at
        which the coefficient will be evaluated.

        """
        assert J == stochastic_points.shape[1]
        
        self._J = J
        
        self._update_stochastic_points_copy(stochastic_points)
        
        # self._stochastic_points and self._stochastic_points_constants
        # are defined in here
        self.reinitialise()

        self._mesh = mesh
        
        self._x = fd.SpatialCoordinate(self._mesh)

        # A bit of fiddling through this, because we want to start
        # indexing at 1 in the sum.
        
        self._psij = np.array([fd.cos(float(jj+1) * np.pi * self._x[0] / j_scaling)
                               * fd.cos(float(jj+2) * np.pi * self._x[1] / j_scaling)
                               for jj in range(self._J)])

        if mesh.geometric_dimension() == 3:
          for jj in range(self._J):
              self._psij[jj] = self._psij[jj]\
                               * fd.cos(float(jj+3) * np.pi * self._x[2] / j_scaling)

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
        if self._stochastic_points.shape[0] == 0:
            raise SamplingError

        else:
            self._update_current_point_stochastic_points()
            self._constants_assign()

    def current_point(self):
        """Grabs the current point."""
        return self._current_point

    def unsampled_points(self):
        return self._stochastic_points

    def current_and_unsampled_points(self):
        return np.vstack((self.current_point(),self.unsampled_points()))

    def reorder(self,indices,include_current_point=False):
        """Reorders the 'stochastic points'.

        Parameters: indices - a numpy array containing the first L integers (including zero) in some order (i.e. the output from np.argsort).

        If include_current_points = True, then the values given by
        self.current_and_unsampled_points() will be sorted according to
        the ordering given by indices, and the stochastic point at the
        top of the resulting array will be set as the current point. In
        this case L should be the number of rows in the output of
        self.current_and_unsampled_points().

        If include_current_points = False, then only the values given by
        self._unsampled_points() will be sorted, and the value of
        self._current_point() will remain unchanged. In this case L
        should be the number of rows in the output of
        self.unsampled_points().
        """
        if include_current_point:
            points_to_reorder = self.current_and_unsampled_points()
        else:
            points_to_reorder = self.unsampled_points()
            
        self._stochastic_points = points_to_reorder[indices,:]

        if include_current_point:
            self._update_current_point_stochastic_points()

            self._constants_assign()        

    def reinitialise(self):
        """Restores all stochastic points, and resets the Constants.

        Note: This will not take into account the use of the reorder()
        method - it will reset the stochastic points to the values they
        had after the last call to init or change_all_points().
        """

        self._stochastic_points = deepcopy(self._stochastic_points_copy)

        self._update_current_point_stochastic_points()

        # Update Constants if they already exist, create them if not.
        try:
            self._constants_assign()
        except AttributeError:
            self._stochastic_points_constants = np.array(
                [fd.Constant(self._current_point[jj])
                 for jj in range(self._J)])

    def _constants_assign(self):
        """Assigns self._current_point to the Constants."""
        for jj in range(self._J):
            self._stochastic_points_constants[jj].assign(
                self._current_point[jj])

    def change_all_points(self,stochastic_points):
        """Completely change all the stochastic points.

        Like calling init with a different stochastic_points.
        """
        
        self._update_stochastic_points_copy(stochastic_points)

        self.reinitialise()

    def _update_current_point_stochastic_points(self):

        self._current_point = self._stochastic_points[0,:]

        self._stochastic_points = self._stochastic_points[1:,:]

    def _update_stochastic_points_copy(self,stochastic_points):
        self._stochastic_points_copy = np.array(deepcopy(stochastic_points),ndmin=2)

    
                   
class SamplingError(Exception):
    """Error raised when all points have been sampled."""

    def __init__(self):
        pass
        
