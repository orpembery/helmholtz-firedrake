import firedrake as fd
import numpy as np

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

    def _heaviside(self,x):
        """Defines the heaviside step function in UFL.

        Parameters:

        x - single coordinate of a UFL SpatialCoordinate.
        """
        return 0.5 * (fd.sign(fd.real(x)) + 1.0)

    def _Iab(self,x,a,b) :
        """Indicator function on [a,b] in UFL.

        Parameters:

        x - single coordinate of a UFL SpatialCoordinate.
        
        a, b - floats in [0,1].
        """
        return self._heaviside(x-a) - self._heaviside(x-b)

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
        
        self._coeff_values = []
        
        for ii in range(self._num_pieces**2):
            if self._coeff_dims == [2,2]:
                self._coeff_values.append(np.array([[0.0,0.0],[0.0,0.0]]))
            elif self._coeff_dims == [1]:
                self._coeff_values.append(np.array(0.0))
            else:
                raise NotImplementedError(
                          "Have only implemented real- and\
                          matrix-valued coefficients")
                
        self._coeff_values = [fd.Constant(coeff_dummy,domain=mesh)
                              for coeff_dummy in self._coeff_values]
        
        # Form coeff by looping over all the subdomains
        x = fd.SpatialCoordinate(mesh)

        self.coeff = coeff_pre
        
        for xii in range(self._num_pieces):
            for yii in range(self._num_pieces):
                self.coeff +=\
                self._list_extract(self._coeff_values,xii,yii,
                                   self._num_pieces)\
                * self._Iab(x[0],xii/self._num_pieces,(xii+1)/
                            self._num_pieces)\
                * self._Iab(x[1],yii/self._num_pieces,(yii+1)/self._num_pieces)

    def sample(self):
        """Samples the coefficient coeff."""
        [coeff_dummy.assign(self._generate_matrix_coeff())
         for coeff_dummy in self._coeff_values]

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

class SmoothNTCoeff(object):
    """Generates random, smooth, NT(mu) almost surely coefficients.

    For a definition of NT(mu), see [Graham, Pembery, Spence, Journal of
    Differential Equations (2018), to appear].
    """

    def __init__(self,mesh,x_centre,N,delta,
                 series_term_lower,series_term_upper,r_max):
        """Initialises the coefficient.

        The coefficient has the form (scalar case)

        n(x) = \sum_{j=0}^N a_{j} \abs{x-x_centre}^j
                            + a_{-1} (\abs{x-x_centre}+delta)^{-1}.

        The value of mu is >= a_{-1} * delta / (r_max + delta)**2, where
        r_max = max_{x \in D} \abs{x}.

        The terms a_j are uniformly distributed on
        [series_term_lower,series_term_upper]

        Parameter:

        mesh - a Firedrake Mesh - the mesh on which the problem is
        defined.

        x_centre - d-tuple of reals (where d is the dimension) - the
        coordinates of the 'centre' of the coefficient, outwards from
        which it is radially symmetric.

        N - int - the number of terms to include in the expansion.

        delta - the amount of shift to put in the lowest order term, to
        remove the singularity at the origin.

        series_term_lower - positive real - the lowest value that the
        terms a_j in the definition of n can take.

        series_term_upper - positive real series_term_upper >=
        series_term_lower - the largest value that the terms a_j in the
        definition of n can take.

        r_max - positive real - the maximum value of abs(x-x_centre)
        over the domain underlying mesh.
        """

        self._N = N

        self._delta = delta

        self._lower = series_term_lower

        self._upper = series_term_upper

        self._r_max = r_max

        self._coeff_values = []
        
        for ii in range(-1,self._N+1):
            self._coeff_values.append(np.array(0.0))
                
        self._coeff_values = [fd.Constant(coeff_dummy,domain=mesh)
                              for coeff_dummy in self._coeff_values]

        x = fd.SpatialCoordinate(mesh)

        r = fd.abs(x-x_centre)
        
        self.n = self._coeff_values[0] * (r + delta)**(-1)

        for j in range(0,self._N+1):
            self.n += self._coeff_values[j+1] * r**j

        self.sample()

    def sample(self):
        """Samples the coefficient."""

        for j in range(-1,self._N+1):
            self._coeff_values[j].assign(
                self._lower
                + (self._upper - self._lower) * np.random.random_sample()
            )

    def mu(self):
        """Returns (an estimate from below of) mu."""
        
        return self._coeff_values[0]\
            * self._delta / (self._r_max + self._delta)**2
        
class LInfinityMonotonicCoeff(object):
    """Generates piecewise-smooth coefficients that are monotonic.

    'Monotic' means that the coefficients are monotonically
    non-decreasing in the radial direction in the sense of [Graham,
    Pembery, Spence, Journal of Differential Equations (2018), to
    appear, Condition 2.6].
    """

    # This has been shelved until the ufl.And has been fixed in complex

    # Basic idea

    # Specify where the `origin' is

    # Choose n_min (possibly have options for randomly choosing in such a way that it is `nice' (in the sense of the hetero paper)

    # choose location of `base' interfaces (number of interfaces optional)

    # perturb the base interfaces in order (making sure they don't overlap)

    # on each `piece', define the sound speeds (define the (random, via series expansion with variable coeffs) growth in the radial direction, and then define the random (again via a truncated (variable truncation length?) Fourier expansion) deviations in the tangential direction, and take the product)

    # Put it all together

