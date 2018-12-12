import firedrake as fd
import numpy as np
import helmholtz_firedrake.utils as utils
import pandas as pd
from scipy import optimise


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
        
class LognormalL1ExponentalCovarianceCoeff(object):
    """A lognormal random field whose log has an exponential covariance.

    The underlying Gaussian random field has a seperable exponential
    covariance, and is generated by truncating its Karhunen-Loeve
    expansion. This object does the work of n_coeff in a
    StochasticHelmholtzProblem.

 Attribute:

    coeff - a UFL expression for the coefficient implemented using
    Firedrake Constants.

    Method:

    sample - randomly updates coeff by sampling from the KL expansion.

    """

    def __init__(self,mesh,mean,sigma,l,J):
        """Initialises a lognormal RF with exponential covariance.

        Parameters:

        mesh - a Firedrake mesh object.

        mean - a UFL expression representing the mean of the logarithm.
        #It's possible that you'd need to pass in the correct SpatialCoordinate, but let's wait and see

        sigma - positive float, where sigma**2 is the variance of the
        random field.

        l - d-tuple of positive floats, where d is the geometric
        dimension of mesh. The correlation length in each direction.

        J - the number of terms in the truncated Karhunen-Loeve
        expansion.
        """
        # sigma does't come in here and I don't know why.
        
        self._x = fd.SpatialCoordinate(mesh)
        
        self._d = mesh.geometric_dimension()

        assert self._d == len(l)

        self._J = J

        # The implementation is based on the fact that we have explicit,
        # if not analytic, expressions for all the quantities we need to
        # construct the KL expansion - see Lord, Powell, Shardlow; An
        # Introduction to Computational Stochastic PDEs; CUP 2014,
        # Examples 7.55 and 7.56, pp. 295-296.

        # Add more explanation?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self._construct_1d_eigenvals()

        self._construct_d_eigenvals()

        self._construct_eigenfunctions()
        
        # set up structure for storing normals
        self._normals_list = [fd.Constant(0.0) for ii in range(self._J)]
        
        # Assemble KL, and take exponential
        self.coeff = mean

        for ii in range(J):
            self.coeff += np.sqrt(self._d_diml_eigenvalues_df.iloc[ii,0]) * self._normals_list[ii] * self._eigenfunctions_df,iloc[ii,0]

        self.coeff = fd.exp(self.coeff)


    def sample(self):
        """Sample the random coefficients."""

        [term.assign(np.rand.randn()) for term in self._normals_list]



        

    def _construct_1d_eigenvals(self):
        """Finds the first J 1-d eigenvalues.

        Parameters:

        self._J - positive int.

        Output:

        self._1d_eigenvals_df - pandas DataFrame of the first J eigenvalues in
        decending order. First column of the dataframe is the
        eigenvalue, second column is whether it is an 'odd' or 'even'
        eigenvalue. For more details, see Lord, Powell, Shardlow.
        """

        # Find first J roots of f_odd and f_even using windowing
        # method
        self._1d_eigenvals_df = pd.DataFrame(columns=['eigenval','fn_type'])
        roots_list_1d = []

        pi2 = np.pi/2.0

        for fn_type in ('odd','even'):
            if fn_type == 'odd':
                def f(omega):
                    return 1.0/l - omega * np.tan(omega)

            elif fn_type == 'even':
                def f(omega):
                    return np.tan(omega)/l + omega

            for jj in range(self._J):

                if fn_type == 'odd':
                    if jj == 0:
                        lower_bound = 0
                        upper_bound = pi2
                    else:
                        lower_bound = (2.0*float(jj) - 1.0)*pi2
                        upper_bound = (2.0*float(jj) + 1.0)*pi2

                elif fn_type == 'even':
                    # Something wrong here???????????????????????????????????????????????????????
                    lower_bound = (2.0*float(jj) - 1.0)*pi2
                    upper_bound = (2.0*float(jj) + 1.0)*pi2

                output = optimize.brentq(f,[lower_bound,upper_bound])
                roots_list_1d.append((output[0],fn_type))

        self._eigenvals_1d_df.append(roots_list_1d)
                    
        # Turn these roots into 1-d eigenvalues, remembering oddness
        # or evenness.
        def omega_to_eigenval(omega):
            return (2.0 * l**(-1.0)) / (omega**2.0 + l**(-2.0))

        self._eigenvals_1d_df.loc['eigenval'].applymap(omega_to_eigenval)

        # Sort into descending order
        self._eigenvals_1d_df.sort_values('eigenval',axis=1,ascending=False)

        self._eigenvals_1d_df = self._1d_eigenvals.iloc[:self._J,]

    
    def _construct_d_eigenvals(self):
        """Given a DataFrame of 1-d eigenvals, generate d-diml eigenvals.

        d-dimensional eigenvalues are d-wise products of 1-d eigenvalues.

        Output:

        self._d_diml_eigenvals_df - pandas DataFrame of the first J d-dimensional
        eigenvalues in descending order, where each row corresponds to an
        eigenvalue. The first column is the eigenvalue, the second is a
        d-tuple of the 1-d eigenvalues in each direction, and the third
        is a d-tuple of strings saying if the eigenvalue in that
        direction is 'odd' or 'even'.
        """
       
        def indices_to_eigenval(indicies):
            return self._eigenvals_1d_df.loc[indices,'eigenval'].prod()

        index_to_accept = tuple(np.zeros(self._d))

        def list_element(index_to_accept):
            return (indices_to_eigenval(index_to_accept),index_to_accept,self._eigenvals_1d_df.iloc[list(index_to_accept),1])
    
        d_eigenvals_list = [list_element(index_to_accept)]
        
        for jj in range(1,self._J):
            value_to_accept = 0.0
            for ii in range(self._d):
                # This is a hack to get the iith 'unit vector'
                candidate_index = d_eigenvals_list[-1] + tuple([int(kk==ii) for kk in range(self._d)])
                candidate_value = indices_to_eigenval(candidate_index)
                
                if candidate_value > value_to_accept:
                    value_to_accept = candidate_value
                    index_to_accept = candidate_index
                    
                    d_eigenvals_list.append(list_element(index_to_accept))

                self._d_eigenvals_df = pd.DataFrame(columns=['eigenval','1d_eigenvals','fn_types'])

        self._d_eigenvals_df.append(d_eigenvals_list)

    def _construct_eigenfunctions(self):
        """Calculates the corresponding eigenfunctions, and stores them.

        Parameters:

        self._d_eigenvals_df - as outputted from self._construct_d_eigenvals

        Output:
        
        self._eigenfunctions_df - pandas DataFrame of length self._J
        with one column, containing the UFL expressions for the
        eigenfunctions.
        """
        # No idea if this will work!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        self._eigenfunctions_df = self._d_eigenvals_df.apply(self._eigenfun,axis=1)

        self._eigenfunctions_df.columns = ['eigenfunction']

    def _eigenfun(self,efun_row):
        """The eigenfunction for row index in self._eigenfunctions."""

        efun = 1.0

        for ii in range(self._d):
            # THIS NEEDS CHECKING THAT ITS THE RIGHT WAY ROUND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if efun_row[2][ii] == 'even':
                efun *= fd.cos(efun_row[1][ii]*self._x[ii])
            elif efun_row[2][ii] == 'odd':
                efun *= fd.sin(efun_row[1][ii]*self._x[ii])

        return efun

        
