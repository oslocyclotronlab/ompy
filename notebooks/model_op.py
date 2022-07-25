import numpy as np
import ompy as om
import theano.tensor as tt
import pymc3 as pm

def diagonal_resolution(matrix, resolution_Ex):
    """Detector resolution at the Ex=Eg diagonal

    Uses gaussian error propagations which assumes independence of
    resolutions along Ex and Eg axis.

    Args:
        matrix (Matrix): Matrix for which the sesoluton shall be calculated

    Returns:
        resolution at Ex = Eg.
    """
    def resolution_Eg(matrix):
        """Resolution along Eg axis for each Ex. Defaults in this class are for OSCAR.

        Args:
            matrix (Matrix): Matrix for which the sesoluton shall be calculated

        Returns:
            resolution
        """
        def fFWHM(E, p):
            return np.sqrt(p[0] + p[1] * E + p[2] * E**2)
        fwhm_pars = np.array([73.2087, 0.50824, 9.62481e-05])
        return fFWHM(matrix.Ex, fwhm_pars)
    
    dEx = matrix.Ex[1] - matrix.Ex[0]
    dEg = matrix.Eg[1] - matrix.Eg[0]
    assert dEx == dEg

    dE_resolution = np.sqrt(resolution_Ex**2
                            + resolution_Eg(matrix)**2)
    return dE_resolution




# Ops to calculate the FG matrix from the NLDs and Ts
class calculate_FG(tt.Op):
    
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)
    
    def __init__(self, matrix, std, E_nld):
        
        self.matrix = matrix.copy()
        self.std = std.copy()
        self.resolution = diagonal_resolution(matrix, 150.)
        self.E_nld = E_nld.copy(order='C')
        
        self.matrix.values, self.std.values = om.extractor.normalize(self.matrix, self.std)
        
        self.std.values = std.values.copy(order='C')
        self.matrix.Ex = self.matrix.Ex.copy(order='C')
        self.matrix.Eg = self.matrix.Eg.copy(order='C')
        self.matrix.values = self.matrix.values.copy(order='C')
        
        
    def perform(self, node, inputs, outputs):
        
        (x,) = inputs
        T = x[:self.matrix.Eg.size]
        nld = x[self.matrix.Eg.size:]
        
        fg_th = om.decomposition.nld_T_product(nld, T, self.resolution, self.E_nld,
                                               self.matrix.Eg, self.matrix.Ex)
        
        z = -0.5*np.array(om.decomposition.chisquare_diagonal(self.matrix.values, fg_th,
                                                              self.std.values, self.resolution,
                                                              self.matrix.Eg, self.matrix.Ex))
        outputs[0][0] = z
        print(z)
        #return outputs[0][0]

class FG_loglike:
    def __init__(self, matrix, std, E_nld):
        
        self.matrix = matrix.copy()
        self.std = std.copy()
        self.resolution = diagonal_resolution(matrix, 150.)
        self.E_nld = E_nld.copy(order='C')
        
        self.matrix.values, self.std.values = om.extractor.normalize(self.matrix, self.std)
        
        self.std.values = std.values.copy(order='C')
        self.matrix.Ex = self.matrix.Ex.copy(order='C')
        self.matrix.Eg = self.matrix.Eg.copy(order='C')
        self.matrix.values = self.matrix.values.copy(order='C')
    
    def __call__(self, x):
        T = x[:self.matrix.Eg.size]
        nld = x[self.matrix.Eg.size:]
        
        fg_th = om.decomposition.nld_T_product(nld, T, self.resolution, self.E_nld,
                                               self.matrix.Eg, self.matrix.Ex)
        
        return om.decomposition.chisquare_diagonal(self.matrix.values, fg_th,
                                                   self.std.values, self.resolution,
                                                   self.matrix.Eg, self.matrix.Ex)

class LogLike2(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

def loglike(theta, x, data, sigma):
    return -0.5*np.sum(theta)
        
# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood