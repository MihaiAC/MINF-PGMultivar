import numpy as np

class TPGM():
    '''
    Class for generating samples from and fitting TPGM distributions.

    Attributes:
        N     : dimensionality of our random variables;
        theta : N x N matrix; theta[s][t] = weight of the edge between nodes (dimensions) s and t.
        R     : maximum count (truncation value).
    '''


    def __init__(self, theta = None, data = None, R = 200):
        '''
        Constructor for TPGM class.

        Parameters:
        theta: N x N matrix, given when we want to generate samples from an existing distribution.
        data: M x N matrix, one row is equivalent to one sample of the distribution to be fitted.
        R: maximum count value, should be an integer.
        '''
        self.R = R
        if(theta != None):
            self.N = len(theta[0])
            self.theta = np.array(theta)

        if(data != None):
            self.data = np.array(data)
            self.data[self.data >= self.R] = self.R
            self.N = len(data[0])

    def generate_samples(self, init, nr_samples=50, burn_in=100, thinning_nr=50):
        '''
        Gibbs sampler for our TPGM distribution.

        Parameters:
            init       : starting point for the sampling process;
            nr_samples : number of samples we want to generate.
            burn_in    : number of initial samples to be discarded.
            thinning_nr: we select every thinning_nr samples after the burn-in period;

        Returns (nr_samples X N) matrix containing one sample per row.
        '''

        assert self.theta != None, "The parameters of the distribution haven't been initialized."
        assert self.R != None and self.R > 0, "The truncation value R hasn't been initialized correctly."
        assert init != None and len(init) == self.N, "Incorrect dimensions for init. Correct dimensions are 1 x self.N."

        






