""" Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.sparse as sp

def identity_matrix(n,sparse):
    """
    Construct (sparse) identity matrix

    :param n: problem dimension
    :returns: (sparse) identity matrix
    """
    return sp.identity(n,format='csr') if sparse else np.eye(n)

def coordinate_matrix(n,S,sparse):
    """
    Construct (sparse) sketching matrix from given coordinates

    :param n: problem dimension
    :param S: given coordinates
    :returns: (sparse) sketching matrix
    """
    if not sparse:
        U_S = np.zeros((n,len(S)))
        U_S[S,range(len(S))] = np.ones(len(S))
    else:
        U_S = sp.csr_matrix((np.ones(len(S)),(S,range(len(S)))),shape=(n,len(S)))
    return U_S


def random_coordinate(n,p,init=False,step=False,sparse=False):
    """
    Sample coordinate block of size p at random from n

    :param n: problem dimension
    :param p: block size
    :param step: adaptively increase block size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S

    if init: # initialization
        S = None
        return

    if p == n: # return full block
        return identity_matrix(n,sparse), 1

    if step: # adaptively increase block size
        rem_inds = np.setdiff1d(np.arange(n),S)
        SA = np.random.choice(rem_inds,size=p,replace=False)
        S = np.append(S,SA)
    else: # fixed block size
        S = np.random.choice(np.arange(n),size=p,replace=False)

    U_S = coordinate_matrix(n,S,sparse)
    scale = 1./np.sqrt(n/len(S))
    return U_S, scale

def gauss_southwell_update_gradient(gradient):
    """
    Update gradient for Gauss-Southwell
    """
    global gradf

    gradf = gradient

def gauss_southwell_coordinate(n,p,init=False,step=False,sparse=False):
    """
    Return coordinate block of p largest gradient entries

    :param n: problem dimension
    :param p: block size
    :param step: adaptively increase block size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S, gradf

    if init: # initialization
        S = None
        gradf = None
        return

    if p == n: # return full block
        return identity_matrix(n,sparse), 1

    # Evaluate and sort full gradient
    sorted_nginds = np.argsort(np.fabs(gradf))[::-1]

    if step: # adaptively increase block size
        S = sorted_nginds[0:len(S)+p]
    else: # fixed block size
        S = sorted_nginds[0:p]

    U_S = coordinate_matrix(n,S,sparse)
    scale = 1./np.sqrt(n/len(S))
    return U_S, scale

def cyclic_coordinate(n,p,init=False,step=False,sparse=False):
    """
    Cycle through contiguous coordinate blocks of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize cyclic state
    :param step: adaptively increase block size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S, cyclic_state

    # initialize cyclic state
    if init:
        S = None
        cyclic_state = 0
        return

    if p == n: # return full block
        return identity_matrix(n,sparse), 1

    # sample coordinates
    i = cyclic_state % n
    SA = np.arange(n)[i:i+p]

    # wrap around if required
    if len(SA) < p:
        SW = np.arange(n)[:p-len(SA)]
        SA = np.append(SA,SW)

    if step: # adaptively increase block size
        S = np.append(S,SA)
    else: # fixed block size
        S = SA

    # update state
    cyclic_state += p

    U_S = coordinate_matrix(n,S,sparse)
    scale = 1./np.sqrt(n/len(S))
    return U_S, scale

# FIXME: step size p is ignored at the moment, use nearest partition size?
def partition_coordinate(n,p,init=False,step=False,sparse=False):
    """
    Randomly sample uniform partition of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize partition
    :param step: adaptively increase block size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S, block_part

    # initialize block partition
    if init:
        S = None
        block_part = np.random.permutation(np.arange(n))
        return

    if p == n: # return full block
        return identity_matrix(n,sparse), 1

    if step: # adaptively increase block size
        rem_part = block_part[~np.in1d(block_part,S)]
        block_ind = np.random.choice(np.arange(0,len(rem_part),p))
        SA = rem_part[block_ind:block_ind+p]
        S = np.append(S,SA)
    else: # fixed block size
        block_ind = np.random.choice(np.arange(0,n,p))
        S = block_part[block_ind:block_ind+p]

    U_S = coordinate_matrix(n,S,sparse)
    scale = 1./np.sqrt(n/len(S))
    return U_S, scale

def random_gaussian(n,p,init=False,step=False,sparse=False):
    """
    Sample Gaussian basis of size p at random

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S

    if init: # initialization
        S = None
        return

    if p == n: # return identity
        return identity_matrix(n,sparse), 1

    if step: # adaptively increase basis size
        S = np.hstack((S,np.random.randn(n,p)))
    else: # fixed basis size
        S = np.random.randn(n,p)

    scale = 1./np.sqrt(S.shape[1])
    return sp.csr_matrix(S) if sparse else S, scale

def random_hashing(n,p,s=3,init=False,step=False,sparse=False):
    """
    Sample s-hashing matrix of size p at random

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: sketching matrix, sketching matrix scaling
    """

    if init: # no initialization required
        return

    if p == n: # return identity
        return identity_matrix(n,sparse), 1

    if step:
       raise RuntimeError('Cannot grow hashing matrices!')

    # sample without replacement
    indices = np.zeros(s*n)
    for j in range(n):
        indices[s*j:s*j+s] = np.random.permutation(p)[:s]
    indptr = s*np.arange(n+1)
    data = np.random.binomial(1,0.5,size=s*n)*2-1
    S = sp.csr_matrix((data,indices,indptr),shape=(n,p))
    if not sparse:
       S = S.toarray()

    scale = 1./np.sqrt(s)
    return S, scale

def random_hashing_variant(n,p,s=3,init=False,step=False,sparse=False):
    """
    Sample variant s-hashing matrix of size p at random

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: sketching matrix, sketching matrix scaling
    """

    if init: # no initialization required
        return

    if p == n: # return identity
        return identity_matrix(n,sparse), 1

    if step:
        raise RuntimeError('Cannot grow hashing matrices!')

    # sample with replacement
    indices = np.random.randint(p,size=s*n)
    indptr = s*np.arange(n+1)
    data = np.random.binomial(1,0.5,size=s*n)*2-1
    S = sp.csr_matrix((data,indices,indptr),shape=(n,p))
    if not sparse:
       S = S.toarray()

    scale = 1./np.sqrt(s)
    return S, scale


""" Thompson Sampling """
class Bandit_d_Thompson_sampling:
    def __init__(self, prior_mean, prior_lambda, tau, gamma):
        self.N = 0 # s_hat
        self.gamma = gamma
        self.prior_mean = prior_mean
        self.prior_lambda = prior_lambda
        self.posterior_mean = prior_mean
        self.posterior_lambda = prior_lambda
        self.summ = 0 # S
        self.tau = tau # by assumption fixed and we give it a value

    def pull_sample_mean(self):
        sigma = 1 / np.sqrt(self.posterior_lambda)
        mean = self.posterior_mean
        # return np.random.random()/np.sqrt(self.posterior_lambda)+self.posterior_mean
        return np.random.normal(mean, sigma) # this is the sampled mean from our known distribution

    def update(self, x, used=True):
        tau = self.tau
        gamma = self.gamma
        # the mean is now itself a normally distribuited random variable with
        # mean: posterior_mean and inverse variance lambda_posterior
        # if self.summ!=0:
        #   delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)/self.summ
        # else:
        #     delta=((gamma-1)*self.N*self.prior_lambda*self.prior_mean+gamma*self.N*self.summ*tau+self.prior_lambda*self.summ)/(self.N*tau+self.prior_lambda)
        delta = gamma
        if used:
            self.N = self.N * gamma + 1
            self.summ = self.summ * delta + x # mean=(1-1/self.N)*self.mean+1/self.N*x
        else:
            self.N = self.N * gamma
            self.summ = self.summ * delta
        self.posterior_lambda = self.prior_lambda + tau * self.N  #
        self.posterior_mean = (self.summ * tau + self.prior_lambda * self.prior_mean) / self.posterior_lambda

def thompson_coordinate(n,p=2,init=False,step=False,sparse=False):
    """
    Thompson sampling

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: sketching matrix, sketching matrix scaling
    """
    global S, coordinate_bandits

    if p == n: # return full block
        return identity_matrix(n,sparse), 1

    if step:
        raise RuntimeError('Cannot grow Thompson sampling matrices!')

    critical_measure_value = 0.5  # we do not take coordinates with measure below this value
    # can make it time dependent i.e. iteration dependent later if needed.

    # pick all coordinates who's measures are above
    prior_mean = 17  # note that prior mean should be slightly above actual values and slightly
    # above critical_measure_value to encourage exploring
    prior_lambda = 0.1
    tau = 1
    gamma = 0.75 # 0.78
    if init: # initialize
        S = None
        coordinate_bandits = np.zeros(n,dtype=object)
        for i in range(n): # use discounted thompson sampling and call the objects bandits for fun
            coordinate_bandits[i] = Bandit_d_Thompson_sampling(prior_mean, prior_lambda, tau, gamma)
            # 1st bandit coresponds to first coordinate and so on
        return

    # else if we do not initialize-here we just sample,
    # a different function will be there for update
    # Measure, delta_x, No_current_coords,
    SS = []
    for i in range(n):
        if critical_measure_value < coordinate_bandits[i].pull_sample_mean(): # if the value I'm sampling is above the critical mean,
            SS.append(i)                                                      # include this coordinate
    if len(SS) == 1:
        rem_inds = np.setdiff1d(np.arange(n),SS)
        SA = np.random.choice(rem_inds, size=1, replace=False)
        SS = np.append(SS,SA) # and we now have 2 coords
    elif len(SS) == 0:
        SS = np.random.choice(np.arange(n), size=2, replace=False)
    S = np.array(SS)

    U_S = coordinate_matrix(n,S,sparse)
    scale = 1./np.sqrt(n/len(S))
    return U_S, scale

def thompson_update_gradient(gradient):
    """
     Update gradient for Thompson sampling
    """
    global coordinate_bandits
    n = len(coordinate_bandits)

    if S is None: # nothing to update
        return

    # note that this can be any other measure but we keep the same name here
    gradient=abs(gradient)

    # update used coordinates
    for coordinate in S:
        coordinate_bandits[coordinate].update(gradient[coordinate],used=True)

    # update unused coordinates
    rem_inds = np.setdiff1d(np.arange(n),S)
    for coordinate in rem_inds:
        coordinate_bandits[coordinate].update(0,used=False)
