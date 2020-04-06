""" Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np

""" Global Variables """
S = None # sampled coordinates
cyclic_state = 0 # cyclic state
block_part = None # block partition

def random_coordinate(n,p,init=False,step=False):
    """
    Sample coordinate block of size p at random from n

    :param n: problem dimension
    :param p: block size
    :param step: adaptively increase block size
    :returns: relevant sketching matrix
    """
    global S

    if init: # no initialization required
        return

    if p == n: # return full block
        return np.eye(n)

    if step: # adaptively increase block size
        rem_inds = np.setdiff1d(np.arange(n),S)
        SA = np.random.choice(rem_inds,size=p,replace=False)
        S = np.hstack((S,SA))
    else: # fixed block size
        S = np.random.choice(np.arange(n),size=p,replace=False)

    U_S = np.zeros((n,len(S)))
    U_S[S,range(len(S))] = np.ones(len(S))
    return U_S

def gauss_southwell_coordinate(n,p,gradfx,init=False,step=False):
    """
    Return coordinate block of p largest gradient entries

    :param n: problem dimension
    :param p: block size
    :param gradfx: gradient of objective at x
    :param step: adaptively increase block size
    :returns: relevant sketching matrix
    """
    global S

    if init: # no initialization required
        return

    if p == n: # return full block
        return np.eye(n)

    # Evaluate and sort full gradient
    sorted_nginds = np.argsort(np.fabs(gradfx))[::-1]

    if step: # adaptively increase block size
        S = sorted_nginds[0:len(S)+p]
    else: # fixed block size
        S = sorted_nginds[0:p]

    U_S = np.zeros((n,len(S)))
    U_S[S,range(len(S))] = np.ones(len(S))
    return U_S

def cyclic_coordinate(n,p,init=False,step=False):
    """
    Cycle through contiguous coordinate blocks of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize cyclic state
    :param step: adaptively increase block size
    :returns: relevant sketching matrix
    """
    global S, cyclic_state

    # initialize by resetting state
    if init:
        cyclic_state = 0
        return

    if p == n: # return full block
        return np.eye(n)

    # sample coordinates
    i = cyclic_state % n
    SA = np.arange(n)[i:i+p]

    # wrap around if required
    if len(SA) < p:
        SW = np.arange(n)[:p-len(SA)]
        SA = np.hstack((SA,SW))

    if step: # adaptively increase block size
        S = np.hstack((S,SA))
    else:
        S = SA

    # update state
    cyclic_state += p

    U_S = np.zeros((n,len(S)))
    U_S[S,range(len(S))] = np.ones(len(S))
    return U_S

# FIXME: step size p is ignored at the moment, use nearest partition size?
def partition_coordinate(n,p,init=False,step=False):
    """
    Randomly sample uniform partition of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize partition
    :param step: adaptively increase block size
    :returns: relevant sketching matrix
    """
    global S, block_part

    # initialize block partition
    if init:
        block_part = np.random.permutation(np.arange(n))
        return

    if p == n: # return full block
        return np.eye(n)

    if step: # adaptively increase block size
        rem_part = block_part[~np.in1d(block_part,S)]
        block_ind = np.random.choice(np.arange(0,len(rem_part),p))
        SA = rem_part[block_ind:block_ind+p]
        S = np.hstack((S,SA))
    else: # fixed block size
        block_ind = np.random.choice(np.arange(0,n,p))
        S = block_part[block_ind:block_ind+p]

    U_S = np.zeros((n,len(S)))
    U_S[S,range(len(S))] = np.ones(len(S))
    return U_S

def random_gaussian(n,p,init=False,step=False):
    """
    Sample Gaussian basis of size p at random

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: relevant sketching matrix
    """
    global S

    if init: # no initialization required
        return

    if p == n: # return identity
        return np.eye(n)

    if step: # adaptively increase basis size
        k = S.shape[1]
        S = np.hstack((S*np.sqrt(k),np.random.randn(n,step)))/np.sqrt(k+step)
    else: # fixed basis size
        S = np.random.randn(n,p)/np.sqrt(p)

    return S

def random_hashing(n,p,s=3,init=False,step=False):
    """
    Sample hashing matrix of size p at random

    :param n: problem dimension
    :param p: basis size
    :param step: adaptively increase basis size
    :returns: relevant sketching matrix
    """

    if init: # no initialization required
        return

    if p == n: # return identity
        return np.eye(n)

    if step: # adaptively increase basis size
       raise RuntimeError('Cannot grow hashing matrices!')
    else: # fixed basis size
        S = np.zeros((n,p))
        for j in range(n):
            samp = np.random.choice(np.arange(p),size=s,replace=False)
            S[j,samp] = np.ones(s)/np.sqrt(s)

    return S