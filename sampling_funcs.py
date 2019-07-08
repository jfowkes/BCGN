""" Default Block Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np

""" Global Variables """
S = None # sampled coordinates
cyclic_state = 0 # cyclic state
block_part = None # block partition

def random_sample(n,p,init=False,step=False):
    """
    Sample coordinate block of size p at random from n

    :param n: problem dimension
    :param p: block size
    :param step: adaptively increase block size
    :returns: array of sampled coordinate indices
    """
    global S

    if init: # no initialization required
        return

    if p == n: # return full block
        return np.arange(n)

    if step: # adaptively increase block size - updates already existing S
        #by adding an extra p coordinates from the remaining ones
        rem_inds = np.setdiff1d(np.arange(n),S) #obtain remaining indices vector
        SA = np.random.choice(rem_inds,size=p,replace=False)#choose at random
        #p coords from the remaining set
        S = np.hstack((S,SA))#stack the two blocks
    else: # fixed block size
        S = np.random.choice(np.arange(n),size=p,replace=False)
        #S is a vector of random coordinate indices
    return S

# FIXME: initialize with sorted(gradfx) as global
def gauss_southwell(n,p,gradfx,init=False,step=False):
    """
    Return coordinate block of p largest gradient entries

    :param n: problem dimension
    :param p: block size
    :param gradfx: gradient of objective at x
    :param step: adaptively increase block size
    :returns: array of sampled coordinate indices
    """
    global S

    if init: # no initialization required
        return

    if p == n: # return full block
        return np.arange(n)

    # Evaluate and sort full gradient
    sorted_nginds = np.argsort(np.fabs(gradfx))[::-1]

    if step: # adaptively increase block size
        S = sorted_nginds[0:len(S)+p]
    else: # fixed block size
        S = sorted_nginds[0:p]

    return S

def greedy_gauss_southwell(n,p,gradfx,init=False,step=False):
    """
    Return coordinate block largest gradient entries such that the subspace norm
    is 99.7% of the full norm
"""
    global S
    
    if init: # no initialization required
        return
    
    full_space_2norm=np.linalg.norm(gradfx,ord=2)
    if p == n: # return full block
        return np.arange(n)

    # Evaluate and sort full gradient
    sorted_nginds = np.argsort(np.fabs(gradfx))[::-1]

    if step: # adaptively increase block size
        S = sorted_nginds[0:len(S)+p]
    else: # fixed block size
        S=np.array([sorted_nginds[0],sorted_nginds[1]])
        subspace_2norm=np.linalg.norm(gradfx[S],ord=2)
        zeta=0.997
        while subspace_2norm<zeta*full_space_2norm:
            i=len(S)
            S=np.append(S,sorted_nginds[i])
            subspace_2norm=np.linalg.norm(gradfx[S],ord=2)
    return S

def adaptive_greedy_gauss_southwell(n,p,gradfx,steplength,zeta,init=False,step=False):
    """
    Return coordinate block largest gradient entries such that the subspace norm
    is zeta|%% of the full norm, where zeta adapts based on the tolerance 
    to ensure convergence at the end
"""
    global S
    
    if init: # no initialization required
        return
    
    full_space_2norm=np.linalg.norm(gradfx,ord=2)
    if p == n: # return full block
        return np.arange(n)

    # Evaluate and sort full gradient
    sorted_nginds = np.argsort(np.fabs(gradfx))[::-1]

    if step: # adaptively increase block size
        S = sorted_nginds[0:len(S)+p]
    else: # fixed block size
        #selecting zeta-----
      #  if steplength>=0.5 or np.isnan(steplength):
       #     zzeta=0.997
       # elif steplength<0.5 and steplength>=0.1:
        #    zzeta=0.9997
       # elif steplength<0.1 and steplength>=0.01:
       #     zzeta=0.99997
       # elif steplength<0.01 and steplength>=0.005:
        #    zzeta=0.999997
       # elif steplength<0.005 and steplength>=0.0007:
         #   zzeta=0.9999997
        #elif steplength<7e-04:
        #    zzeta=0.999999997
        #zeta=max(zzeta,zeta)
        #-----end selecting zeta
        zeta=0.99997
        S=np.array([sorted_nginds[0],sorted_nginds[1]])
        subspace_2norm=np.linalg.norm(gradfx[S],ord=2)
        while subspace_2norm<zeta*full_space_2norm:
            i=len(S)
            S=np.append(S,sorted_nginds[i])
            subspace_2norm=np.linalg.norm(gradfx[S],ord=2)
    return S, zeta


def cyclic(n,p,init=False,step=False):
    """
    Cycle through contiguous coordinate blocks of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize cyclic state
    :param step: adaptively increase block size
    :returns: array of sampled coordinate indices
    """
    global S, cyclic_state

    # initialize by resetting state
    if init:
        cyclic_state = 0
        return

    if p == n: # return full block
        return np.arange(n)

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

    return S

# FIXME: step size p is ignored at the moment, use nearest partition size?
def partition(n,p,init=False,step=False):
    """
    Randomly sample uniform partition of size p

    :param n: problem dimension
    :param p: block size
    :param init: initialize partition
    :param step: adaptively increase block size
    :returns: array of sampled coordinate indices
    """
    global S, block_part

    # initialize block partition
    if init:
        block_part = np.random.permutation(np.arange(n))
        return

    if p == n: # return full block
        return np.arange(n)

    if step: # adaptively increase block size
        rem_part = block_part[~np.in1d(block_part,S)]
        block_ind = np.random.choice(np.arange(0,len(rem_part),p))
        SA = rem_part[block_ind:block_ind+p]
        S = np.hstack((S,SA))
    else: # fixed block size
        block_ind = np.random.choice(np.arange(0,n,p))
        S = block_part[block_ind:block_ind+p]

    return S

def random_sample_save_info(n,p,init=False,step=0,SS=-1):
    """
    Sample coordinate block of size p at random from n

    :param n: problem dimension
    :param p: block size
    :param step: adaptively increase block size
    :returns: array of sampled coordinate indices
    """
    global S
    if type(SS)!=type(None):
        if type(SS)!=int:
            S=SS

    if init: # no initialization required
        return

    if p == n: # return full block
        if step==0:
            budget_increment=n
            return np.arange(n), budget_increment
        else:
            return np.arange(n), 0
    
    if step!=0: # adaptively increase block size - updates already existing S
        if type(S)!=type(None):
            if len(S)==n: #if we got to the full GN block size
                budget_increment=0
                return np.arange(n), budget_increment
        #by adding an extra p coordinates from the remaining ones
        rem_inds = np.setdiff1d(np.arange(n),S) #obtain remaining indices vector
        if step>n-len(S):
            SA = np.random.choice(rem_inds,size=n-len(S),replace=False)#choose at random
            budget_increment=n-len(S)
        else:
            SA = np.random.choice(rem_inds,size=step,replace=False)
            budget_increment=step
        #p coords from the remaining set
        S = np.hstack((S,SA))#stack the two blocks
    else: # fixed block size
        S = np.random.choice(np.arange(n),size=p,replace=False)
        budget_increment=p
        #S is a vector of random coordinate indices
    return S, budget_increment

        
        
