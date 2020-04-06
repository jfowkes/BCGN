""" Test Default Block Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np

# dimensions
n = 10
p = 2
step = 3

def test_random_sample():
    from sampling_funcs import random_coordinate as sample

    print('Testing random sampling...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_gauss_southwell():
    from sampling_funcs import gauss_southwell_coordinate as sample

    gradfx = np.random.random(n)

    print('Testing Gauss-Southwell...')
    print(sample(n,p,gradfx,init=True))
    print(sample(n,p,gradfx))
    print(sample(n,p,gradfx))
    print(sample(n,p,gradfx,step=step))
    print(sample(n,p,gradfx))
    print(sample(n,p,gradfx,step=step))
    print(sample(n,p,gradfx,step=step))

def test_cyclic():
    from sampling_funcs import cyclic_coordinate as sample

    print('Testing cyclic...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_partition():
    from sampling_funcs import partition_coordinate as sample

    print('Testing partition...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_gaussian():
    from sampling_funcs import random_gaussian as sample

    print('Testing Gaussian...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_hashing():
    from sampling_funcs import random_hashing as sample
    p = 5

    print('Testing hashing...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))

test_random_sample()
test_gauss_southwell()
test_cyclic()
test_partition()
test_gaussian()
test_hashing()
