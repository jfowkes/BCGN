""" Test Default Block Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np

# dimensions
n = 10
p = 2
step = 3

def test_generic(sample):
    sample(n,p,init=True)

    S, scale = sample(n,p)
    print(S), print(1/scale**2)
    S, scale = sample(n,p)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,step=step)
    print(S), print(1/scale**2)
    S, scale = sample(n,p)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,step=step)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,step=step)
    print(S), print(1/scale**2)

def test_random_coordinate():
    from sampling_funcs import random_coordinate
    print('\nTesting random sampling...')
    test_generic(random_coordinate)

def test_cyclic_coordinate():
    from sampling_funcs import cyclic_coordinate

    print('\nTesting cyclic...')
    test_generic(cyclic_coordinate)

def test_partition_coordinate():
    from sampling_funcs import partition_coordinate

    print('\nTesting partition...')
    test_generic(partition_coordinate)

def test_gauss_southwell_coordinate():
    from sampling_funcs import gauss_southwell_coordinate as sample

    gradfx = np.arange(n)

    print('\nTesting Gauss-Southwell...')
    sample(n,p,gradfx,init=True)

    S, scale = sample(n,p,gradfx)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,gradfx)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,gradfx,step=step)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,gradfx)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,gradfx,step=step)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,gradfx,step=step)
    print(S), print(1/scale**2)

def test_random_gaussian():
    from sampling_funcs import random_gaussian

    print('\nTesting Gaussian...')
    test_generic(random_gaussian)

def test_random_hashing():
    from sampling_funcs import random_hashing as sample
    p = 5

    print('\nTesting hashing...')
    sample(n,p,init=True)

    print('s = 1')
    S, scale = sample(n,p,s=1)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,s=1)
    print(S), print(1/scale**2)

    print('s = 3')
    S, scale = sample(n,p,s=3)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,s=3)
    print(S), print(1/scale**2)

def test_random_hashing_variant():
    from sampling_funcs import random_hashing_variant as sample
    p = 5

    print('\nTesting hashing variant...')
    sample(n,p,init=True)

    print('s = 1')
    S, scale = sample(n,p,s=1)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,s=1)
    print(S), print(1/scale**2)

    print('s = 3')
    S, scale = sample(n,p,s=3)
    print(S), print(1/scale**2)
    S, scale = sample(n,p,s=3)
    print(S), print(1/scale**2)

# run tests
test_random_coordinate()
test_cyclic_coordinate()
test_partition_coordinate()
test_gauss_southwell_coordinate()
test_random_gaussian()
test_random_hashing()
test_random_hashing_variant()
