""" Test Default Block Sampling Functions """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np

# dimensions
n = 10
p = 2
step = 3

def test_random_sample():
    from sampling_funcs import random_sample as sample

    print('Testing random sampling...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_gauss_southwell():
    from sampling_funcs import gauss_southwell as sample

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
    from sampling_funcs import cyclic as sample

    print('Testing cyclic...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

def test_partition():
    from sampling_funcs import partition as sample

    print('Testing partition...')
    print(sample(n,p,init=True))
    print(sample(n,p))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p))
    print(sample(n,p,step=step))
    print(sample(n,p,step=step))

test_random_sample()
test_gauss_southwell()
test_cyclic()
test_partition()
