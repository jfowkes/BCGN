""" Plot Hessian sparsity structure """
from __future__ import division
from BCGN import get_test_problem
import numpy as np
import matplotlib.pyplot as plt

# CUTEst 32 test functions
funcs = ['ARGTRIG', 'ARTIF', 'BDVALUES', 'BRATU2D', 'BROWNALE', 'BROYDN3D', 'BROYDNBD', 'CBRATU2D', 'CHANDHEQ',
         'CHEMRCTA',
         'CHNRSBNE', 'DRCAVTY1', 'DRCAVTY3', 'EIGENA', 'EIGENB', 'FLOSP2TL', 'FLOSP2TM', 'HYDCAR20', 'INTEGREQ',
         'MOREBVNE',
         'MSQRTA', 'MSQRTB', 'OSCIGRNE', 'POWELLSE', 'SEMICN2U', 'SEMICON2', 'SPMSQRT', 'VARDIMNE', 'LUKSAN11',
         'LUKSAN21',
         'YATP1SQ', 'YATP2SQ']
args = [{'N': 100}, {'N': 100}, {'NDP': 102}, {'P': 10}, {'N': 100}, {'N': 100}, {'N': 100}, {'P': 7}, {'N': 100},
        {'N': 50},
        {'N': 50}, {'M': 10}, {'M': 10}, {'N': 10}, {'N': 10}, {'M': 2}, {'M': 2}, None, {'N': 100}, {'N': 100},
        {'P': 10}, {'P': 10}, {'N': 100}, {'N': 100}, {'N': 100, 'LN': 90}, {'N': 100, 'LN': 90}, {'M': 34},
        {'N': 100}, None, None,
        {'N': 10}, {'N': 10}]

for ifunc, func in enumerate(funcs):

    # Get test function
    r, J, x0 = get_test_problem(func, args[ifunc],'tr')

    # Plot sparsity
    plt.figure(ifunc)
    plt.spy(np.dot(J(x0).T,J(x0)))
    plt.ylabel('m')
    plt.xlabel('m')
    plt.title('Hessian Sparsity: '+func)
    plt.show()

