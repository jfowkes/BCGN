""" Block-Coordinate Gauss-Newton (Individual Functions) """
from __future__ import absolute_import, division, unicode_literals, print_function
from RBCGN_IND import RBCGN_IND
import numpy as np
import pycutest
import sys

""" Main function """
def main():

    # Main parameters
    RUNTYPE = sys.argv[1] # 'plot' - plot runs, 'metrics' - plot timings
    INSTANCES = 5  # no. random runs
    IT_MAX = 20  # max iterations
    TAU = 1e-1  # objective decrease

    # Algorithm settings
    ALGORITHM = 'tr_approx' # globalisation algorithm
    SUBPROB = 'normal' # subproblem solver
    SAMPLING = sys.argv[2] # type of sampling
    kappas = [1] # 1 - block GN, (0,1) - adaptive GN
    bsizes = [0.01,0.05,0.1,0.5,1]
    ASTEP = 5 # adaptive BCGN step size

    # Test functions
    funcs = ['OSCIGRNE','ARTIF','BRATU2D']
    args = [{'N':10000},{'N':5000},{'P':72}]
    fxopts = [0,0,0]

    # Set up plotting / storage
    column_labels = ['Full-Block' if b==1 else str(b)+'n-'+('BCGN' if k==1 else str(k)+'A-BCGN') for k in kappas for b in bsizes]

    # Loop over test functions
    for ifunc, func in enumerate(funcs):
        print('====== '+func+' ======')

        # Get test function
        if func.isupper(): # CUTEst
            r, J, x0 = get_test_problem(func,args[ifunc],ALGORITHM)
        else: # logistic regression
            from problems.logistic_regression import logistic_regression
            r, J, x0 = logistic_regression(func)
        n = x0.size
        fxopt = fxopts[ifunc]

        for ikappa, kappa in enumerate(kappas):
            print('\n====== Kappa: '+str(kappa)+' ======')

            # For each block size
            blocks = [int(n*b+0.5) for b in bsizes]
            for ip, p in enumerate(blocks):
                print('\n======',column_labels[ikappa*len(bsizes)+ip],'======')
                if(p==0):
                    raise ValueError('Zero block size selected!')

                # Set up storage
                data = np.zeros((IT_MAX+1,INSTANCES))

                # Set RNG seeds
                if p == n:
                    seeds = [0] # No randomness for GN
                else:
                    seeds = np.linspace(0,1e3,INSTANCES,dtype=int)

                # For each random instance
                for iseed, seed in enumerate(seeds):
                    np.random.seed(seed) # Fix RNG seed
                    print('Run: '+str(iseed+1))

                    # Run RBCGN
                    data[:,iseed] = RBCGN_IND(r,J,x0,p,sampling=SAMPLING,kappa=kappa,astep=ASTEP,fxopt=fxopt,
                                              it_max=IT_MAX,tau=TAU,runtype=RUNTYPE,algorithm=ALGORITHM,subproblem=SUBPROB)

                if RUNTYPE == 'plot': # save plotdata
                    np.save(func+'_'+SAMPLING+'_'+str(p)+'_plotdata',data)
                else: # save runtimes
                    np.save(func+'_'+SAMPLING+'_'+str(p)+'_runtimes',data)


""" Test Problem Selector """
def get_test_problem(name, sifParams, algorithm):

    prob = pycutest.import_problem(name,sifParams=sifParams)
    def r(x): return prob.cons(x)
    if 'approx' in algorithm: # sparse Jacobian
        def J(x): return prob.scons(x,gradient=True)[1].tocsr()
    else: # dense Jacobian
        def J(x): return prob.cons(x,gradient=True)[1]
    x0 = prob.x0

    return r, J, x0

if __name__ == "__main__":
    main()
