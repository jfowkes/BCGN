""" Sketching Gauss-Newton (Individual Functions) """
from __future__ import absolute_import, division, unicode_literals, print_function
from RSGN import RSGN
import numpy as np
import pycutest

""" Main function """
def main():

    # Main parameters
    RUNTYPE = 'metrics' # 'plot' - plot runs, 'metrics' - plot timings
    INSTANCES = 5  # no. random runs
    IT_MAX = 100  # max iterations
    TAU = 1e-3  # objective decrease

    # Algorithm settings
    ALGORITHM = 'tr_approx' # globalisation algorithm
    SUBPROB = 'normal' # subproblem solver
    SAMPLING = 'coordinate' # type of sampling
    kappas = [1] # 1 - block GN, (0,1) - adaptive GN
    bsizes = [1.2] # sketch size multiplier (p = b*n)
    ASTEP = 5 # adaptive SGN step size

    # Test functions
    #funcs = ['chemotherapy',gisette']
    funcs = ['DMN15103','SPMSQRT','ARWHDNE','FREURONE']
    args = [None,{'M':1667},{'N':5000},{'N':5000}]
    #fxopts = [0,0,0]

    # Set up plotting / storage
    column_labels = [str(b)+'d-'+('SGN' if k==1 else str(k)+'A-SGN') for k in kappas for b in bsizes]#+['GN']

    # Loop over test functions
    for ifunc, func in enumerate(funcs):
        print('====== '+func+' ======')

        # Get test function
        if func.isupper(): # CUTEst
            r, J, x0, m = get_test_problem(func,args[ifunc],ALGORITHM)
        else: # logistic regression
            from problems.logistic_regression import logistic_regression
            r, J, x0 = logistic_regression(func)
        n = x0.size

        for ikappa, kappa in enumerate(kappas):
            print('\n====== Kappa: '+str(kappa)+' ======')

            # For each block size
            blocks = [int(n*b+0.5) for b in bsizes]#+[m]
            for ip, p in enumerate(blocks):
                print('\n======',column_labels[ikappa*len(bsizes)+ip],'======')
                print('Sketch size: '+str(p))
                if(p==0):
                    raise ValueError('Zero sketch size selected!')

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
                    data[:,iseed] = RSGN(r,J,x0,p,m,sampling=SAMPLING,kappa=kappa,astep=ASTEP,
                                         it_max=IT_MAX,tau=TAU,runtype=RUNTYPE,algorithm=ALGORITHM,subproblem=SUBPROB)

                if RUNTYPE == 'plot': # save plotdata
                    np.save(func+'_'+str(p)+'_'+SAMPLING.upper()+'_plotdata',data)
                else: # save runtimes
                    np.save(func+'_'+str(p)+'_'+SAMPLING.upper()+'_runtimes',data)


""" Test Problem Selector """
def get_test_problem(name, sifParams, algorithm):

    prob = pycutest.import_problem(name,sifParams=sifParams)
    def r(x): return prob.cons(x)
    if 'approx' in algorithm: # sparse Jacobian
        def J(x,inds=None): 
            if inds is None or len(inds) == len(x): # full GN
                return prob.scons(x,gradient=True)[1].tocsr()
            else: # sketch in m
                return prob.scons(x,gradient=True)[1].tocsr()[inds,:]
    else: # dense Jacobian
        def J(x): return prob.cons(x,gradient=True)[1]
    x0 = prob.x0
    m = prob.m

    return r, J, x0, m

if __name__ == "__main__":
    main()
