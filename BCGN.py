""" Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from RBCGN import RBCGN
from scipy.sparse import csr_matrix
import numpy as np
import warnings
import pickle
import time
import os
import sys
sys.path.append('../pycutest/')
import pycutest

""" Main function """
def main():

    # Main parameters
    IT_MAX = 50 # Max iterations (plot=True) / full gradient evaluations (plot=False)
    NO_INSTANCES = 100 # No. random runs
    FTOL = 1e-10
    ALG = 'tr'

    # Plotting parameters
    PLOT = False
    SAVEFIG = False

    # Sampling function
    from sampling_funcs import random_sample as sampling_func

    # Test functions
    from problems.cutest32_zero import funcs, args, dimen, fxopts
    kappas = [1,0.7]
    #from problems.oscillatory import funcs, args, dimen, fxopts
    #kappas = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
    #kappas = [0.6,0.7,0.8,0.9]

    # Performance profile data
    if PLOT:
        import matplotlib.pyplot as plt
    else:
        metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']
        measures = np.full((len(funcs),len(kappas)+2,len(metrics),NO_INSTANCES),np.nan)
        basename = 'BCGN-'+ALG.upper()+'-'+time.strftime('%d.%m.%Y-%H:%M:%S')
        pickle.dump(funcs, open(basename+'.funcs', 'wb'), protocol=-1)

    # Loop over test functions
    for ifunc, func in enumerate(funcs):
        print('====== ' + func + ' ======')

        # Get test function
        r, J, x0 = get_test_problem(func, args[ifunc], ALG)
        n = x0.size
        fxopt = fxopts[ifunc]

        all_labels = []
        for ikappa, kappa in enumerate(kappas):
            print('\n====== Kappa: ' + str(kappa) + ' ======')

            # Plotting
            if PLOT:
                fig = plt.figure(ifunc+1,figsize=(24, 6))
                ax1 = fig.add_subplot(1,3,1)
                ax2 = fig.add_subplot(1,3,2)
                ax3 = fig.add_subplot(1,3,3)

            # Set block sizes
            legend = []
            if kappa == 1:
                #blocks = np.arange(1,n+1)
                #labels = [r'$' + str(p) + '$-BCGN' for p in range(1,n)]
                #labels += ['GN']
                ishift = 0
                blocks = [2,int(round(n/2)),n]
                labels = [r'$2$-BCGN',r'$\frac{n}{2}$-BCGN','GN']
            else:
                ishift = 2+ikappa
                blocks = [2]
                labels = [r'$2$-A-BCGN:'+str(kappa)]
            all_labels += labels

            # For each block size
            for ip, p in enumerate(blocks):
                legend += ['Block Size ' + str(p)]
                print('\n======', labels[ip], '======')

                # Plotting
                if PLOT:
                    X = np.arange(IT_MAX+1)
                    Ys = np.full((3,IT_MAX+1,NO_INSTANCES),np.nan)

                # Set RNG seeds
                if p == n:
                    seeds = [0] # No randomness for GN
                else:
                    seeds = np.linspace(0,1e3,NO_INSTANCES,dtype=int)
                for iseed, seed in enumerate(seeds):
                    print('Run:',iseed+1)
                    np.random.seed(seed) # Fix RNG seed

                    # Run RBCGN
                    if PLOT: # Plotting
                        Ys[:,:,iseed] = RBCGN(r,J,x0,sampling_func,fxopt,IT_MAX,FTOL,p,fig,kappa,algorithm=ALG)
                    else: # performance profiles
                        measures[ifunc,ishift+ip,:,iseed] = RBCGN(r,J,x0,sampling_func,fxopt,IT_MAX,FTOL,p,None,kappa,algorithm=ALG)

                # Plotting
                if PLOT:
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ax1.semilogy(X,np.nanmean(Ys[0,:,:],axis=-1),nonposy='clip',linewidth=2)
                    ax1.fill_between(X,np.nanmin(Ys[0,:,:],axis=-1),np.nanmax(Ys[0,:,:],axis=-1),alpha=0.5)
                    ax2.semilogy(X,np.nanmean(Ys[1,:,:],axis=-1),nonposy='clip',linewidth=2)
                    ax2.fill_between(X,np.nanmin(Ys[1,:,:],axis=-1),np.nanmax(Ys[1,:,:],axis=-1),alpha=0.5)
                    ax3.plot(X,np.nanmean(Ys[2,:,:],axis=-1),linewidth=2)
                    ax3.fill_between(X,np.nanmin(Ys[2,:,:],axis=-1),np.nanmax(Ys[2,:,:],axis=-1),alpha=0.5)
                    warnings.resetwarnings()
                else:
                    pickle.dump(np.nanmean(measures, axis=-1), open(basename+'.measure', 'wb'), protocol=-1)
                    pickle.dump(dimen, open(basename+'.dimen', 'wb'), protocol=-1)
                    pickle.dump(all_labels, open(basename+'.labels', 'wb'), protocol=-1)

            # Plotting
            if PLOT:
                xlimu = int(ax1.get_xlim()[1])
                ax1.axhline(y=FTOL,xmin=0,xmax=xlimu,color='k',linestyle='--')
                #ax2.semilogy(X[1:xlimu],1/X[1:xlimu],'k--')
                plt.suptitle('RBCGN - ' + func + ' function (' + str(n) + 'D)',fontsize=13)
                ax1.legend(legend)
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Norm Residual')
                ax1.grid(True)
                ax2.set_xlabel('Iterations')
                ax2.set_ylabel('Norm Gradient')
                ax2.grid(True)
                ax3.set_xlabel('Iterations')
                ax3.set_ylabel('Block Size')
                ax3.grid(True)
                plt.gcf().set_tight_layout(True)
                if SAVEFIG:
                    dir = 'results/'+ALG.upper()+'/'+str(kappa)
                    if not os.path.exists(dir): os.makedirs(dir)
                    alg = 'BCGN' if kappa == 1 else 'A-BCGN'
                    plt.savefig(dir+'/'+func+'_'+alg+'_'+str(NO_INSTANCES)+'runs')
                    plt.clf()
                else:
                    plt.show()

""" Test Problem Selector """
def get_test_problem(name, sifParams, algorithm):

    if name.isupper(): # CUTEst problem
        prob = pycutest.import_problem(name,sifParams=sifParams)
        def r(x): return prob.cons(x)
        if 'approx' in algorithm: # sparse Jacobian
            def J(x): return prob.scons(x,gradient=True)[1].tocsr()
        else: # dense Jacobian
            def J(x): return prob.cons(x,gradient=True)[1]
        x0 = prob.x0

    else: # More-Garbow-Hillstrom problem
        mod = __import__('MGH', fromlist=[name])
        prob = getattr(mod, name)()
        r = prob.r
        if 'approx' in algorithm: # sparse Jacobian
            def J(x): return csr_matrix(prob.jacobian(x))
        else: # dense Jacobian
            J = prob.jacobian
        x0 = prob.initial

    return r, J, x0

if __name__ == "__main__":
    main()
