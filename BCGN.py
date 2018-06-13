""" Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from RBCGN import RBCGN
import numpy as np
import warnings
import pickle
import time
import os
import sys
sys.path.append('./cutest/')
import cutestmgr

""" Main function """
def main():

    # Main parameters
    IT_MAX = 50 # Max iterations (plot=True) / full gradient evaluations (plot=False)
    NO_INSTANCES = 100 # No. random runs
    FTOL = 1e-10
    GS = False
    ALG = 'tr'

    # Plotting parameters
    PLOT = False
    SAVEFIG = False

    # Loop over test functions
    from problems.cutest32_zero import funcs, args, dimen, fxopts
    kappas = [1, 0.7]

    # Performance profile data
    if PLOT:
        import matplotlib.pyplot as plt
    else:
        metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']
        measures = np.full((len(funcs),len(kappas)+2,len(metrics),NO_INSTANCES),np.nan)
        basename = 'BCGN-'+ALG.upper()+'-'+time.strftime('%d.%m.%Y-%H:%M:%S')
        pickle.dump(funcs, open(basename+'.funcs', 'wb'), protocol=-1)

    #dimen = []
    for ifunc, func in enumerate(funcs):
        print('====== ' + func + ' ======')

        # Get test function
        r, J, x0 = get_test_problem(func, args[ifunc])
        n = x0.size
        #dimen += [n]
        fxopt = fxopts[ifunc]

        labels = []
        for ikappa, kappa in enumerate(kappas):
            print('\n====== Kappa: ' + str(kappa) + ' ======')

            # Plotting
            if PLOT:
                fig = plt.figure(ifunc+1,figsize=(24, 6))
                ax1 = fig.add_subplot(1,3,1)
                ax2 = fig.add_subplot(1,3,2)
                ax3 = fig.add_subplot(1,3,3)

            legend = []
            if kappa == 1:
                #blocks = np.arange(1,n+1)
                #labels += [r'$' + str(p) + '$-BCGN' for p in range(1,n)]
                #labels += ['GN']
                ishift = 0
                blocks = [2,int(round(n/2)),n]
                labels += [r'$2$-BCGN',r'$\frac{n}{2}$-BCGN','GN']
            else:
                ishift = 2+ikappa
                blocks = [2]
                labels += [r'$2$-A-BCGN:'+str(kappa)]
            for ip, p in enumerate(blocks):
                legend += ['Block Size ' + str(p)]
                print('\n======', labels[ishift+ip], '======')

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
                    print('Run:',iseed)
                    np.random.seed(seed) # Fix RNG seed

                    # Run RBCGN
                    if PLOT: # Plotting
                        Ys[:,:,iseed] = RBCGN(r,J,x0,fxopt,IT_MAX,FTOL,p,fig,kappa,algorithm=ALG,gaussSouthwell=GS)
                    else: # performance profiles
                        measures[ifunc,ishift+ip,:,iseed] = RBCGN(r,J,x0,fxopt,IT_MAX,FTOL,p,None,kappa,algorithm=ALG,gaussSouthwell=GS)

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
                    pickle.dump(labels, open(basename+'.labels', 'wb'), protocol=-1)

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
                    sfix = ''
                    if GS: sfix = '_GS'
                    dir = 'results/'+ALG.upper()+'/'+str(kappa)+sfix
                    if not os.path.exists(dir): os.makedirs(dir)
                    alg = 'BCGN' if kappa == 1 else 'A-BCGN'
                    plt.savefig(dir+'/'+func+'_'+alg+'_'+str(NO_INSTANCES)+'runs')
                    plt.clf()
                else:
                    plt.show()

""" Test Problem Selector """
def get_test_problem(name, sifParams):

    if name.isupper(): # CUTEst problem
        if not cutestmgr.isCached(name):
            cutestmgr.prepareProblem(name,sifParams=sifParams)
        prob = cutestmgr.importProblem(name)
        if sifParams != prob.getinfo()['sifparams']:
            raise RuntimeError('Cached parameters for '+name+' do not match, please recompile.')

        # Bugfix: ignore fixed variables
        lb = prob.getinfo()['bl']
        ub = prob.getinfo()['bu']
        idx_fixed = np.where(ub==lb)[0]
        idx_free = np.where(ub!=lb)[0]
        def pad(x):
            x_full = np.zeros(len(lb))
            x_full[idx_free] = x
            x_full[idx_fixed] = lb[idx_fixed]
            return x_full

        def r(x): return prob.cons(pad(x))
        def J(x): return prob.cons(pad(x),True)[1][:,idx_free]
        x0 = prob.getinfo()['x'][idx_free]

    else: # More-Garbow-Hillstrom problem
        mod = __import__('MGH', fromlist=[name])
        prob = getattr(mod, name)()
        r = prob.r
        J = prob.jacobian
        x0 = prob.initial

    return r, J, x0

if __name__ == "__main__":
    main()
