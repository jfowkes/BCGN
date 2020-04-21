""" Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from RBCGN import RBCGN
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import warnings
import os
import sys
sys.path.append('../pycutest/')
import pycutest

""" Main function """
def main():

    # Main parameters
    RUNTYPE = 'metrics' # 'plot' - plot runs, 'metrics' - perf. profiles
    INSTANCES = 100 # no. random runs
    ALGORITHM = 'tr' # globalisation algorithm
    SUBPROB = 'normal' # subproblem solver
    SAMPLING = 'coordinate' # type of sampling
    kappas = [1] # 1 - block GN, (0,1) - adaptive GN
    bsizes = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1] # fraction of full block size
    ASTEP = 5 # adaptive BCGN step size

    # Runtype 'metrics'
    GRAD_EVALS = 50 # no. full gradient evaluations
    METRICS = [1e-1, 1e-3, 1e-5] # function decrease metrics

    # Runtype 'plot'
    PLOT_TYPE = 'all' # 'all' - plot all runs, 'avg' - plot run average
    IT_MAX = 100 # max iterations
    FTOL = 1e-10 # tolerance
    SAVEFIG = False # save plot figures

    # Test functions
    from problems.cutest32_zero import funcs, args, fxopts
    #funcs = ['BROWNALE', 'HYDCAR20', 'YATP1NE', 'YATP2SQ']
    #args = [{'N':100}, None, {'N':10}, {'N':10}]
    #fxopts = 4*[0]

    # Set up plotting / storage
    if RUNTYPE == 'plot': # set up plotting
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.lines import Line2D
        markers = ['o','v','^','<','>','s','p','P','H','D']
    else: # set up storage
        budgets = np.zeros((len(funcs)*INSTANCES,len(kappas)*len(bsizes),len(METRICS)))
        runtimes = np.zeros((len(funcs)*INSTANCES,len(kappas)*len(bsizes),len(METRICS)))
        row_labels = [func+' Run '+str(r+1) for func in funcs for r in range(INSTANCES)]
        basename = 'BCGN-'+ALGORITHM.upper()+'-'+SUBPROB.upper()+'-'+SAMPLING.upper()#+'-'+time.strftime('%d.%m.%Y-%H:%M:%S')
    column_labels = ['GN' if b==1 else str(b)+'n-'+('BCGN' if k==1 else str(k)+'A-BCGN') for k in kappas for b in bsizes]
    dimen = np.zeros(len(funcs))

    # Loop over test functions
    for ifunc, func in enumerate(funcs):
        print('====== '+func+' ======')

        # Get test function
        r, J, x0 = get_test_problem(func,args[ifunc],ALGORITHM)
        n = x0.size
        dimen[ifunc] = n
        fxopt = fxopts[ifunc]

        for ikappa, kappa in enumerate(kappas):
            print('\n====== Kappa: '+str(kappa)+' ======')

            # Set up plotting
            if RUNTYPE == 'plot':
                legend = []
                if PLOT_TYPE == 'avg': # average runs for plotting
                    fig = plt.figure(ifunc+1,figsize=(24,6))
                    ax1 = fig.add_subplot(1,3,1)
                    ax2 = fig.add_subplot(1,3,2)
                    ax3 = fig.add_subplot(1,3,3)
                else: # plot all runs
                    fig = plt.figure(ifunc+1,figsize=(24,6))
                    ax1 = fig.add_subplot(1,2,1)
                    ax2 = fig.add_subplot(1,2,2)
                    legend_lines = []

            # For each block size
            blocks = [int(n*b+0.5) for b in bsizes]
            for ip, p in enumerate(blocks):
                print('\n======',column_labels[ikappa*len(bsizes)+ip],'======')

                # Set up storage
                if RUNTYPE == 'plot':
                    legend += ['Block Size '+str(p)]
                    X = np.arange(IT_MAX+1)
                    Ys = np.zeros((3,IT_MAX+1,INSTANCES))
                else:
                    budget = np.zeros((INSTANCES,len(METRICS)))
                    runtime = np.zeros((INSTANCES,len(METRICS)))

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
                    if RUNTYPE == 'plot': # Plotting
                        Ys[:,:,iseed] = RBCGN(r,J,x0,p,sampling=SAMPLING,kappa=kappa,astep=ASTEP,fxopt=fxopt,
                                              it_max=IT_MAX,ftol=FTOL,runtype=RUNTYPE,algorithm=ALGORITHM,subproblem=SUBPROB)
                    else: # performance profiles
                        budget[iseed,:], runtime[iseed,:] = RBCGN(r,J,x0,p,sampling=SAMPLING,kappa=kappa,astep=ASTEP,fxopt=fxopt,
                                                                  grad_evals=GRAD_EVALS,metrics=METRICS,runtype=RUNTYPE,algorithm=ALGORITHM,subproblem=SUBPROB)
                        if p == n:  # GN: all runs are the same
                            budget = np.tile(budget[iseed,:],(INSTANCES,1))
                            runtime = np.tile(runtime[iseed,:],(INSTANCES,1))

                if RUNTYPE == 'plot': # plot each run
                    try: # truncate to last converged run
                        warnings.simplefilter("ignore", RuntimeWarning)
                        lrun = min(np.where((Ys[0,:,:] <= FTOL).all(axis=1))[0])+1
                        X = X[:lrun]; Ys = Ys[:,:lrun,:]
                        warnings.resetwarnings()
                    except ValueError:
                        pass
                    col = cm.tab10(ip/len(bsizes))
                    if PLOT_TYPE == 'avg': # average runs for plotting
                        ax1.semilogy(X,np.nanmean(Ys[0,:,:],axis=-1),color=col,linewidth=2)
                        ax1.fill_between(X,np.nanmin(Ys[0,:,:],axis=-1),np.nanmax(Ys[0,:,:],axis=-1),color=col,alpha=0.5)
                        ax2.semilogy(X,np.nanmean(Ys[1,:,:],axis=-1),color=col,linewidth=2)
                        ax2.fill_between(X,np.nanmin(Ys[1,:,:],axis=-1),np.nanmax(Ys[1,:,:],axis=-1),color=col,alpha=0.5)
                        ax3.plot(X,np.nanmean(Ys[2,:,:],axis=-1),color=col,linewidth=2)
                        ax3.fill_between(X,np.nanmin(Ys[2,:,:],axis=-1),np.nanmax(Ys[2,:,:],axis=-1),color=col,alpha=0.5)
                    else: # plot all runs
                        legend_lines += [Line2D([0],[0],color=col,linewidth=2)]
                        for iseed, seed in enumerate(seeds):
                            ax1.semilogy(X,Ys[0,:,iseed],color=col,marker=markers[iseed],markevery=10)
                            ax2.semilogy(X,Ys[1,:,iseed],color=col,marker=markers[iseed],markevery=10)
                else: # save performance profiles
                    budgets[ifunc*INSTANCES:(ifunc+1)*INSTANCES,ikappa*len(bsizes)+ip,:] = budget
                    runtimes[ifunc*INSTANCES:(ifunc+1)*INSTANCES,ikappa*len(bsizes)+ip,:] = runtime
                    for im, m in enumerate(METRICS):
                        dfb = pd.DataFrame(data=budgets[:,:,im],index=row_labels,columns=column_labels)
                        dfb.to_pickle(basename+'_'+'{:.0e}'.format(m)+'.budget')
                        dfr = pd.DataFrame(data=runtimes[:,:,im],index=row_labels,columns=column_labels)
                        dfr.to_pickle(basename+'_'+'{:.0e}'.format(m)+'.runtime')
                    dfd = pd.DataFrame(data=dimen[np.newaxis,:],index=['n'],columns=funcs)
                    dfd.to_pickle(basename+'.dimen')

            # Plotting
            if RUNTYPE == 'plot':
                title = 'BCGN - '+func+' ('+str(n)+'D)'
                plt.suptitle(str(kappa)+'A-'+title if kappa !=1 else title,fontsize=13)
                if PLOT_TYPE == 'avg': # average runs for plotting
                    xlimu = int(ax1.get_xlim()[1])
                    ax1.axhline(y=FTOL,xmin=0,xmax=xlimu,color='k',linestyle='--')
                    #ax2.semilogy(X[1:xlimu],1/X[1:xlimu],'k--')
                    ax1.set_xlabel('Iterations')
                    ax1.set_ylabel('Norm Residual')
                    ax1.grid(True)
                    ax2.set_xlabel('Iterations')
                    ax2.set_ylabel('Norm Gradient')
                    ax2.grid(True)
                    ax3.legend(legend, loc='upper right')
                    ax3.set_xlabel('Iterations')
                    ax3.set_ylabel('Block Size')
                    ax3.grid(True)
                else: # plot all runs
                    xlimu = int(ax1.get_xlim()[1])
                    ax1.axhline(y=FTOL,xmin=0,xmax=xlimu,color='k',linestyle='--')
                    ax1.legend(legend_lines,legend,loc='upper right')
                    ax1.set_xlabel('Iterations')
                    ax1.set_ylabel('Norm Residual')
                    ax1.grid(True)
                    ax2.set_xlabel('Iterations')
                    ax2.set_ylabel('Norm Gradient')
                    ax2.grid(True)
                plt.gcf().set_tight_layout(True)
                if SAVEFIG:
                    dir = 'results/'+ALGORITHM.upper()+'/'+str(kappa)
                    if not os.path.exists(dir): os.makedirs(dir)
                    alg = 'BCGN' if kappa == 1 else 'A-BCGN'
                    plt.savefig(dir+'/'+func+'_'+alg+'_'+str(INSTANCES)+'runs')
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
