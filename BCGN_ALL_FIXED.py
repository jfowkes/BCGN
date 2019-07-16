""" Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from RBCGN_DFO_Update import RBCGN_DFO_Update
from RBCGN_Zhen_scaling_drop_and_add import RBCGN_Zhen_scaling_drop_and_add
from RBCGN import RBCGN
from RBCGN_scaling_TR import RBCGN_scaling_TR
from RBCGN_variable_rho import RBCGN_variable_rho
from scipy.sparse import csr_matrix
import numpy as np
import warnings
import pickle
import time
import os
import sys
sys.path.append('../pycutest/')
os.environ['PYCUTEST_CACHE'] = "/home/constantin/Desktop/pycutest_cache/"
import pycutest
#funcs = ['ARGTRIG','ARTIF','BDVALUES','BRATU2D','BROWNALE','BROYDN3D','
        # BROYDNBD','CBRATU2D','CHANDHEQ','CHEMRCTA',
       #  'CHNRSBNE','DRCAVTY1','DRCAVTY3','EIGENA','EIGENB','FLOSP2TL',
       # 'FLOSP2TM','HYDCAR20','INTEGREQ','MOREBVNE',
       #  'MSQRTA','MSQRTB','OSCIGRNE','POWELLSE','SEMICN2U','SEMICON2','
       # SPMSQRT','VARDIMNE','LUKSAN11','LUKSAN21',
       #  'YATP1SQ','YATP2SQ']

""" Main function """
def main():
    #pycharm
    # Main parameters----------------------------------------------------------
    IT_MAX = 50 # Max iterations (plot=True) / full gradient evaluations (plot=False)
    NO_INSTANCES = 5# No. random runs
    FTOL = 1e-10
    ALG = 'tr'
    #-------------------------------------------------------------------------
    # Plotting parameters-----------------------------------------------------
    PLOT = False
    SAVEFIG = False
    #--------------------------------------------------------------------------

    # Sampling function---------------------------------------------------------
    from sampling_funcs import random_sample as sampling_func
    from sampling_funcs import random_sample_save_info as sampling_func2
    # contains multiple sampling strategies - see file sampling_func.py
    #-------------------------------------------------------------------------
    
    # Test functions----------------------------------------------------------
    from problems.cutest32_zero import funcs, args, dimen, fxopts
#!!!!!!!!#KEEP JUST THE PROBLEM YOU WANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    problems_you_want=(1,4,10,20,22,27,30)
    funcs=[funcs[i] for i in problems_you_want]
    args=[args[i] for i in problems_you_want]
    dimen=[dimen[i] for i in problems_you_want]
    fxopts=[fxopts[i] for i in problems_you_want]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    kappas = [1,0.7]  #kappa=1 <=> NONADAPTIVE, FIXED
    #kappa!=1 <=> ADAPTIVE starting from size p, as in the nonadaptive case.
    #-------------------------------------------------------------------------
    
    #from problems.oscillatory import funcs, args, dimen, fxopts
    #kappas = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
    #kappas = [0.6,0.7,0.8,0.9]

    # Performance profile data------------------------------------------------
    if PLOT:
        import matplotlib.pyplot as plt
        
    else:
        metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']
        #measures initialization !
        measures = np.full((len(funcs),len(kappas)+2,len(metrics),NO_INSTANCES),np.nan)
        basename = 'BCGN-'+ALG.upper()+'-'+time.strftime('%d.%m.%Y-%H:%M:%S')
        #pickle.dump(funcs, open(basename+'.funcs', 'wb'), protocol=-1)
        filehandler=open(basename+'.funcs', 'wb')
        pickle.dump(funcs, filehandler, protocol=-1)
        filehandler.close()
    #--------------------------------------------------------------------------
    
    # Loop over test functions*******************************************
    for ifunc, func in enumerate(funcs):
        print('====== ' + func + ' ======')

        # Get test function---------------------------------------------
        r, J, x0 = get_test_problem(func, args[ifunc], ALG)
        n = x0.size
        fxopt = fxopts[ifunc] #fxopt is the value of f at the global minimum
        #---------------------------------------------------------------
        all_labels = []
        
        #for all kappas=================================================
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
            if kappa == 1: #for nonadaptive -do 2, n/2 and n coords i.e. fullGN
                #blocks = np.arange(1,n+1)
                #labels = [r'$' + str(p) + '$-BCGN' for p in range(1,n)]
                #labels += ['GN']
                ishift = 0
                blocks = [2,int(round(n/2)),n]
                labels = [r'$2$-BCGN',r'$\frac{n}{2}$-BCGN','GN']
            else: #for adaptive
                ishift = 2+ikappa
                blocks = [2]
                labels = [r'$2$-A-BCGN:'+str(kappa)]
            all_labels += labels

            # For each block size-----------------------------------------
            for ip, p in enumerate(blocks):
                legend += ['Block Size ' + str(p)]
                print('\n======', labels[ip], '======')

                # Plotting
                if PLOT:
                    X = np.arange(IT_MAX+1)
                    Ys = np.full((3,IT_MAX+1,NO_INSTANCES),np.nan)

                # Set RNG seeds - iseeds ARE SIMPLY THE NUMBER OF RUNS LOL
                if p == n:
                    seeds = [0] # No randomness for GN
                else:
                    seeds = np.linspace(0,1e3,NO_INSTANCES,dtype=int)
                for iseed, seed in enumerate(seeds):
                    print('Run:',iseed+1)
                    np.random.seed(seed) # Fix RNG seed

                    # Run RBCGN
                    if PLOT: # Plotting
                        Ys[:,:,iseed] = RBCGN_scaling_TR(r,J,x0,sampling_func,fxopt,IT_MAX,FTOL,p,fig,kappa,func,iseed,algorithm=ALG)
                    else: # performance profiles - THIS IS WHERE RBCGN is called
                        measures[ifunc,ishift+ip,:,iseed] = RBCGN_scaling_TR(r,J,x0,sampling_func,fxopt,IT_MAX,FTOL,p,None,kappa,func,iseed,algorithm=ALG)
                        #RBCGN returns nan if not solved or no of coord eval if solved
                # Plotting
                if PLOT:
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ax1.semilogy(X,np.nanmean(Ys[0,:,:],axis=-1),linewidth=2)
                    ax1.fill_between(X,np.nanmin(Ys[0,:,:],axis=-1),np.nanmax(Ys[0,:,:],axis=-1),alpha=0.5)
                    ax2.semilogy(X,np.nanmean(Ys[1,:,:],axis=-1),linewidth=2)
                    ax2.fill_between(X,np.nanmin(Ys[1,:,:],axis=-1),np.nanmax(Ys[1,:,:],axis=-1),alpha=0.5)
                    ax3.plot(X,np.nanmean(Ys[2,:,:],axis=-1),linewidth=2)
                    ax3.fill_between(X,np.nanmin(Ys[2,:,:],axis=-1),np.nanmax(Ys[2,:,:],axis=-1),alpha=0.5)
                    warnings.resetwarnings()
                    #nonposy='clip' has been removed  -seems that there is a bug with
                    #mathplot lib and cannot use this attribute anymore
                else:
                    filehandler = open(basename+'.measure', 'wb')
                    pickle.dump(measures, filehandler, protocol=-1)
#!!!                #take the average wrt the last axis
                    #it seems you only average out the SUCCESSFUL RUNS
                    #but that somehow makes the solver look better as you'd think
                    #that's the average over all runs and it always solves but it's actually
                    #the average over when it manages to solve....
                    filehandler.close()
                    
                    filehandler = open(basename+'.dimen', 'wb')
                    pickle.dump(dimen, filehandler, protocol=-1)
                    filehandler.close()
                    
                    filehandler=open(basename+'.labels', 'wb')
                    pickle.dump(all_labels, filehandler, protocol=-1)
                    filehandler.close()                    
                    #pickle.dump(np.nanmean(measures, axis=-1), open(basename+'.measure', 'wb'), protocol=-1)
                    #pickle.dump(dimen, open(basename+'.dimen', 'wb'), protocol=-1)
                    #pickle.dump(all_labels, open(basename+'.labels', 'wb'), protocol=-1)
            #----------------------------------------------------------------
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
            #=================================================================
    #************************************************************************
""" Test Problem Selector """
def get_test_problem(name, sifParams, algorithm):

    if name.isupper(): # CUTEst problem
        prob = pycutest.import_problem(name,sifParams=sifParams)
        def r(x): return prob.cons(x)
        if 'tr_approx' in algorithm: # sparse Jacobian
            def J(x): return prob.scons(x,gradient=True)[1].tocsr()
        else: # dense Jacobian
            def J(x): return prob.cons(x,gradient=True)[1]
        x0 = prob.x0

    else: # More-Garbow-Hillstrom problem
        mod = __import__('MGH', fromlist=[name])
        prob = getattr(mod, name)()
        r = prob.r
        if 'tr_approx' in algorithm: # sparse Jacobian
            def J(x): return csr_matrix(prob.jacobian(x))
        else: # dense Jacobian
            J = prob.jacobian
        x0 = prob.initial

    return r, J, x0

if __name__ == "__main__":
    main()
