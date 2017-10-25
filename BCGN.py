""" Block-Coordinate Gauss-Newton """
from __future__ import division
from numba import jit
import numpy as np
import scipy.linalg as linalg
import math as ma
import warnings
import pickle
import cutermgr
import time
import os

""" Main function """
def main():

    # Main parameters
    IT_MAX = 50 # Max iterations (plot=True) / full gradient evaluations (plot=False)
    NO_INSTANCES = 100 # No. random runs
    FTOL = 1e-10
    GS = False
    ALG = 'tr_approx'

    # Plotting parameters
    PLOT = False
    SAVEFIG = False

    # Loop over test functions
    kappas = [1, 0.6, 0.7, 0.8, 0.9]
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
    fxopts = np.zeros(32)

    # Performance profile data
    if PLOT:
        import matplotlib.pyplot as plt
    else:
        metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']
        measures = np.full((len(funcs),len(kappas)+2,len(metrics),NO_INSTANCES),np.nan)
        basename = 'BCGN-'+ALG.upper()+'-'+time.strftime('%d.%m.%Y-%H:%M:%S')
        pickle.dump(funcs, open(basename+'.funcs', 'wb'), protocol=-1)

    dimen = []
    for ifunc, func in enumerate(funcs):
        print '====== ' + func + ' ======'

        # Get test function
        r, J, x0 = get_test_problem(func, args[ifunc])
        n = x0.size
        dimen += [n]
        fxopt = fxopts[ifunc]

        labels = []
        for ikappa, kappa in enumerate(kappas):
            print '\n====== Kappa: ' + str(kappa) + ' ======'

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
                print '\n======', labels[ishift+ip], '======'

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
                    print 'Run:',iseed
                    np.random.seed(seed) # Fix RNG seed

                    #if not (func == 'OSCIPANE' and kappa == 0.7 and iseed in [40]):
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
                    dir = 'figures/'+ALG.upper()+'/'+str(kappa)+sfix
                    if not os.path.exists(dir): os.makedirs(dir)
                    alg = 'BCGN' if kappa == 1 else 'A-BCGN'
                    plt.savefig(dir+'/'+func+'_'+alg+'_'+str(NO_INSTANCES)+'runs')
                    plt.clf()
                else:
                    plt.show()

""" Random Block-Coordinate Gauss-Newton """
def RBCGN(r, J, x0, fxopt, it_max, ftol, p, fig, kappa, algorithm='tr', partitionBlock=False, gaussSouthwell=False):

    # Adaptive BCGN step size
    STEP = 5

    # Full function and gradient
    def f(z): return 0.5 * np.dot(r(z), r(z))
    def gradf(z): return J(z).T.dot(r(z))

    # Plotting
    if fig is not None:
        plot_data = np.full((3,it_max+1),np.nan)
        plot_data[0,0] = f(x0)-fxopt
        plot_data[1,0] = linalg.norm(gradf(x0))

    # Metrics
    budget = 0
    tau_budget = np.full(4,np.nan)

    # Set initial trust region radius
    delta = None
    if algorithm.startswith('tr') or algorithm == 'reg':
        delta = linalg.norm(gradf(x0))/10

    # Initialize block partition
    if partitionBlock:
        block_part = np.random.permutation(np.arange(x0.size))

    k = 0
    x = x0
    n = x.size
    while (not fig and budget < it_max*n) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):

        # Evaluate full gradient for Gauss-Southwell
        if gaussSouthwell:
            sorted_nginds = np.argsort(np.fabs(gradf(x)))[::-1]

        # Randomly select blocks
        if gaussSouthwell:
            S = sorted_nginds[0:p]
        elif partitionBlock:
            block_ind = np.random.choice(np.arange(0,n,p))
            S = block_part[block_ind:block_ind+p]
        else:
            S = np.random.permutation(np.arange(n))[0:p]
        U_S = np.zeros((n,len(S)))
        for j in range(0,len(S)):
            U_S[S[j],j] = 1

        # Assemble block-reduced matrices
        Jx = J(x)
        J_S = Jx.dot(U_S)
        rx = r(x)
        gradf_S = J_S.T.dot(rx)

        # Output
        #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

        # Solve subproblem
        if algorithm == 'tr':
            s_S = trs(J_S, gradf_S, delta)
        elif algorithm == 'tr_approx':
            s_S = trs_approx(J_S, gradf_S, delta)
        elif algorithm == 'reg':
            s_S, delta = reg(J_S, gradf_S, delta)
        else:
            s_S, delta = line_search(f, x, U_S, J_S, gradf_S)

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*linalg.norm(rx)**2 > 0
        #Jx_S = J_S.dot(x.dot(U_S))
        #stopping_rule = -Delta_m + np.dot(Js_S,Jx_S) + (sigma/2)*linalg.norm(s_S)**2 > 0

        # Iteratively refine block size
        p_in = p
        while p_in != n and stopping_rule:

            # Increase block size
            step = min(STEP,n-p_in)
            #print 'Increasing block size to:', p_in+step
            if gaussSouthwell:
                inds = sorted_nginds[p_in:p_in+step]
            else:
                S = np.nonzero(U_S)[0]
                rem_inds = np.setdiff1d(np.arange(n),S)
                inds = np.random.choice(rem_inds,step,replace=False)
            U_inds = np.zeros((n,step))
            for j in range(0,len(inds)):
                U_inds[inds[j],j] = 1
            U_S = np.hstack((U_S,U_inds))
            J_S = Jx.dot(U_S)
            gradf_S = J_S.T.dot(rx)
            p_in += step

            # Output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            elif algorithm == 'tr_approx':
                s_S = trs_approx(J_S, gradf_S, delta)
            elif algorithm == 'reg':
                s_S, delta = reg(J_S, gradf_S, delta)
            else:
                s_S, delta = line_search(f, x, U_S, J_S, gradf_S)

            # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*linalg.norm(rx)**2 > 0
            #Jx_S = J_S.dot(x.dot(U_S))
            #stopping_rule = -Delta_m + np.dot(Js_S,Jx_S) + (sigma/2)*linalg.norm(s_S)**2 > 0

        # Update parameter and take step
        if algorithm.startswith('tr'):
            x, delta = tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta)
        elif algorithm == 'reg':
            x, delta = tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta, GAMMA1=2., GAMMA2=0.5) # grow/shrink swapped
        else:
            x = x + delta*U_S.dot(s_S)

        k += 1
        budget += len(S)
        #print 'Iteration:', k, 'max block size:', len(S)

        # function decrease metrics
        if fig is None:
            for itau, tau in enumerate([1e-1,1e-3,1e-5,1e-7]):
                if np.isnan(tau_budget[itau]) and linalg.norm(gradf(x)) <= tau*linalg.norm(gradf(x0)):
                    tau_budget[itau] = budget
            if np.all(np.isfinite(tau_budget)): # Stop if all function decrease metrics satisfied
                return tau_budget
        else: # plotting
            plot_data[0,k] = f(x)-fxopt
            plot_data[1,k] = linalg.norm(gradf(x))
            plot_data[2,k] = len(S)

    # Output
    #monitor(k, r, x, f, delta, algorithm, gradf)

    # Return function decrease metrics (some unsatisfied)
    if fig is None:
        return tau_budget
    else: # plotting
        return plot_data

""" Trust Region Update """
def tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta, GAMMA1=0.5, GAMMA2=2., update='standard'):

    # Trust Region parameters
    ETA1 = 0.25
    ETA2 = 0.75
    COUPL = 0.1
    DELTA_MIN = 1e-150
    DELTA_MAX = 1e150

    # Evaluate sufficient decrease
    s = U_S.dot(s_S)
    rho = (f(x) - f(x+s))/Delta_m

    # Couple delta to ng_S
    if update == 'coupled':

        if rho >= ETA1 and linalg.norm(gradf_S) > COUPL*delta:
            x = x + s
        else:
            delta *= GAMMA1
            delta = max(delta, DELTA_MIN)

    # Standard update
    else:

        # Accept trial point
        if rho >= ETA1:
            x = x + s

        # Update trust region radius
        if rho < ETA1:
            delta *= GAMMA1
            delta = max(delta,DELTA_MIN)
        elif rho >= ETA2:
            delta *= GAMMA2
            delta = min(delta,DELTA_MAX)

    return x, delta

""" Trust Region Subproblem """
def trs(J_S, gradf_S, delta):
    p = J_S.shape[1]

    # Trust Region subproblem parameters
    LEPS = 1e-8
    KE = 0.01

    # J_S'J_S full rank
    if np.linalg.matrix_rank(J_S) == p:

        # Set lambda (for newton iteration)
        lamda = 0

        # Solve normal equations to find search direction
        _, R_S = linalg.qr(J_S, mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

        # Trust region inactive: interior solution
        if ns_S < delta:
            return s_S
        # Else trust region active

    # J_S'J_S singular: lamda_1 = 0
    else:

        # Set lambda for newton iteration
        lamda = LEPS

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

        # Hard case: find eigenvector of zero eigenvalue
        if ns_S < delta:
            u_S = linalg.solve_triangular(R_S, np.zeros(p)) # since Q.T*zeros(m+p)=zeros(p)
            alpha1, alpha2 = quadeq(np.dot(u_S, u_S), 2 * np.dot(s_S, u_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic roots
            return s_S + alpha1 * u_S # FIXME: choosing alpha at random?
        # Else trust region active

    # Trust region active: newton iteration
    while ma.fabs(ns_S - delta) > KE * delta:

        # Solve R'w = s and calculate new lamda
        w_S = linalg.solve_triangular(R_S.T, s_S, lower=True)
        nw_S = linalg.norm(w_S)
        lamda += (ns_S - delta)/delta * (ns_S/nw_S)**2

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

    return s_S

""" Regularization Subproblem """
def reg(J_S, gradf_S, delta):
    p = J_S.shape[1]

    # Regularization parameters
    SIGMA_MIN = 1e-8

    # J_S'J_S singular: limit sigma to sigma_min
    if np.linalg.matrix_rank(J_S) != p:
        delta = max(delta,SIGMA_MIN)

    # Solve *perturbed* normal equations to find search direction
    _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(delta) * np.eye(p))), mode='economic')
    t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
    s_S = linalg.solve_triangular(R_S, t_S)

    return s_S, delta

""" Steihaug-Toint Conjugate Gradient """
@jit(nopython=True, nogil=True, cache=True)
def trs_approx(J_S, gradf_S, delta):
    p = J_S.shape[1]

    # Tolerance
    TAU = 1e-5

    # Initialize
    s_S = np.zeros(p)
    g_S = gradf_S
    H_S = np.dot(J_S.T, J_S)
    ng_S2 = norm(g_S)**2
    p_S = -g_S

    while norm(np.dot(H_S, s_S) + gradf_S) > TAU*norm(gradf_S):

        # Calculate curvature
        Jp_S = np.dot(J_S, p_S)
        kappa = np.dot(Jp_S, Jp_S)

        # Check for zero curvature
        if kappa < 1e-30: # Find boundary solution
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Calculate step length for s and g
        alpha = ng_S2/kappa

        # Trust region active: boundary solution
        if norm(s_S + alpha*p_S) >= delta:
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Take step for s and g
        s_S = s_S + alpha * p_S
        g_S = g_S + alpha * np.dot(H_S, p_S)

        # Calculate step length for p
        ng_S2_new = norm(g_S)**2
        beta = ng_S2_new/ng_S2
        ng_S2 = ng_S2_new

        # Take step for p
        p_S = -g_S + beta * p_S

    # Trust region inactive: interior solution
    return s_S


""" Return roots of quadratic equation """
def quadeq(a, b, c):
    warnings.simplefilter("error", RuntimeWarning)
    try:
        x1 = (-b + ma.sqrt(b * b - 4 * a * c)) / (2 * a)
        x2 = (-b - ma.sqrt(b * b - 4 * a * c)) / (2 * a)
    except RuntimeWarning: # failed step: delta too large
        x1 = 0
        x2 = 0
    warnings.resetwarnings()
    return x1, x2

""" Return positive root of quadratic equation """
@jit(nopython=True, nogil=True, cache=True)
def quadeq_pos(a, b, c):
    x1 = np.divide(-b + ma.sqrt(b * b - 4 * a * c), 2 * a)
    x2 = np.divide(-b - ma.sqrt(b * b - 4 * a * c), 2 * a)
    return max(0,x1,x2)

""" Return 2-norm of vector """
@jit(nopython=True, nogil=True, cache=True)
def norm(v):
    return ma.sqrt(np.dot(v,v))

""" Line Search """
def line_search(f, x, U_S, J_S, gradf_S):

    # Solve block-reduced normal equations to find search direction
    s_S = search_direction(J_S, gradf_S)

    # Do backtracking line search to find step length
    alpha = b_Armijo(U_S, s_S, x, f, gradf_S)

    return s_S, alpha

""" Search Direction from Normal Equations"""
def search_direction(J_S, gradf_S):
    p = J_S.shape[1]

    # Regularization parameters
    KAPPA_TOL = 1e8
    SIGMA = 1e-8

    # Solve block-reduced normal equations to find search direction
    kappa = np.linalg.cond(J_S)
    print 'k(J_S): %.2e' % kappa
    if kappa >= KAPPA_TOL:
        print 'WARNING: Jacobian ill-conditioned!!'
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(SIGMA) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
    else:
        _, R_S = linalg.qr(J_S, mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)

    return s_S

""" Backtracking-Armijo Line Search """
def b_Armijo(U_S, s_S, x, f, gradf_S):

    # Linesearch parameters
    alpha = 5  # ALPHA_MAX > 0
    C = 0.5  # in (0,1)
    RHO = 0.5  # in (0,1)

    fx = f(x)
    s = U_S.dot(s_S)
    delta = C*np.dot(gradf_S,s_S)
    while f(x + alpha*s) > fx + alpha*delta and alpha > 0:
        alpha *= RHO
    return alpha

""" Output Monitoring Information """
def monitor(k, r, x, f, delta, algorithm, gradf, gradf_S=None):

    print '++++ Iteration', k, '++++'
    if algorithm == 'tr':
        print 'delta: %.2e' % delta
    elif algorithm == 'reg':
        print 'sigma: %.2e' % delta
    elif delta is not None:
        print 'alpha: %.2e' % delta

    nr = linalg.norm(r(x))
    ng = linalg.norm(gradf(x))
    nJrr = ng / nr
    if gradf_S is not None:
        ng_S = linalg.norm(gradf_S)
        nJ_Srr = ng_S / nr

    print 'x:', x, 'f(x):', f(x)
    print '||r(x)||: %.2e' % nr, '||gradf(x)||: %.2e' % ng,
    if  gradf_S is not None: print '||gradf_S(x)||: %.2e' % ng_S
    print "||J'r||/||r||: %.2e" % nJrr,
    if gradf_S is not None: print "||J_S'r||/||r||: %.2e" % nJ_Srr

    if gradf_S is None: print

""" Test Problem Selector """
def get_test_problem(name, sifParams):

    if name.isupper(): # CUTEst problem
        if not cutermgr.isCached(name):
            cutermgr.prepareProblem(name,sifParams=sifParams)
        prob = cutermgr.importProblem(name)
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

main()
