""" Block-Coordinate Gauss-Newton """
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import math as ma
import warnings
import pickle
import cutermgr
import os

""" Main function """
def main():

    # Load test function minimizers
    minimizers = pickle.load(open('minimizers.ser','rb'))

    # Main parameters
    TOL = 1e-15
    K_MAX = 200
    GS = False
    ALG = 'tr'

    # Plotting parameters
    PLOT = False
    SAVEFIG = False

    # Loop over test functions
    kappas = [1.,0.9999,0.999,0.99,0.9,0.8]
    funcs = ['BroydenTridiagonal', 'Chebyquad', 'Osborne2', 'AIRCRFTA','PowellSingular','Osborne1','COOLHANS']
    metrics = ['accuracy','revals','budget']
    measure = np.zeros((len(funcs),len(kappas),11,len(metrics)))
    labels = []
    for ifunc, func in enumerate(funcs):
        for ikappa, kappa in enumerate(kappas):
            print '\n====== Kappa: ' + str(kappa) + ' ======'
            labels += ['Kappa: ' + str(kappa)]

            # Fix RNG seed
            np.random.seed(0)

            # Get test function
            r, J, x0 = get_test_problem(func)
            n = x0.size
            xopt = minimizers[func]

            # Plotting
            if PLOT:
                fig = plt.figure(ifunc + 1,figsize=(24,6))
                ax1 = fig.add_subplot(1,3,1)
                ax2 = fig.add_subplot(1,3,2)
                ax3 = fig.add_subplot(1,3,3)
            else:
                fig = None

            # Run RBCGN
            print '====== ' + func + ' ======'
            legend = []
            for p in range(1,n+1):
                legend += ['Block Size ' + str(p)]
                print '\n======',legend[p-1], '======'
                acc,revals,budget = RBCGN(r,J,x0,xopt,K_MAX,TOL,p,fig,kappa,algorithm=ALG,gaussSouthwell=GS)
                measure[ifunc,ikappa,p-1,0] = acc
                measure[ifunc,ikappa,p-1,1] = revals
                measure[ifunc,ikappa,p-1,2] = budget

            # Plotting
            if PLOT:
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
                    plt.savefig(dir+'/'+func)
                plt.show()

    # Plot performance profiles
    for imetr, metr in enumerate(metrics):
        for p in range(4):
            fig_title = metr + ' - block size: ' + str(p+1)
            save_dir = 'figures/perf_profiles/' + ALG  + ("_GS" if GS else "") + '/' + str(p+1) + '/'
            performance_profile(measure[:,:,p,imetr],labels,fig_title,metr,save_dir)

""" Random Block-Coordinate Gauss-Newton """
def RBCGN(r, J, x0, xopt, k_max, tol, p, fig, kappa, algorithm='tr', redrawFailed=True, plotFailed=True, gaussSouthwell=False):

    # Full function and gradient
    def f(z): return 0.5 * np.dot(r(z), r(z))
    def gradf(z): return J(z).T.dot(r(z))

    # Plotting
    if fig is not None:
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        hl1, = ax1.semilogy(0,f(x0),nonposy='clip',linewidth=2)
        hl2, = ax2.semilogy(0,linalg.norm(gradf(x0)),nonposy='clip',linewidth=2)
        hl3, = ax3.plot(0, p, linewidth=2)

    # Set initial trust region radius
    delta = None
    if algorithm == 'tr':
        delta = linalg.norm(gradf(x0))/10

    k = 0
    pk = 0
    x = x0
    n = x.size
    accepted = True
    budget = 0
    while f(x) > tol and k < k_max:

        # Evaluate full gradient for Gauss-Southwell
        if gaussSouthwell:
            sorted_nginds = np.argsort(np.fabs(gradf(x)))[::-1]
        
        # Randomly select blocks
        if redrawFailed or accepted:
            if gaussSouthwell:
                S = sorted_nginds[0:p]
            else:
                S = np.random.permutation(np.arange(n))[0:p]
            U_S = np.zeros((n,p))
            for j in range(0,p):
                U_S[S[j],j] = 1

        # Assemble block-reduced matrices
        J_S = J(x).dot(U_S)
        rx = r(x)
        gradf_S = J_S.T.dot(rx)

        # Output
        #monitor(k, r, x, f, delta, algorithm, accepted, gradf, gradf_S)

        # Solve trust region subproblem
        if algorithm == 'tr':
            s_S = trs(J_S, gradf_S, delta)
        else:
            s_S, delta = line_search(f, x, U_S, J_S, gradf_S)

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*linalg.norm(rx)**2 > 0

        # Iteratively refine block size
        p_in = p
        while p_in != n and stopping_rule:

            # Increase block size
            #print 'Increasing block size to:', p_in+1
            if gaussSouthwell:
                ind = sorted_nginds[p_in]
            else:
                S = np.nonzero(U_S)[0]
                inds = np.setdiff1d(np.arange(n), S)
                ind = np.random.choice(inds, 1)
            U_ind = np.zeros((n, 1))
            U_ind[ind, :] = 1
            U_S = np.hstack((U_S, U_ind))
            J_S = J(x).dot(U_S)
            rx = r(x)
            gradf_S = J_S.T.dot(rx)
            p_in += 1

            # Output
            #monitor(k, r, x, f, delta, algorithm, True, gradf, gradf_S)

            # Solve trust region subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            else:
                s_S, delta = line_search(f, x, U_S, J_S, gradf_S)

            # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*linalg.norm(rx)**2 > 0

        # Update trust region
        if algorithm == 'tr':
            x, delta, accepted = tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta)
        else:
            x = x + delta*U_S.dot(s_S)

        k += 1

        # Plotting
        if (fig is not None) and (plotFailed or accepted):
            pk += 1
            update_line(ax1,hl1,pk,f(x))
            update_line(ax2,hl2,pk,linalg.norm(gradf(x)))
            update_line(ax3, hl3, k + 1, p_in)

        if accepted:
            budget += p_in

    # Output
    #monitor(k, r, x, f, delta, algorithm, accepted, gradf)

    # Return metrics
    return ma.fabs(f(x)-f(xopt)), r.calls, budget

""" Trust Region Update """
def tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta, update = 'none'):
    accepted = False

    # Trust Region parameters
    ETA1 = 0.25
    ETA2 = 0.75
    GAMMA1 = 0.5
    GAMMA2 = 2.
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
            accepted = True
        else:
            delta *= GAMMA1
            delta = max(delta, DELTA_MIN)

    # Standard update
    else:

        # Accept trial point
        if rho >= ETA1:
            x = x + s
            accepted = True

        # Update trust region radius
        if rho < ETA1:
            delta *= GAMMA1
            delta = max(delta,DELTA_MIN)
        elif rho >= ETA2:
            delta *= GAMMA2
            delta = min(delta,DELTA_MAX)

    return x, delta, accepted

""" Trust Region Subproblem """
def trs(J_S, gradf_S, delta):
    p = J_S.shape[1]

    # Trust Region subproblem parameters
    LEPS = 1e-5
    KE = 0.01

    # QR on J_S
    _, R_S = linalg.qr(J_S, mode='economic')

    # J_S'J_S full rank
    if np.linalg.matrix_rank(J_S) == p:

        # Set lambda (for newton iteration)
        lamda = 0

        # Solve normal equations to find search direction
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
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))),mode='economic')
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


""" Calculate and Plot Performance Profile """
def performance_profile(measure,solver_labels,fig_title,fig_name,save_dir):
    '''
    :param measure: prob x solver array,
     smallest values assumed to be the best
    '''
    pn = measure.shape[0]
    sn = measure.shape[1]

    ratio = np.zeros((pn,sn))
    for p in range(pn):
        for s in range(sn):
            ratio[p,s] = measure[p,s]/min(measure[p,:])

    def profile(s,t):
        prob = 0
        for p in range(pn):
            if ratio[p,s] <= t:
                prob += 1
        return prob/pn

    t = np.linspace(1,20)
    prof = np.vectorize(profile)
    plt.figure(100)
    plt.clf()
    for s in range(sn):
        y = prof(s,t)
        plt.plot(t,y,'-',linewidth=2,clip_on=False)
    plt.grid()
    plt.xlabel('Performance Ratio')
    plt.legend(solver_labels,loc='lower right')
    plt.title(fig_title,fontsize=13)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

""" Output Monitoring Information """
def monitor(k, r, x, f, delta, algorithm, accepted, gradf, gradf_S=None):

    print '++++ Iteration', k, '++++'
    if algorithm == 'tr':
        print 'delta: %.2e' % delta
        if not accepted: print "Step Failed!"
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
def get_test_problem(name):

    # TODO: make this more efficient
    if name.isupper(): # CUTEst problem
        if not cutermgr.isCached(name):
            cutermgr.prepareProblem(name)
        prob = cutermgr.importProblem(name)
        def r(x): return prob.cons(x)
        def J(x): return prob.cons(x,True)[1]
        xinit = prob.getinfo()['x']

    else: # More-Garbow-Hillstrom
        mod = __import__('MGH', fromlist=[name])
        prob = getattr(mod, name)()
        r = prob.r
        J = prob.jacobian
        xinit = prob.initial

    # Define decorator for counting function calls
    # TODO: add option to disable this
    def count_calls(fn):
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            return fn(*args, **kwargs)

        wrapper.calls = 0
        wrapper.__name__ = fn.__name__
        return wrapper
    @count_calls
    def rw(x): return r(x)

    return rw, J, xinit

""" Real-time plotting """
def update_line(ax, hl, x_data, y_data):
    hl.set_xdata(np.append(hl.get_xdata(), x_data))
    hl.set_ydata(np.append(hl.get_ydata(), y_data))
    ax.relim()
    ax.autoscale_view()
    plt.draw()

main()
