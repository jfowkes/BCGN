""" Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update
from trs.trs_approx import trs_approx
from trs.reg import reg, reg_update
from trs.line_search import line_search
import numpy as np
import scipy.linalg as linalg
import math as ma

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

    # Initialize block partition
    if partitionBlock:
        block_part = np.random.permutation(np.arange(x0.size))

    k = 0
    x = x0
    n = x.size
    delta = None
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

        # Set initial trust region radius
        if k == 0 and algorithm.startswith('tr') or algorithm == 'reg':
            delta = linalg.norm(gradf_S)/10
            if delta == 0:
                delta = 1

        # Debug output
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
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta
        #Jx_S = J_S.dot(x.dot(U_S))
        #stopping_rule = -Delta_m + np.dot(Js_S,Jx_S) + (sigma/2)*np.power(linalg.norm(s_S),2) > 0

        # Iteratively refine block size
        p_in = len(S)
        while kappa != 1 and p_in != n and stopping_rule:

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

            # Debug output
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
            stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
            #stopping_rule = linalg.norm(gradf_S) > kappa*delta
            #Jx_S = J_S.dot(x.dot(U_S))
            #stopping_rule = -Delta_m + np.dot(Js_S,Jx_S) + (sigma/2)*np.power(linalg.norm(s_S),2) > 0

        budget += p_in
        #print 'Iteration:', k, 'max block size:', p_in

        # Update parameter and take step
        #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
        if algorithm.startswith('tr'):
            x, delta = tr_update(f, x, s_S, U_S, gradf_S, Delta_m, delta)
        elif algorithm == 'reg':
            x, delta = reg_update(f, x, s_S, U_S, Delta_m, delta) # same as tr_update with grow/shrink swapped
        else:
            x = x + delta*U_S.dot(s_S)
        k += 1

        # function decrease metrics
        if fig is None:
            for itau, tau in enumerate([1e-1,1e-3,1e-5,1e-7]):
                if np.isnan(tau_budget[itau]) and np.linalg.norm(gradf(x)) <= tau*np.linalg.norm(gradf(x0)):
                    tau_budget[itau] = budget
            if np.all(np.isfinite(tau_budget)): # Stop if all function decrease metrics satisfied
                return tau_budget
        else: # plotting
            plot_data[0,k] = f(x)-fxopt
            plot_data[1,k] = linalg.norm(gradf(x))
            plot_data[2,k] = p_in

    # Debug output
    #monitor(k, r, x, f, delta, algorithm, gradf)

    # Return function decrease metrics (some unsatisfied)
    if fig is None:
        return tau_budget
    else: # plotting
        return plot_data

""" Output Monitoring Information """
def monitor(k, r, x, f, delta, algorithm, gradf, gradf_S=None):

    print('++++ Iteration', k, '++++')
    if algorithm.startswith('tr'):
        print('delta: %.2e' % delta)
    elif algorithm == 'reg':
        print('sigma: %.2e' % delta)
    elif delta is not None:
        print('alpha: %.2e' % delta)

    nr = linalg.norm(r(x))
    ng = linalg.norm(gradf(x))
    nJrr = ng / nr
    if gradf_S is not None:
        ng_S = linalg.norm(gradf_S)
        nJ_Srr = ng_S / nr

    print('x:', x, 'f(x):', f(x))
    print('||r(x)||: %.2e' % nr, '||gradf(x)||: %.2e' % ng,end='')
    if  gradf_S is not None: print('||gradf_S(x)||: %.2e' % ng_S)
    print("||J'r||/||r||: %.2e" % nJrr,end='')
    if gradf_S is not None: print("||J_S'r||/||r||: %.2e" % nJ_Srr)

    if gradf_S is None: print()
