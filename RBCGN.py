""" Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update_fancy
from trs.trs_approx import trs_approx, trs_approx_precon
from trs.reg import reg, reg_update
from trs.reg_approx import reg_approx
from trs.line_search import line_search
from scipy.sparse import csr_matrix
import numpy as np
import scipy.linalg as linalg
import math as ma

def RBCGN(r, J, x0, sampling_func, fxopt, it_max, ftol, p, fig, kappa, algorithm='tr'):
    n = x0.size

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

    # Initialize block sampling function
    sampling_func(n,p,init=True)

    k = 0
    x = x0
    delta = None
    while (not fig and budget < it_max*n) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):

        # Randomly select blocks
        S = sampling_func(n,p)

        # Assemble block-reduced matrices
        if 'tr_approx' in algorithm: # sparse
            U_S = csr_matrix((np.ones(len(S)),(S,range(len(S)))),shape=(n,len(S)))
            J_S = J(x).dot(U_S)
            J_ST = J_S.T.tocsr()
            rx = r(x)
            gradf_S = J_ST.dot(rx)
        else: # dense
            J_S = J(x)[:,S]
            rx = r(x)
            gradf_S = J_S.T.dot(rx)

        # Set initial trust region radius
        if k == 0 and algorithm.startswith('tr') or algorithm.startswith('reg'):
            delta = linalg.norm(gradf_S)/10
            if delta == 0:
                delta = 1

        # Debug output
        #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

        # Solve subproblem
        if algorithm == 'tr':
            s_S = trs(J_S, gradf_S, delta)
        elif algorithm == 'tr_approx':
            s_S = trs_approx(J_S, J_ST, gradf_S, delta)
        elif algorithm == 'tr_approx_precon':
            s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
        elif algorithm == 'reg':
            s_S, delta = reg(J_S, gradf_S, delta)
        elif algorithm == 'reg_approx':
            s_S = reg_approx(J_S, rx, delta)
        else:
            s_S, delta = line_search(f, x, S, J_S, gradf_S)

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = -Delta_m + kappa*delta*delta > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Iteratively refine block size
        p_in = len(S)
        while kappa != 1 and p_in != n and stopping_rule:

            # Increase block size
            step = min(STEP,n-p_in)
            #print('Increasing block size to:',p_in+step)
            S = sampling_func(n,step,step=True)

            # Assemble block-reduced matrices
            if 'tr_approx' in algorithm: # sparse
                U_S = csr_matrix((np.ones(len(S)),(S,range(len(S)))),shape=(n,len(S)))
                J_S = J(x).dot(U_S)
                J_ST = J_S.T.tocsr()
                gradf_S = J_ST.dot(rx)
            else: # dense
                J_S = J(x)[:,S]
                gradf_S = J_S.T.dot(rx)

            # Set initial trust region radius
            if k == 0 and algorithm.startswith('tr') or algorithm.startswith('reg'):
               delta = linalg.norm(gradf_S)/10

            p_in += step

            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            elif algorithm == 'tr_approx':
                s_S = trs_approx(J_S, J_ST, gradf_S, delta)
            elif algorithm == 'tr_approx_precon':
                s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
            elif algorithm == 'reg':
                s_S, delta = reg(J_S, gradf_S, delta)
            elif algorithm == 'reg_approx':
                s_S = reg_approx(J_S, rx, delta)
            else:
                s_S, delta = line_search(f, x, S, J_S, gradf_S)

            # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
            #stopping_rule = -Delta_m + kappa*delta*delta > 0
            #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        budget += p_in
        #print('Iteration:', k, 'max block size:', p_in)

        # Update parameter and take step
        #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
        if algorithm.startswith('tr'):
            #x, delta = tr_update(f, x, s_S, S, gradf_S, Delta_m, delta)
            x, delta = tr_update_fancy(f, x, s_S, S, gradf_S, Js_S, delta)
        elif algorithm.startswith('reg'):
            x, delta = reg_update(f, x, s_S, S, Delta_m, delta)
            #x, delta = reg_update_fancy(f, x, s_S, S, gradf_S, Js_S, delta)
        else:
            s = np.zeros(n)
            s[S] = s_S
            x = x + delta*s
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
    elif algorithm.startswith('reg'):
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
