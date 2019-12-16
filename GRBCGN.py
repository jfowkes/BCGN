""" Gaussian Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs
from trs.reg import reg
import numpy as np
import scipy.linalg as linalg
import math as ma

def GRBCGN(r, J, x0, sampling_func, fxopt, it_max, ftol, p, fig, kappa, algorithm='tr', subproblem='normal'):
    n = x0.size

    # Adaptive BCGN step size
    STEP = 5

    # Full function and gradient
    def f(z): return 0.5 * np.dot(r(z), r(z))
    def gradf(z): return J(z).T.dot(r(z))

    # Plotting
    if fig is not None:
        plot_data = np.full((3,it_max+1),ftol)
        plot_data[2,:] = np.full(it_max+1,np.nan)
        plot_data[0,0] = f(x0)-fxopt
        plot_data[1,0] = linalg.norm(gradf(x0))

    # Metrics
    budget = 0
    tau_budget = np.full(4,np.nan)

    k = 0
    x = x0
    delta = None
    x_prev = None
    while (not fig and budget < it_max*n) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):

        # Randomly select subspace
        if kappa == 1 and p == n: # GN
            S = np.eye(n)
        else: # random projection
            S = gaussian_basis_randn(n,p)

        # Assemble block-reduced matrices
        J_S = J(x).dot(S)
        rx = r(x)
        gradf_S = J_S.T.dot(rx)

        # Set initial trust region radius
        if k == 0 and (algorithm.startswith('tr') or algorithm.__contains__('reg')):
            delta = linalg.norm(gradf_S)/10
            if delta == 0:
                delta = 1

        # Debug output
        #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

        # Solve subproblem
        if algorithm == 'tr':
            s_S = trs(J_S, gradf_S, delta)
        elif algorithm == 'reg':
            s_S, delta = reg(J_S, gradf_S, delta)
        else:
            raise RuntimeError(algorithm + ' unimplemented!')

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = -Delta_m + kappa*delta*delta > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Iteratively refine block size
        while kappa != 1 and S.shape[1] != n and stopping_rule:

            # Increase block size
            step = min(STEP,n-S.shape[1])
            #print('Increasing block size to:',S.shape[1]+step)

            # Grow subspace
            S = gaussian_basis_grow(S, step)

            # Assemble block-reduced matrices
            J_S = J(x).dot(S)
            gradf_S = J_S.T.dot(rx)

            # Set initial trust region radius
            if k == 0 and (algorithm.startswith('tr') or algorithm.__contains__('reg')):
                delta = linalg.norm(gradf_S)/10

            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            elif algorithm == 'reg':
                s_S, delta = reg(J_S, gradf_S, delta)
            else:
                raise RuntimeError(algorithm + ' unimplemented!')

            # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
            #stopping_rule = -Delta_m + kappa*delta*delta > 0
            #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Update budget
        if kappa == 1 and p == n and np.all(x == x_prev): # GN failed step
            budget += 0 # don't count already evaluated coords
        else: # all coords are at new location
            budget += S.shape[1]
        #print('Iteration:', k, 'max block size:', S.shape[1])
        x_prev = x

        # Update parameter and take step
        #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
        if algorithm.startswith('tr'):
            x, delta = tr_update(f, x, s_S, S, Delta_m, delta)
        elif algorithm.startswith('reg'):
            x, delta = reg_update(f, x, s_S, S, Delta_m, delta)
        else:
            raise RuntimeError(algorithm + ' unimplemented!')
        k += 1

        # function decrease metrics
        if fig is None:
            for itau, tau in enumerate([1e-1,1e-3,1e-5,1e-7]):
                #if np.isnan(tau_budget[itau]) and np.linalg.norm(gradf(x)) <= tau*np.linalg.norm(gradf(x0)):
                if np.isnan(tau_budget[itau]) and f(x) <= tau*f(x0): # function decrease condition as opposed to gradient
                    tau_budget[itau] = budget
            if np.all(np.isfinite(tau_budget)): # Stop if all function decrease metrics satisfied
                return tau_budget
        else: # plotting
            plot_data[0,k] = f(x)-fxopt
            plot_data[1,k] = linalg.norm(gradf(x))
            plot_data[2,k] = S.shape[1]

    # Debug output
    #monitor(k, r, x, f, delta, algorithm, gradf)

    # Return function decrease metrics (some unsatisfied)
    if fig is None:
        return tau_budget
    else: # plotting
        return plot_data

""" Generate random partial basis """
def gaussian_basis_randn(n, k):
    return np.random.randn(n,k)/ma.sqrt(k)

""" Grow partial basis """
def gaussian_basis_grow(U, l):
    n, k = U.shape
    return np.hstack((U*ma.sqrt(k),np.random.randn(n,l)))/ma.sqrt(k+l)

""" Trust Region Update """
def tr_update(f, x, s_S, S, Delta_m, delta):

    # Trust Region parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = 0.5
    GAMMA2 = 2.
    DELTA_MIN = 1e-15
    DELTA_MAX = 1e3

    # Evaluate sufficient decrease
    s = S.dot(s_S)
    rho = (f(x) - f(x+s))/Delta_m

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

""" Regularization Update (Standard) """
def reg_update(f, x, s_S, S, Delta_m, sigma):

    # Regularisation parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = ma.sqrt(2.)
    GAMMA2 = ma.sqrt(0.5)
    SIGMA_MIN = 1e-15
    SIGMA_MAX = 1e3

    # Evaluate sufficient decrease
    s = S.dot(s_S)
    rho = (f(x) - f(x+s))/(Delta_m - 0.5*sigma*np.dot(s_S,s_S))

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update regularisation parameter
    if rho < ETA1:
        sigma *= GAMMA1
        sigma = min(sigma,SIGMA_MAX)
    elif rho >= ETA2:
        sigma *= GAMMA2
        sigma = max(sigma,SIGMA_MIN)

    return x, sigma

""" Output Monitoring Information """
def monitor(k, r, x, f, delta, algorithm, gradf, gradf_S=None):

    print('++++ Iteration', k, '++++')
    if algorithm.startswith('tr'):
        print('delta: %.2e' % delta)
    elif algorithm.__contains__('reg'):
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
    print('||r(x)||: %.2e' % nr, '||g(x)||: %.2e' % ng,end='')
    if  gradf_S is not None: print(' ||g_S(x)||: %.2e' % ng_S)
    print("||J'r||/||r||: %.2e" % nJrr,end='')
    if gradf_S is not None: print(" ||J_S'r||/||r||: %.2e" % nJ_Srr)

    if gradf_S is None: print()
