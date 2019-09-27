""" Gaussian Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs
import numpy as np
import scipy.linalg as linalg
import math as ma
import warnings

def GRBCGN(r, J, x0, sampling_func, fxopt, it_max, ftol, p, fig, kappa, algorithm='tr'):
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

    k = 0
    x = x0
    delta = None
    while (not fig and budget < it_max*n) or (fig and k < it_max and ma.fabs(f(x) - fxopt) > ftol):

        # Randomly select subspace
        if kappa == 1 and p == n: # GN
            S = np.eye(n)
        else: # random projection
            S = gaussian_basis_randn(n, p)

        # Assemble block-reduced matrices
        J_S = J(x).dot(S)
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
        else:
            raise RuntimeError(algorithm + 'unimplemented!')

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = -Delta_m + kappa*delta*delta > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Iteratively refine block size
        p_in = p
        while kappa != 1 and p_in != n and stopping_rule:

            # Increase block size
            step = min(STEP,n-p_in)
            # print 'Increasing block size to:', p_in+step

            # Grow subspace
            S = gaussian_basis_grow(S, step)

            # Assemble block-reduced matrices
            J_S = J(x).dot(S)
            gradf_S = J_S.T.dot(rx)

            p_in += step

            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            else:
                raise RuntimeError(algorithm + 'unimplemented!')

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
            x, delta = tr_update(f, x, s_S, S, Delta_m, delta)
        else:
            raise RuntimeError(algorithm + 'unimplemented!')
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
    DELTA_MIN = 1e-150
    DELTA_MAX = 1e150

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