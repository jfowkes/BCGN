""" Random Block-Coordinate Gauss-Newton """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update, tr_update_fancy
from trs.trs_approx import trs_approx_precon
from trs.reg import reg, reg_update, reg_update_fancy
from trs.creg import creg, creg_update, creg_update_fancy
from trs.reg_approx import reg_approx
from trs.line_search import line_search
from scipy.sparse import csr_matrix
import numpy as np
import scipy.linalg as linalg
import math as ma
import time

def RBCGN(r, J, x0, p, sampling='coordinate', kappa=1, astep=None, it_max=100, ftol=1e-10, fxopt=0, runtype='plot', grad_evals=None, metrics=None, algorithm='tr', subproblem='normal'):
    n = x0.size

    # Full function and gradient
    def f(z): return 0.5*np.dot(r(z),r(z))
    def gradf(z): return J(z).T.dot(r(z))

    if runtype == 'plot': # plotting
        plot_data = np.full((3,it_max+1),np.nan)
        plot_data[0,0] = f(x0)-fxopt
        plot_data[1,0] = linalg.norm(gradf(x0))
    elif runtype == 'metrics': # metrics
        budget = 0
        fail_count = 0
        x_prev = None
        tau_budget = np.full(len(metrics),np.inf)
        tau_runtime = np.full(len(metrics),np.inf)
    else:
        raise ValueError('Uknown runtype '+runtype)

    # Set sampling function
    if sampling == 'coordinate':
        from sampling_funcs import random_coordinate as sampling_func
    elif sampling == 'cyclic':
        from sampling_funcs import cyclic_coordinate as sampling_func
    elif sampling == 'gaussian':
        from sampling_funcs import random_gaussian as sampling_func
    elif sampling == 'hashing':
        from sampling_funcs import random_hashing as sampling_func
    elif sampling == 'hashing_variant':
        from sampling_funcs import random_hashing_variant as sampling_func
    elif sampling == 'gauss_southwell':
        from sampling_funcs import gauss_southwell_coordinate as sampling_func
        from sampling_funcs import gauss_southwell_update_gradient as update_grad
    elif sampling == 'thompson':
        from sampling_funcs import thompson_coordinate as sampling_func
        from sampling_funcs import thompson_update_gradient as update_grad
    else:
        raise ValueError('Sampling type ' + sampling + ' unimplemented')

    # Start timer
    start_time = time.time()

    # Initialize block sampling function
    sampling_func(n,p,init=True)

    k = 0
    x = x0
    delta = None
    while (runtype == 'metrics' and budget < grad_evals*n and fail_count < 100) or (runtype == 'plot' and k < it_max and ma.fabs(f(x) - fxopt) > ftol):

        # Update gradient (for Gauss-Southwell and Thompson)
        if sampling == 'gauss_southwell':
            update_grad(gradf(x))
        elif sampling == 'thompson':
            update_grad(gradf(x)/f(x))

        # Randomly select blocks
        S, S_scale = sampling_func(n,p)

        # Assemble block-reduced matrices
        if 'tr_approx' in algorithm: # sparse
            J_S = J(x).dot(csr_matrix(S*S_scale))
            J_ST = J_S.T.tocsr()
            rx = r(x)
            gradf_S = J_ST.dot(rx)
        else: # dense
            J_S = J(x).dot(S*S_scale)
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
        elif algorithm == 'tr_approx':
            s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
        elif algorithm == 'reg':
            s_S, delta = reg(J_S, gradf_S, delta)
        elif algorithm == 'creg':
            s_S, delta = creg(J_S, gradf_S, delta)
        elif algorithm == 'reg_approx':
            s_S = reg_approx(J_S, rx, delta)
        else: # linesearch
            s_S, delta = line_search(f, x, S*S_scale, J_S, gradf_S)

        # Loop tolerance
        Js_S = J_S.dot(s_S)
        Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
        stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
        #stopping_rule = -Delta_m + kappa*delta*delta > 0
        #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Iteratively refine block size
        while kappa != 1 and S.shape[1] != n and stopping_rule:

            # Increase block size
            step = min(astep,n-S.shape[1])
            #print('Increasing block size to:',S.shape[1]+step)
            S, S_scale = sampling_func(n,step,step=True)

            # Assemble block-reduced matrices
            if 'tr_approx' in algorithm: # sparse
                J_S = J(x).dot(csr_matrix(S*S_scale))
                J_ST = J_S.T.tocsr()
                gradf_S = J_ST.dot(rx)
            else: # dense
                J_S = J(x).dot(S*S_scale)
                gradf_S = J_S.T.dot(rx)

            # Set initial trust region radius
            if k == 0 and (algorithm.startswith('tr') or algorithm.__contains__('reg')):
                delta = linalg.norm(gradf_S)/10

            # Debug output
            #monitor(k, r, x, f, delta, algorithm, gradf, gradf_S)

            # Solve subproblem
            if algorithm == 'tr':
                s_S = trs(J_S, gradf_S, delta)
            elif algorithm == 'tr_approx':
                s_S = trs_approx_precon(J_S, J_ST, gradf_S, delta)
            elif algorithm == 'reg':
                s_S, delta = reg(J_S, gradf_S, delta)
            elif algorithm == 'creg':
                s_S, delta = creg(J_S, gradf_S, delta)
            elif algorithm == 'reg_approx':
                s_S = reg_approx(J_S, rx, delta)
            else: # linesearch
                s_S, delta = line_search(f, x, S*S_scale, J_S, gradf_S)

            # Loop tolerance
            Js_S = J_S.dot(s_S)
            Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
            stopping_rule = -Delta_m + (1-kappa)/2*np.power(np.linalg.norm(rx),2) > 0
            #stopping_rule = -Delta_m + kappa*delta*delta > 0
            #stopping_rule = linalg.norm(gradf_S) > kappa*delta

        # Update coord/column budget
        if runtype == 'metrics':
            if np.any(x != x_prev): # we are at new location
                budget += S.shape[1]
                S_prev = S
                fail_count = 0
            else: # x == x_prev so don't count already evaluated coords/columns
                budget += len(set(map(tuple,S.T)) - set(map(tuple,S_prev.T))) # column difference
                S_prev = np.array(list(set(map(tuple,S_prev.T)) | set(map(tuple,S.T)))).T # column union
                fail_count += 1
            # print('Iteration:', k, 'max block size:', S.shape[1])
            x_prev = x

        # Update parameter and take step
        #Delta_m = -np.dot(gradf_S,s_S) - 0.5*np.dot(Js_S,Js_S)
        if algorithm.startswith('tr'):
            if subproblem != 'fancy': # standard update
                x, delta = tr_update(f, x, s_S, S*S_scale, Delta_m, delta)
            else: # sophisticated update
                x, delta = tr_update_fancy(f, x, s_S, S*S_scale, gradf_S, Js_S, delta)
        elif algorithm.startswith('reg'):
            if subproblem != 'fancy': # standard update
                x, delta = reg_update(f, x, s_S, S*S_scale, Delta_m, delta)
            else: # sophisticated update
                x, delta = reg_update_fancy(f, x, s_S, S*S_scale, gradf_S, Js_S, delta)
        elif algorithm.startswith('creg'):
            if subproblem != 'fancy': # standard update
                x, delta = creg_update(f, x, s_S, S*S_scale, Delta_m, delta)
            else: # sophisticated update
                x, delta = creg_update_fancy(f, x, s_S, S*S_scale, gradf_S, Js_S, delta)
        else: # linesearch
            s = S_scale*S.dot(s_S)
            x = x + delta*s
        k += 1

        # function decrease metrics
        if runtype == 'metrics':
            for itau, tau in enumerate(metrics):
                #if np.isinf(tau_budget[itau]) and np.linalg.norm(gradf(x)) <= tau*np.linalg.norm(gradf(x0)):
                if np.isinf(tau_budget[itau]) and f(x) <= fxopt + tau*(f(x0)-fxopt): # function decrease condition as opposed to gradient
                    tau_budget[itau] = budget
                    tau_runtime[itau] = time.time()-start_time
            if np.all(np.isfinite(tau_budget)): # Stop if all function decrease metrics satisfied
                return tau_budget, tau_runtime
        else: # plotting
            plot_data[0,k] = f(x)-fxopt
            plot_data[1,k] = linalg.norm(gradf(x))
            plot_data[2,k] = S.shape[1]

    # Debug output
    #monitor(k, r, x, f, delta, algorithm, gradf)

    # Return function decrease metrics (some unsatisfied)
    if runtype == 'metrics':
        return tau_budget, tau_runtime
    else: # else return plot data
        return plot_data

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
