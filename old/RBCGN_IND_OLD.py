""" Random Block-Coordinate Gauss-Newton (Individual Functions Old Version) """
from __future__ import absolute_import, division, unicode_literals, print_function
from trs.trs_exact import trs, tr_update, tr_update_fancy
from trs.trs_approx import trs_approx_precon
from trs.reg import reg, reg_update, reg_update_fancy
from trs.creg import creg, creg_update, creg_update_fancy
from trs.reg_approx import reg_approx
from trs.line_search import line_search
import numpy as np
import scipy.linalg as linalg
import time

def RBCGN_IND_OLD(r, J, x0, p, sampling='coordinate', kappa=1, astep=None, it_max=100, tau=1e-1, fxopt=0, runtype='plot', algorithm='tr', subproblem='normal'):
    n = x0.size

    # Full function and gradient
    def f(z): return 0.5*np.dot(r(z),r(z))
    #def gradf(z): return J(z).T.dot(r(z))

    if runtype == 'plot':  # plotting
        plot_data = np.full(it_max+1,np.nan)
        plot_data[0] = f(x0)
    elif runtype == 'metrics':  # runtime
        runtime = np.full(it_max+1,np.nan)
        runtime[0] = 0
    else:
        raise ValueError('Uknown runtype ' + runtype)

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
    else:
        raise ValueError('Sampling type ' + sampling + ' unimplemented')

    # Start timer
    start_time = time.time()

    # Initialize block sampling function
    sampling_func(n,p,init=True)

    k = 0
    x = x0
    delta = None
    while k < it_max and f(x) > fxopt + tau*(f(x0)-fxopt): # objective decrease

        # Assemble block-reduced matrices
        if 'approx' in algorithm: # sparse
            S, S_scale = sampling_func(n,p,sparse=True)
            #inds = np.argmax(S==1,axis=0).A1
            #J_S = J(x,inds)*S_scale
            J_S = J(x).dot(S*S_scale)
            J_ST = J_S.T.tocsr()
            rx = r(x)
            gradf_S = J_ST.dot(rx)
        else: # dense
            S, S_scale = sampling_func(n,p,sparse=False)
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

            # Assemble block-reduced matrices
            if 'approx' in algorithm: # sparse
                S, S_scale = sampling_func(n,step,step=True,sparse=True)
                #inds = np.argmax(S==1,axis=0).A1
                #J_S = J(x,inds)*S_scale
                J_S = J(x).dot(S*S_scale)
                J_ST = J_S.T.tocsr()
                gradf_S = J_ST.dot(rx)
            else: # dense
                S, S_scale = sampling_func(n,step,step=True,sparse=False)
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
            runtime[k] = time.time()-start_time
        else: # plotting
            plot_data[k] = f(x)

    # Debug output
    #monitor(k, r, x, f, delta, algorithm, gradf)

    # Return function decrease metrics (some unsatisfied)
    if runtype == 'metrics':
        return runtime
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
