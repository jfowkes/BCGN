""" Quadratic Regularization Subproblem """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
import math as ma
import warnings

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
    if S is not None: # sketching in n
        s = S.dot(s_S)
    else: # sketching in m
        s = s_S
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

""" Regularization Update (Sophisticated) """
def reg_update_fancy(f, x, s_S, S, gradf_S, Js_S, sigma):

    # Regularisation parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = ma.sqrt(2.)
    GAMMA2 = ma.sqrt(0.5)
    GAMMA3 = ma.sqrt(0.1)
    EPSX = 1e-8
    BETA = 1./100
    ALPHAMAX = 2.
    SIGMA_MIN = 1e-15
    SIGMA_MAX = 1e3

    # Evaluate sufficient decrease
    if S is not None: # sketching in n
        s = S.dot(s_S)
    else: # sketching in m
        s = s_S
    fx = f(x)
    fxs = f(x+s)
    ss = np.dot(s_S,s_S)
    gs = np.dot(gradf_S,s_S)
    sHs = np.dot(Js_S,Js_S)
    qs = fx + gs + 0.5*sHs
    rho = (fx - fxs)/(-gs-0.5*sHs - 0.5*sigma*ss)

    # Evaluate model agreement with f(x+s)
    xi = qs+0.5*sigma*ss - max(fxs,qs)

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update regularisation parameter (quadratic version of Gould, Porcelli, Toint)
    if rho >= 1 and xi >= EPSX: # very successful: match f(x+s)
        if fxs >= qs:
            roots = np.array(quadeq(2*(fxs-qs)+sHs,gs,2*BETA*xi))
            roots = roots[roots >= BETA**(1./2)]
            if roots.size == 0:
                sigma *= GAMMA3
                sigma = max(sigma,SIGMA_MIN)
            else: # exists root >= beta^1/2
                alpha_g = roots[np.argmin(roots - BETA**(1./2))]
                if alpha_g <= ALPHAMAX:
                    sigma += 2*(xi/ss)*((BETA-alpha_g**2)/alpha_g**2)
                else:
                    sigma *= GAMMA3
                    sigma = max(sigma,SIGMA_MIN)
        else:
            roots = np.array(quadeq(sHs,gs,2*BETA*xi))
            roots = roots[roots >= BETA**(1./2)]
            if roots.size == 0:
                sigma *= GAMMA3
                sigma = max(sigma,SIGMA_MIN)
            else: # exists root >= beta^1/2
                alpha_g = roots[np.argmin(roots - BETA**(1./2))]
                if alpha_g <= ALPHAMAX:
                    sigma *= BETA/alpha_g**2
                else:
                    sigma *= GAMMA3
                    sigma = max(sigma,SIGMA_MIN)
    elif rho >= 1 and xi < EPSX: # very successful
        sigma *= GAMMA2
        sigma = max(sigma,SIGMA_MIN)
    elif rho < 0: # very unsuccessful
        alpha_bad = (1-ETA1/2)*gs/(fx+gs-fxs)
        sigma = (-gs-sHs*alpha_bad)/(alpha_bad*ss)
    elif rho < ETA1: # unsuccessful
        sigma *= GAMMA1
        sigma = min(sigma,SIGMA_MAX)
    elif rho >= ETA2: # very successful
        sigma *= GAMMA2
        sigma = max(sigma,SIGMA_MIN)

    return x, sigma

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

""" Return roots of quadratic equation """
def quadeq(a, b, c):
    warnings.simplefilter("error", RuntimeWarning)
    try:
        x1 = (-b + ma.sqrt(b * b - 4 * a * c)) / (2 * a)
        x2 = (-b - ma.sqrt(b * b - 4 * a * c)) / (2 * a)
    except RuntimeWarning: # failed step: sigma too large
        x1 = 0
        x2 = 0
    warnings.resetwarnings()
    return x1, x2
