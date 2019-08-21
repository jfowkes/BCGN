""" Quadratic Regularization Subproblem """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
import math as ma

""" Regularization Update (Standard) """
def reg_update(f, x, s_S, S, Delta_m, sigma):

    # Trust Region parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = 2.
    GAMMA2 = 0.5
    SIGMA_MIN = 1e-150
    SIGMA_MAX = 1e150

    # Evaluate sufficient decrease
    s = np.zeros(len(x))
    s[S] = s_S
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
    GAMMA1 = 2.
    GAMMA2 = 0.5
    SIGMA_MIN = 1e-150
    SIGMA_MAX = 1e150

    # Evaluate sufficient decrease
    s = np.zeros(len(x))
    s[S] = s_S
    fx = f(x)
    fxs = f(x+s)
    ss = np.dot(s_S,s_S)
    gs = np.dot(gradf_S,s_S)
    sHs = np.dot(Js_S,Js_S)
    rho = (fx - fxs)/(-gs-0.5*sHs - 0.5*sigma*ss)

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update regularisation parameter
    if rho < 0: # very unsuccessful
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
