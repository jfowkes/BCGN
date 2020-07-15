""" Trust Region Subproblem using QR and More-Sorensen update """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
import math as ma
import warnings

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
    if S is not None: # sketching in n
        s = S.dot(s_S)
    else: # sketching in m
        s = s_S
    warnings.simplefilter("ignore", RuntimeWarning)
    rho = (f(x) - f(x+s))/Delta_m
    warnings.resetwarnings()

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

""" Trust Region Update (Sophisticated) """
def tr_update_fancy(f, x, s_S, S, gradf_S, Js_S, delta):

    # Trust Region parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA3 = 0.1
    GAMMA1 = 0.5
    GAMMA2 = 2.
    DELTA_MIN = 1e-15
    DELTA_MAX = 1e3

    # Evaluate sufficient decrease
    if S is not None: # sketching in n
        s = S.dot(s_S)
    else: # sketching in m
        s = s_S
    fx = f(x)
    fxs = f(x+s)
    gs = np.dot(gradf_S,s_S)
    sHs = 0.5*np.dot(Js_S,Js_S)
    warnings.simplefilter("ignore", RuntimeWarning)
    rho = (fx - fxs)/(-gs-0.5*sHs)
    warnings.resetwarnings()

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update trust region radius
    if rho < 0: # very unsuccessful
        alpha_bad = (1-ETA1)*gs/((1-ETA1)*(fx+gs)+ETA1*(fx+gs+sHs)-fxs)
        delta *= max(GAMMA3,alpha_bad)
    elif rho < ETA1: # unsuccessful
        delta *= GAMMA1
        delta = max(delta,DELTA_MIN)
    elif rho >= ETA2: # very successful
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
            _, u_S = linalg.eigh(R_S.T.dot(R_S),eigvals=(0,0)) # since R_S'R_S = J_S'J_S + lamda*I
            u_S = u_S[:,0] # flatten array
            alpha1, alpha2 = quadeq(np.dot(u_S, u_S), 2 * np.dot(s_S, u_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic roots
            return modelmin(s_S+alpha1*u_S, s_S+alpha2*u_S, J_S, gradf_S) # Find step that makes trs model smallest
        # Else trust region active

    # Trust region active: newton iteration
    k = 0
    lamda0 = lamda
    while ma.fabs(ns_S - delta) > KE * delta:

        # Solve R'w = s and calculate new lamda
        w_S = linalg.solve_triangular(R_S.T, s_S, lower=True)
        nw_S = linalg.norm(w_S)
        lamda += (ns_S - delta)/delta * (ns_S/nw_S)**2

        # Handle issues in newton iteration: return suboptimal step
        if lamda < 0 or k == 30:
            return s_S

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

        # Handle cycling in newton iteration: restart with shifted lambda
        if k == 15:
            lamda = lamda0 + 1e-10
        k += 1

    return s_S

""" Hard case: find step that makes trs model smallest """
def modelmin(s1_S, s2_S, J_S, gradf_S):
    Js1_S = J_S.dot(s1_S)
    Js2_S = J_S.dot(s2_S)
    qs1_S = np.dot(gradf_S,s1_S) + 0.5*np.dot(Js1_S,Js1_S)
    qs2_S = np.dot(gradf_S,s2_S) + 0.5*np.dot(Js2_S,Js2_S)
    if qs1_S < qs2_S:
        return s1_S
    else:
        return s2_S

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
