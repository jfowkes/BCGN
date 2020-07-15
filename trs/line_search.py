""" Backtracking-Armijo Line Search """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
import math as ma

""" Line Search """
def line_search(f, x, S, J_S, gradf_S):

    # Solve block-reduced normal equations to find search direction
    s_S = search_direction(J_S, gradf_S)

    # Do backtracking line search to find step length
    alpha = b_Armijo(S, s_S, x, f, gradf_S)

    return s_S, alpha

""" Search Direction from Normal Equations"""
def search_direction(J_S, gradf_S):
    p = J_S.shape[1]

    # Regularization parameters
    KAPPA_TOL = 1e8
    SIGMA = 1e-8

    # Solve block-reduced normal equations to find search direction
    kappa = np.linalg.cond(J_S)
    print('k(J_S): %.2e' % kappa)
    if kappa >= KAPPA_TOL:
        print('WARNING: Jacobian ill-conditioned!!')
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(SIGMA) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
    else:
        _, R_S = linalg.qr(J_S, mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)

    return s_S

""" Backtracking-Armijo Line Search """
def b_Armijo(S, s_S, x, f, gradf_S):

    # Linesearch parameters
    alpha = 5  # ALPHA_MAX > 0
    C = 0.5  # in (0,1)
    RHO = 0.5  # in (0,1)

    fx = f(x)
    if S is not None: # sketching in n
        s = S.dot(s_S)
    else: # sketching in m
        s = s_S
    delta = C*np.dot(gradf_S,s_S)
    while f(x + alpha*s) > fx + alpha*delta and alpha > 0:
        alpha *= RHO
    return alpha
