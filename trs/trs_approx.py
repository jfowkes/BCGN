""" Approximate Trust Region Subproblem using Steihaug-Toint CG """
from __future__ import absolute_import, division, unicode_literals, print_function
from numba import jit
import numpy as np
import math as ma

""" Steihaug-Toint Conjugate Gradient """
@jit(nopython=True, nogil=True, cache=True)
def trs_approx(J_S, gradf_S, delta):
    p = J_S.shape[1]

    # Parameters
    TAU = 1e-5 # Tolerance
    MAXITER = 2*p # Max iterations

    # Initialize
    s_S = np.zeros(p)
    g_S = gradf_S
    H_S = np.dot(J_S.T, J_S)
    ng_S2 = norm(g_S)**2
    p_S = -g_S

    k = 0
    while norm(np.dot(H_S, s_S) + gradf_S) > TAU*norm(gradf_S) and k < MAXITER:

        # Calculate curvature
        Jp_S = np.dot(J_S, p_S)
        kappa = np.dot(Jp_S, Jp_S)

        # Check for zero curvature
        if kappa < 1e-30: # Find boundary solution
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Calculate step length for s and g
        alpha = ng_S2/kappa

        # Trust region active: boundary solution
        if norm(s_S + alpha*p_S) >= delta:
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Take step for s and g
        s_S = s_S + alpha * p_S
        g_S = g_S + alpha * np.dot(H_S, p_S)

        # Calculate step length for p
        ng_S2_new = norm(g_S)**2
        beta = ng_S2_new/ng_S2
        ng_S2 = ng_S2_new

        # Take step for p
        p_S = -g_S + beta * p_S

        # Update iteration count
        k += 1

    # Trust region inactive: interior solution (or failed to converge)
    return s_S

""" Return positive root of quadratic equation """
@jit(nopython=True, nogil=True, cache=True)
def quadeq_pos(a, b, c):
    x1 = np.divide(-b + ma.sqrt(b * b - 4 * a * c), 2 * a)
    x2 = np.divide(-b - ma.sqrt(b * b - 4 * a * c), 2 * a)
    return max(0,x1,x2)

""" Return 2-norm of vector """
@jit(nopython=True, nogil=True, cache=True)
def norm(v):
    return ma.sqrt(np.dot(v,v))
