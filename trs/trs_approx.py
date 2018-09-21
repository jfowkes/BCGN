""" Approximate Trust Region Subproblem using Steihaug-Toint CG """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
import math as ma

""" Steihaug-Toint Conjugate Gradient """
def trs_approx(J_S, J_ST, gradf_S, delta):
    p = J_S.shape[1]

    # Parameters
    TAU = 1e-5 # Tolerance
    MAXITER = 2*p # Max iterations

    # Initialize
    s_S = np.zeros(p)
    g_S = gradf_S
    H_S_mul = lambda s: J_ST.dot(J_S.dot(s))
    ng_S2 = np.dot(g_S,g_S)
    p_S = -g_S

    k = 0
    while linalg.norm(H_S_mul(s_S) + gradf_S) > TAU*linalg.norm(gradf_S) and k < MAXITER:

        # Calculate curvature
        Jp_S = J_S.dot(p_S)
        kappa = Jp_S.dot(Jp_S)

        # Check for zero curvature
        if kappa < 1e-30: # Find boundary solution
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Calculate step length for s and g
        alpha = ng_S2/kappa

        # Trust region active: boundary solution
        if linalg.norm(s_S + alpha*p_S) >= delta:
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Take step for s and g
        s_S = s_S + alpha * p_S
        g_S = g_S + alpha * H_S_mul(p_S)

        # Calculate step length for p
        ng_S2_new = np.dot(g_S,g_S)
        beta = ng_S2_new/ng_S2
        ng_S2 = ng_S2_new

        # Take step for p
        p_S = -g_S + beta * p_S

        # Update iteration count
        k += 1

    # Trust region inactive: interior solution (or failed to converge)
    return s_S

""" Preconditioned Steihaug-Toint Conjugate Gradient """
def trs_approx_precon(J_S, J_ST, gradf_S, delta):
    p = J_S.shape[1]

    # Parameters
    TAU = 1e-5 # Tolerance
    MAXITER = 2*p # Max iterations

    # Jacobi preconditioner
    Minv = 1. / np.asarray((J_ST.multiply(J_ST)).sum(axis=-1)).ravel()  # 1/diag(J^TJ)

    # Initialize
    s_S = np.zeros(p)
    g_S = gradf_S
    v_S = Minv * g_S
    H_S_mul = lambda s: J_ST.dot(J_S.dot(s))
    g_Sv_S = np.dot(g_S,v_S)
    p_S = -v_S

    k = 0
    while linalg.norm(H_S_mul(s_S) + gradf_S) > TAU*linalg.norm(gradf_S) and k < MAXITER:

        # Calculate curvature
        Jp_S = J_S.dot(p_S)
        kappa = Jp_S.dot(Jp_S)

        # Check for zero curvature
        if kappa < 1e-30: # Find boundary solution
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Calculate step length for s and g
        alpha = g_Sv_S/kappa

        # Trust region active: boundary solution
        if linalg.norm(s_S + alpha*p_S) >= delta:
            sigma = quadeq_pos(np.dot(p_S, p_S), 2 * np.dot(s_S, p_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic root
            return s_S + sigma * p_S

        # Take step for s and g
        s_S = s_S + alpha * p_S
        g_S = g_S + alpha * H_S_mul(p_S)
        v_S = Minv * g_S

        # Calculate step length for p
        g_Sv_S_new = np.dot(g_S,v_S)
        beta = g_Sv_S_new/g_Sv_S
        g_Sv_S = g_Sv_S_new

        # Take step for p
        p_S = -v_S + beta * p_S

        # Update iteration count
        k += 1

    # Trust region inactive: interior solution (or failed to converge)
    return s_S

""" Return positive root of quadratic equation """
def quadeq_pos(a, b, c):
    x1 = np.divide(-b + ma.sqrt(b * b - 4 * a * c), 2 * a)
    x2 = np.divide(-b - ma.sqrt(b * b - 4 * a * c), 2 * a)
    return max(0,x1,x2)
