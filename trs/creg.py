""" Cubic Regularisation Subproblem using QR and More-Sorensen-like update """
from __future__ import absolute_import, division, unicode_literals, print_function
import numpy as np
import scipy.linalg as linalg
from scipy.sparse.linalg import eigsh
import math as ma
import warnings

""" Cubic Regularisation Update """
def creg_update(f, x, s_S, S, Delta_m, sigma):

    # Regularisation parameters
    ETA1 = 0.1
    ETA2 = 0.75
    GAMMA1 = ma.sqrt(2.)
    GAMMA2 = ma.sqrt(0.5)
    SIGMA_MIN = 1e-15
    SIGMA_MAX = 1e3

    # Evaluate sufficient decrease
    s = S.dot(s_S)
    rho = (f(x) - f(x+s))/(Delta_m - sigma*linalg.norm(s_S)**3/3)

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

""" Cubic Regularisation Update (Sophisticated) """
def creg_update_fancy(f, x, s_S, S, gradf_S, Js_S, sigma):

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
    s = S.dot(s_S)
    fx = f(x)
    fxs = f(x+s)
    s3 = linalg.norm(s_S)**3
    gs = np.dot(gradf_S,s_S)
    sHs = np.dot(Js_S,Js_S)
    qs = fx + gs + 0.5*sHs
    rho = (fx - fxs)/(-gs-0.5*sHs - sigma*s3/3)

    # Evaluate model agreement with f(x+s)
    xi = qs+sigma*s3/3 - max(fxs,qs)

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update regularisation parameter (c.f. Gould, Porcelli, Toint)
    if rho >= 1 and xi >= EPSX: # very successful: match f(x+s)
        if fxs >= qs:
            roots = np.roots([3*(fxs-qs),sHs,gs,3*BETA*xi])
            roots = roots[roots >= BETA**(1./3)]
            if roots.size == 0:
                sigma *= GAMMA3
                sigma = max(sigma,SIGMA_MIN)
            else: # exists root >= beta^1/3
                alpha_g = roots[np.argmin(roots - BETA**(1./3))]
                if alpha_g <= ALPHAMAX:
                    sigma += 3*(xi/s3)*((BETA-alpha_g**3)/alpha_g**3)
                else:
                    sigma *= GAMMA3
                    sigma = max(sigma,SIGMA_MIN)
        else:
            roots = np.array(quadeq(sHs,gs,3*BETA*xi))
            roots = roots[roots >= BETA**(1./3)]
            if roots.size == 0:
                sigma *= GAMMA3
                sigma = max(sigma,SIGMA_MIN)
            else: # exists root >= beta^1/3
                alpha_g = roots[np.argmin(roots - BETA**(1./3))]
                if alpha_g <= ALPHAMAX:
                    sigma *= BETA/alpha_g**3
                else:
                    sigma *= GAMMA3
                    sigma = max(sigma,SIGMA_MIN)
    elif rho >= 1 and xi < EPSX: # very successful
        sigma *= GAMMA2
        sigma = max(sigma,SIGMA_MIN)
    elif rho < 0: # very unsuccessful
        alpha_b = quadeq_pos(6*(fxs-qs),(3-ETA1)*sHs,2*(3-2*ETA1)*gs)
        sigma = (-gs-sHs*alpha_b)/(alpha_b*alpha_b*s3)
    elif rho < ETA1: # unsuccessful
        sigma *= GAMMA1
        sigma = min(sigma,SIGMA_MAX)
    elif rho >= ETA2: # very successful
        sigma *= GAMMA2
        sigma = max(sigma,SIGMA_MIN)

    return x, sigma

""" Cubic Regularisation Subproblem """
def creg(J_S, gradf_S, sigma):
    p = J_S.shape[1]

    # Regularisation subproblem parameters
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

        # Then newton iteration

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
        if ns_S < lamda/sigma:
            _, u_S = eigsh(R_S.T.dot(R_S), k=1, which='SM') # since R_S'R_S = J_S'J_S + lamda*I
            u_S = u_S[:,0] # flatten array
            alpha1, alpha2 = quadeq(np.dot(u_S,u_S), 2*np.dot(s_S,u_S), np.dot(s_S,s_S)-(lamda/sigma)**2) # Find quadratic roots
            return modelmin(s_S+alpha1*u_S, s_S+alpha2*u_S, J_S, gradf_S, sigma)  # Find step that makes creg model smallest
        # Else newton iteration

    # Newton iteration for secular equation 1/ns_S = sigma/lamda
    k = 1
    #lamda0 = lamda
    while ma.fabs(ns_S - lamda/sigma) > KE * (lamda/sigma):

        # Solve R'w = s and calculate new lamda
        w_S = linalg.solve_triangular(R_S.T, s_S, lower=True)
        nw_S = linalg.norm(w_S)
        #lamda += lamda*(ns_S - lamda/sigma)/(ns_S + (lamda*nw_S/ns_S)**2/sigma) # painfully slow
        lamda += quadeq_pos(nw_S**2/ns_S**3, 1/ns_S+lamda*nw_S**2/ns_S**3, lamda/ns_S-sigma) # only linearise 1/ns_S

        # Handle issues in newton iteration: return suboptimal step
        if lamda < 0 or k == 15:
            return s_S, sigma

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

        # Handle cycling in newton iteration: restart with shifted lambda
        #if k == 100:
        #    lamda = lamda0 + 1e-10
        #    k = 0
        k += 1

    return s_S, sigma

""" Hard case: find step that makes creg model smallest """
def modelmin(s1_S, s2_S, J_S, gradf_S, sigma):
    Js1_S = J_S.dot(s1_S)
    Js2_S = J_S.dot(s2_S)
    cs1_S = np.dot(gradf_S,s1_S) + 0.5*np.dot(Js1_S,Js1_S) + sigma*linalg.norm(s1_S)**3/3
    cs2_S = np.dot(gradf_S,s2_S) + 0.5*np.dot(Js2_S,Js2_S) + sigma*linalg.norm(s2_S)**3/3
    if cs1_S < cs2_S:
        return s1_S, sigma
    else:
        return s2_S, sigma

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

""" Return positive root of quadratic equation """
def quadeq_pos(a, b, c):
    warnings.simplefilter("error", RuntimeWarning)
    try:
        x1 = (-b + ma.sqrt(b * b - 4 * a * c)) / (2 * a)
        x2 = (-b - ma.sqrt(b * b - 4 * a * c)) / (2 * a)
    except RuntimeWarning: # failed step: sigma too large
        x1 = 0
        x2 = 0
    warnings.resetwarnings()
    return max(0,x1,x2)
