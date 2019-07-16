""" Approximate Quadratic Regularization Subproblem using LSQR """
from __future__ import absolute_import, division, unicode_literals, print_function
from scipy.sparse.linalg import lsqr
import math as ma

""" Approximate Regularization Subproblem using LSQR """
def reg_approx(J_S, rx, delta):
    p = J_S.shape[1]

	# Parameters
    TOL = 1e-5 # Tolerance
    MAXITER = 2*p # Max iterations

    res = lsqr(J_S,-rx,damp=ma.sqrt(delta),atol=TOL,btol=TOL,iter_lim=MAXITER)

    return res[0]
