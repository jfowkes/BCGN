""" Block-Coordinate Gauss-Newton """
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import math as ma


""" Main function """
def main():

    # Fix RNG seed
    np.random.seed(0)

    # Main parameters
    TOL = 1e-6
    K_MAX = 100

    # Loop over test functions
    funcs = ['PowellSingular', 'BroydenTridiagonal', 'Osborne1', 'Osborne2']
    for func in funcs:
        mod = __import__('MGH', fromlist=[func])
        cls = getattr(mod, func)
        f = cls()

        # Run RBCGN
        print '====== ' + f.name + ' ======'
        legend = []
        for p in range(1,f.n+1):
            legend += ['Block Size: ' + str(p)]
            print
            print '======',legend[p-1], '======'
            RBCGN(f.r,f.jacobian,f.initial,K_MAX,TOL,p)
        plt.title('RBCGN - ' + f.name + ' function (' + str(f.n) + 'D)')
        plt.xlabel('Iterations')
        plt.ylabel('Residual Norm')
        plt.legend(legend)
        plt.grid()
        plt.show()

""" Random Block-Coordinate Gauss-Newton """
def RBCGN(r, J, x, k_max, tol, p, alg='tr'):

    # Full function and gradient
    f = lambda z: 0.5 * np.dot(r(z), r(z))
    gradf = lambda z: J(z).T.dot(r(z))

    # Output
    hl, = plt.semilogy(0,f(x),linewidth=2)
    print '++++ Iteration 0 ++++'
    print 'x:', x, 'f(x):', f(x)
    print 'gradf(x):', gradf(x)

    # Set initial trust region radius
    if alg == 'tr':
        delta = 10

    k = 0
    n = x.size
    while f(x) > tol and k < k_max:

        # Randomly select blocks
        S = np.random.permutation(np.arange(n))[0:p]
        U_S = np.zeros((n,p))
        for j in range(0,p):
            U_S[S[j],j] = 1

        # Assemble block-reduced matrices
        J_S = J(x).dot(U_S)
        gradf_S = J_S.T.dot(r(x))

        if alg == 'tr':
            x, delta = trust_region(f, x, U_S, J_S, gradf_S, delta)
        else:
            x, alpha = line_search(f, x, U_S, J_S, gradf_S)
        k += 1

        # Output
        update_line(hl,k,f(x))
        print '++++ Iteration', k, '++++'
        print 'x:', x, 'f(x):', f(x)
        print '||gradf(x)||_2: %.2e' % np.linalg.norm(gradf(x)), '||gradf(x)||_inf: %.2e' % np.linalg.norm(gradf(x),ord=np.inf)
        if alg == 'tr':
            print 'delta: %.2e' % delta
        else:
            print 'alpha: %.2e' % alpha

    return x

""" Gauss-Newton Line Search """
def line_search(f, x, U_S, J_S, gradf_S):

    # Linesearch parameters
    ALPHA_MAX = 10  # > 0
    C = 0.01  # in (0,1)
    RHO = 0.5  # in (0,1)

    # Solve block-reduced normal equations to find search direction
    s_S = search_direction(J_S, gradf_S)

    # Do backtracking line search to find step length
    alpha = b_Armijo(ALPHA_MAX, RHO, C, s_S, x, f, gradf_S, U_S)
    x = x + alpha * U_S.dot(s_S)

    return x, alpha

""" Search Direction from Normal Equations"""
def search_direction(J_S, gradf_S):
    p = J_S.shape[1]

    # Regularization parameter
    KAPPA_TOL = 1e8
    SIGMA = 1e-8

    # Solve block-reduced normal equations to find search direction
    kappa = np.linalg.cond(J_S)
    print 'k(J_S): %.2e' % kappa
    if kappa >= KAPPA_TOL:
        print 'WARNING: Jacobian ill-conditioned!!'
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(SIGMA) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
    else:
        _, R_S = linalg.qr(J_S, mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)

    return s_S

""" Backtracking-Armijo Line Search """
def b_Armijo(alpha, rho, c, s_S, x, f, gradf_S, U_S):
    fx = f(x)
    delta = c*np.dot(gradf_S,s_S)
    s = U_S.dot(s_S)
    while f(x + alpha*s) > fx + alpha*delta:
        alpha = rho*alpha
    return alpha

""" Gauss-Newton Trust Region """
def trust_region(f, x, U_S, J_S, gradf_S, delta):

    # Trust Region parameters
    ETA1 = 0.25
    ETA2 = 0.75
    GAMMA1 = 0.5
    GAMMA2 = 2

    # Trust Region subproblem parameters
    MEPS = 1e-15
    LEPS = 1e-5
    KE = 0.01

    # Solve trust region subproblem
    s_S = trs(J_S, gradf_S, delta, MEPS, LEPS, KE)

    # Evaluate sufficient decrease
    Js_S = J_S.dot(s_S)
    Delta_m = -np.dot(gradf_S,s_S) -0.5*np.dot(Js_S,Js_S)
    s = U_S.dot(s_S)
    rho = (f(x) - f(x+s))/Delta_m

    # Accept trial point
    if rho >= ETA1:
        x = x + s

    # Update trust region radius
    if rho < ETA1:
        delta *= GAMMA1
    elif rho >= ETA2:
        delta *= GAMMA2

    return x, delta

""" Trust Region Subproblem """
def trs(J_S, gradf_S, delta, meps, leps, Ke):
    p = J_S.shape[1]

    # QR on J_S
    lamda = None
    _, R_S = linalg.qr(J_S, mode='economic')

    # J_S'J_S full rank
    if(ma.fabs(R_S.diagonal().prod()) > meps):

        # Set lambda (for newton iteration)
        lamda = 0

        # Solve normal equations to find search direction
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
        lamda = leps

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))),mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)

        # Hard case: find eigenvector of zero eigenvalue
        if ns_S < delta:
            u_S = linalg.solve_triangular(R_S, np.zeros(p)) # since Q.T*zeros(m+p)=zeros(p)
            alpha1, alpha2 = quadeq(np.dot(u_S, u_S), 2 * np.dot(s_S, u_S), np.dot(s_S, s_S) - delta ** 2) # Find quadratic roots
            return s_S + alpha1 * u_S # FIXME: choosing alpha at random?
        # Else trust region active

    # Trust region active: newton iteration
    while ma.fabs(ns_S - delta) > Ke * delta:

        # Solve R'w = s and calculate new lamda
        w_S = linalg.solve_triangular(R_S.T, s_S, lower=True)
        nw_S = linalg.norm(w_S)
        lamda += (ns_S - delta)/delta * (ns_S/nw_S)**2

        # Solve *perturbed* normal equations to find search direction
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(lamda) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
        ns_S = linalg.norm(s_S)
            
    return s_S

""" Return roots of quadratic equation """
def quadeq(a, b, c):
   x1 = (-b + ma.sqrt(b * b - 4 * a * c)) / (2 * a)
   x2 = (-b - ma.sqrt(b * b - 4 * a * c)) / (2 * a)
   return x1, x2


""" Real-time plotting """
def update_line(hl, x_data, y_data):
    hl.set_xdata(np.append(hl.get_xdata(), x_data))
    hl.set_ydata(np.append(hl.get_ydata(), y_data))
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.draw()

main()
