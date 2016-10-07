""" Block-Coordinate Gauss-Newton """
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import math as ma
import cutermgr

""" Main function """
def main():

    # Fix RNG seed
    np.random.seed(0)

    # Main parameters
    TOL = 1e-15
    K_MAX = 100

    # Loop over test functions
    funcs = ['PowellSingular', 'BroydenTridiagonal', 'Osborne1', 'Chebyquad', 'Osborne2', 'COOLHANS']
    for func in funcs:

        # Get test function
        r, J, x0 = get_test_problem(func)
        n = x0.size

        # Run RBCGN
        plt.figure(funcs.index(func)+1)
        print '====== ' + func + ' ======'
        legend = []
        for p in range(1,n+1):
            legend += ['Block Size: ' + str(p)]
            print
            print '======',legend[p-1], '======'
            RBCGN(r,J,x0,K_MAX,TOL,p)
        plt.title('RBCGN - ' + func + ' function (' + str(n) + 'D)')
        plt.xlabel('Iterations')
        plt.ylabel('Residual Norm')
        plt.legend(legend)
        plt.grid()
        #plt.savefig('figures/'+func+'')
        plt.show()

    plt.show()

""" Random Block-Coordinate Gauss-Newton """
def RBCGN(r, J, x, k_max, tol, p, alg='tr', redrawFailed=False, plotFailed=True, dynamicUpdate=False, gaussSouthwell=False):

    # Full function and gradient
    def f(z): return 0.5 * np.dot(r(z), r(z))
    def gradf(z): return J(z).T.dot(r(z))

    # Plotting
    hl, = plt.semilogy(0,f(x),linewidth=2)

    # Set initial trust region radius
    delta = None
    if alg == 'tr':
        delta = linalg.norm(gradf(x))/10

    k = 0
    pk = 0
    n = x.size
    accepted = True
    while f(x) > tol and k < k_max:

        # Evaluate full gradient for Gauss-Southwell
        if gaussSouthwell:
            ngradf = np.fabs(gradf(x))
        
        # Randomly select blocks
        if redrawFailed or accepted:
            if gaussSouthwell:
                S = np.argpartition(ngradf, -p)[-p:]
            else:
                S = np.random.permutation(np.arange(n))[0:p]
            U_S = np.zeros((n,p))
            for j in range(0,p):
                U_S[S[j],j] = 1

        # Assemble block-reduced matrices
        J_S = J(x).dot(U_S)
        gradf_S = J_S.T.dot(r(x))

        # Dynamic: increase block size
        if dynamicUpdate and p < n and linalg.norm(gradf_S) < ma.sqrt(tol):
            print 'Gradient small, increasing block size to: ', p+1
            S = np.nonzero(U_S)[0]
            inds = np.setdiff1d(np.arange(n),S)
            if gaussSouthwell:
                ind = np.argmax(ngradf[inds])    
            else:
                ind = np.random.choice(inds,1)
            U_ind = np.zeros((n,1))
            U_ind[ind,:] = 1
            U_S = np.hstack((U_S,U_ind))
            J_S = J(x).dot(U_S)
            gradf_S = J_S.T.dot(r(x))
            p += 1

        # Output
        monitor(k, r, x, f, delta, alg, accepted, gradf, gradf_S)

        if alg == 'tr':
            x, delta, accepted = trust_region(f, x, U_S, J_S, gradf_S, delta)
        else:
            x, delta = line_search(f, x, U_S, J_S, gradf_S)
        k += 1

        # Plotting
        if plotFailed or accepted:
            pk += 1
            update_line(hl,pk,f(x))

    # Output
    monitor(k, r, x, f, delta, alg, accepted, gradf)

    return x

""" Output Monitoring Information """
def monitor(k, r, x, f, delta, alg, accepted, gradf, gradf_S=None):

    print '++++ Iteration', k, '++++'
    if alg == 'tr':
        print 'delta: %.2e' % delta
        if not accepted: print "Step Failed!"
    elif delta is not None:
        print 'alpha: %.2e' % delta

    nr = linalg.norm(r(x))
    ng = linalg.norm(gradf(x))
    nJrr = ng / nr
    if gradf_S is not None:
        ng_S = linalg.norm(gradf_S)
        nJ_Srr = ng_S / nr

    print 'x:', x, 'f(x):', f(x)
    print '||r(x)||: %.2e' % nr, '||gradf(x)||: %.2e' % ng,
    if  gradf_S is not None: print '||gradf_S(x)||: %.2e' % ng_S
    print "||J'r||/||r||: %.2e" % nJrr,
    if gradf_S is not None: print "||J_S'r||/||r||: %.2e" % nJ_Srr

    if gradf_S is None: print

""" Gauss-Newton Line Search """
def line_search(f, x, U_S, J_S, gradf_S):

    # Linesearch parameters
    ALPHA_MAX = 5  # > 0
    C = 0.5  # in (0,1)
    RHO = 0.5  # in (0,1)

    # Regularization parameters
    KAPPA_TOL = 1e8
    SIGMA = 1e-8

    # Solve block-reduced normal equations to find search direction
    s_S = search_direction(J_S, gradf_S, KAPPA_TOL, SIGMA)
    s = U_S.dot(s_S)

    # Do backtracking line search to find step length
    alpha = b_Armijo(ALPHA_MAX, RHO, C, s, s_S, x, f, gradf_S)
    x = x + alpha * s

    return x, alpha

""" Search Direction from Normal Equations"""
def search_direction(J_S, gradf_S, kappa_tol, sigma):
    p = J_S.shape[1]

    # Solve block-reduced normal equations to find search direction
    kappa = np.linalg.cond(J_S)
    print 'k(J_S): %.2e' % kappa
    if kappa >= kappa_tol:
        print 'WARNING: Jacobian ill-conditioned!!'
        _, R_S = linalg.qr(np.vstack((J_S, ma.sqrt(sigma) * np.eye(p))), mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)
    else:
        _, R_S = linalg.qr(J_S, mode='economic')
        t_S = linalg.solve_triangular(R_S.T, -gradf_S, lower=True)
        s_S = linalg.solve_triangular(R_S, t_S)

    return s_S

""" Backtracking-Armijo Line Search """
def b_Armijo(alpha, rho, c, s, s_S, x, f, gradf_S):
    fx = f(x)
    delta = c*np.dot(gradf_S,s_S)
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
    accepted = False
    if rho >= ETA1:
        x = x + s
        accepted = True

    # Update trust region radius
    if rho < ETA1:
        delta *= GAMMA1
    elif rho >= ETA2:
        delta *= GAMMA2

    return x, delta, accepted

""" Trust Region Subproblem """
def trs(J_S, gradf_S, delta, meps, leps, Ke):
    p = J_S.shape[1]

    # QR on J_S
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


""" Test Problem Selector """
def get_test_problem(name):

    # TODO: make this more efficient
    if name.isupper(): # CUTEst problem
        if not cutermgr.isCached(name):
            cutermgr.prepareProblem(name)
        prob = cutermgr.importProblem(name)
        def r(x): return prob.cons(x)
        def J(x): return prob.cons(x,True)[1]
        x0 = prob.getinfo()['x']

    else: # More-Garbow-Hillstrom
        mod = __import__('MGH', fromlist=[name])
        cls = getattr(mod, name)
        prob = cls()
        r = prob.r
        J = prob.jacobian
        x0 = prob.initial

    return r, J, x0

""" Real-time plotting """
def update_line(hl, x_data, y_data):
    hl.set_xdata(np.append(hl.get_xdata(), x_data))
    hl.set_ydata(np.append(hl.get_ydata(), y_data))
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.draw()

main()
