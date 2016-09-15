""" Block-Coordinate Gauss-Newton """
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math as ma

""" Main function """
def main():

    # Fix RNG seed
    np.random.seed(0)

    # Powell singular function
    r = lambda x: np.array(
        [x[0] + 10 * x[1], ma.sqrt(5) * (x[2] - x[3]), (x[1] - 2 * x[2]) ** 2, ma.sqrt(10) * (x[0] - x[3]) ** 2])
    J = lambda x: np.array(
        [[1, 10, 0, 0], [0, 0, ma.sqrt(5), -ma.sqrt(5)], [0, 2 * x[1] - 4 * x[2], 8 * x[2] - 4 * x[1], 0],
         [ma.sqrt(10) * (2 * x[0] - 2 * x[3]), 0, 0, ma.sqrt(10) * (2 * x[3] - 2 * x[0])]])
    x_0 = np.array([3, -1, 0, 1])

    # Main parameters
    K_MAX = 50
    TOL = 10 ** -6

    # Run RBCGN on Powell Singular
    legend = []
    for p in range(1,x_0.size+1):
        legend += ['Block Size: ' + str(p)]
        print
        print '======',legend[p-1], '======'
        RBCGN(r,J,x_0,K_MAX,TOL,p)
    plt.title('RBCGN - Powell Singular Function (4D)')
    plt.xlabel('Iterations')
    plt.ylabel('Residual Norm')
    plt.legend(legend)
    plt.grid()
    plt.show()

    # Powell tridiagonal function
    r = lambda x: np.array(
        [(3 - 2 * x[0]) * x[0] - 2 * x[1] + 1, (3 - 2 * x[1]) * x[1] - x[0] - 2 * x[2] + 1,
         (3 - 2 * x[2]) * x[2] - x[1] - 2 * x[3] + 1, (3 - 2 * x[3]) * x[3] - x[2] - 2 * x[4] + 1,
         (3 - 2 * x[4]) * x[4] - x[3] - 2 * x[5] + 1, (3 - 2 * x[5]) * x[5] - x[4] + 1])
    J = lambda x: np.array(
        [[3 - 4 * x[0], -2, 0, 0, 0, 0], [-1, 3 - 4 * x[1], -2, 0, 0, 0], [0, -1, 3 - 4 * x[2], -2, 0, 0],
         [0, 0, -1, 3 - 4 * x[3], -2, 0], [0, 0, 0, -1, 3 - 4 * x[4], -2], [0, 0, 0, 0, -1, 3 - 4 * x[5]]])
    x_0 = np.array([-1, -1, -1, -1, -1, -1])

    # Main parameters
    K_MAX = 100
    TOL = 10 ** -6

    # Run RBCGN on Broyden Tridiagonal
    plt.clf()
    legend = []
    for p in range(1, x_0.size + 1):
        legend += ['Block Size: ' + str(p)]
        print
        print '======', legend[p - 1], '======'
        RBCGN(r, J, x_0, K_MAX, TOL, p)
    plt.title('RBCGN - Broyden Tridiagonal Function (6D)')
    plt.xlabel('Iterations')
    plt.ylabel('Residual Norm')
    plt.legend(legend)
    plt.grid()
    plt.show()

""" Random Block-Coordinate Gauss-Newton """
def RBCGN(r, J, x, k_max, tol, p):

    # Linesearch parameters
    ALPHA_MAX = 10 # > 0
    C = 0.01 # in (0,1)
    RHO = 0.5 # in (0,1)

    # Regularization parameter
    KAPPA_TOL = 10**8

    # Full function and gradient
    f = lambda z: 0.5 * np.dot(r(z), r(z))
    gradf = lambda z: J(z).T.dot(r(z))

    # Output
    hl, = plt.semilogy(0,f(x),linewidth=2)
    print '++++ Iteration 0 ++++'
    print 'x:', x, 'f(x):', f(x)
    print 'gradf(x):', gradf(x)

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
        H_S = J_S.T.dot(J_S) # FIXME: don't assemble H: use QR/SVD directly
        gradf_S = J_S.T.dot(r(x))

        # Solve block-reduced normal equations
        kappa = np.linalg.cond(H_S)
        print '%.2e' % kappa
        if kappa >= KAPPA_TOL:
            print 'WARNING: Hessian ill-conditioned!!'
            #W,_ = np.linalg.eig(H_S)
            #lmax = np.amax(W)
            #lmin = np.amin(W)
            #sigma = (lmax - KAPPA_TOL*lmin)/(KAPPA_TOL-1)
            sigma = 1e-5
            s_S = np.linalg.solve(H_S + sigma*np.eye(n,n).dot(U_S),-gradf_S)
        else:
            s_S = np.linalg.solve(H_S,-gradf_S)

        # Linesearch to find next step
        alpha = b_Armijo(ALPHA_MAX,RHO,C,s_S,x,f,gradf_S,U_S)
        x = x + alpha*U_S.dot(s_S)
        k += 1

        # Output
        update_line(hl,k,f(x))
        print '++++ Iteration', k, '++++'
        print 'x:', x, 'f(x):', f(x)
        print '||gradf(x)||_2: %.2e' % np.linalg.norm(gradf(x)), '||gradf(x)||_inf: %.2e' % np.linalg.norm(gradf(x),ord=np.inf), 'alpha:', alpha

    return x

""" Backtracking-Armijo Linesearch """
def b_Armijo(alpha, rho, c, s_S, x, f, gradf_S, U_S):
    fx = f(x)
    delta = c*np.dot(gradf_S,s_S)
    s = U_S.dot(s_S)
    while f(x + alpha*s) > fx + alpha*delta:
        alpha = rho*alpha
    return alpha

""" Real-time plotting """
def update_line(hl, x_data, y_data):
    hl.set_xdata(np.append(hl.get_xdata(), x_data))
    hl.set_ydata(np.append(hl.get_ydata(), y_data))
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.draw()

main()