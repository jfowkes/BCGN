""" Randomly Generated Sparse Quadratics """
from __future__ import absolute_import, division, unicode_literals, print_function
import scipy.sparse as sparse
import numpy as np

def rsparse_quad(m,n,seed,type='convex'):

    # Fix RNG seed
    np.random.seed(seed)

    # Generate random quadratic Hessians
    Qs = []
    for i in range(m):
        A = sparse.random(n,n,density=0.001)
        if type == 'convex':
            Qs.append((A.T.dot(A)).toarray())
        elif type == 'nonconvex':
            U = sparse.triu(A)
            Qs.append((U.T+U).toarray())
        else:
            raise RuntimeError('Incorrect type '+type)

    # Residual
    def r(x):
        res = np.zeros(m)
        for i in range(m):
            res[i] = 0.5*x.dot(Qs[i].dot(x))
        return res

    # Jacobian
    def J(x):
        jac = np.zeros((m,n))
        for i in range(m):
            jac[i,:] = Qs[i].dot(x)
        return jac

    # Initial guess
    x0 = np.ones(n)

    return r, J, x0


# Plot Hessian sparsity structure
def main():
    import matplotlib.pyplot as plt

    funcs = ['Random Sparse Quadratic ' + str(i) for i in range(1, 16)]
    for ifunc, func in enumerate(funcs):

        r, J, x0 = rsparse_quad(20,100,ifunc,type='nonconvex')

        # Plot sparsity
        plt.figure(ifunc)
        plt.spy(np.dot(J(x0).T,J(x0)))
        plt.ylabel('m')
        plt.xlabel('m')
        plt.title('Hessian Sparsity: '+func)
        plt.show()

if __name__ == "__main__":
    main()
