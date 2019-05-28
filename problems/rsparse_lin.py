""" Randomly Generated Linear Least-Squares Problems """
from __future__ import absolute_import, division, unicode_literals, print_function
import scipy.sparse as sparse
import numpy as np

def rsparse_lin(m,n,seed,density=0.1):

    # Fix RNG seed
    np.random.seed(seed)

    # Generate random linear Jacobians
    A = sparse.random(m,n,density=density).toarray()

    # Generate random rhs
    b = np.random.random(m)

    # Residual
    def r(x):
        return A.dot(x)-b

    # Jacobian
    def J(x):
        return A

    # Initial guess
    x0 = np.ones(n)

    return r, J, x0


# Plot Hessian sparsity structure
def main():
    import matplotlib.pyplot as plt

    funcs = ['Random Sparse Linear ' + str(i) for i in range(1, 16)]
    for ifunc, func in enumerate(funcs):

        r, J, x0 = rsparse_lin(20,100,ifunc)

        # Plot sparsity
        plt.figure(ifunc)
        plt.spy(np.dot(J(x0).T,J(x0)))
        plt.ylabel('m')
        plt.xlabel('m')
        plt.title('Hessian Sparsity: '+func)
        plt.show()

if __name__ == "__main__":
    main()
