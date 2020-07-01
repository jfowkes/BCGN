""" Multiclass Logistic Regression """
from __future__ import absolute_import, division, unicode_literals, print_function
from scipy.sparse import csr_matrix
from scipy.special import logsumexp, softmax
import numpy as np

def multiclass_logistic_regression(datasetname):
    """
    Multiclass logistic regression with regularisation as a nonlinear least squares problem:
    r_j(x) = -sum_c y_jc*a_j^Tx_c + log(sum_c exp(a_j^Tx_c)) and r_m+1(x) = 0.5*lambda*||x||^2
    for class labels y_jc \in \{0,1\}
    """
    LAMBDA = 1e-10 # regularisation

    # Load logistic regression dataset
    print('Loading multiclass logistic regression '+datasetname+' dataset...',end='',flush=True)
    data = np.loadtxt('problems/'+datasetname+'.csv',delimiter=',',skiprows=1)

    # Parse dataset
    a = data[:,:-1] # input data
    labels = data[:,-1].astype(int) # class labels
    m,n = a.shape

    # one-hot encode class label
    C = max(labels)+1 # no. classes
    y = np.zeros((m,C))
    for j in range(m):
        y[j,labels[j]] = 1
    print('done')

    # r_j(x) = -sum_c y_jc*a_j^Tx_c + log(sum_c exp(a_j^Tx_c)) and r_m+1(x) = 0.5*lambda*||x||^2

    # Residual
    def r(x):
        res = np.zeros(m+1)
        for j in range(m):
            aTx = np.zeros(C)
            for c in range(C):
                aTx[c] = np.dot(a[j,:],x[c*n:c*n+n])
            res[j] = -np.sum(y[j,:]*aTx) + logsumexp(aTx)
        res[m] = 0.5*LAMBDA*np.dot(x,x) # regularisation term
        return res

    # Jacobian
    def J(x,indices=None):
        if indices is None:
            indices = np.arange(C*n)
        jac = np.zeros((m+1,len(indices)))

        # translate indices
        i_inds = indices % n
        c_inds = indices // n

        for j in range(m):
            aTx = np.zeros(C)
            for c in range(C):
                aTx[c] = np.dot(a[j,:],x[c*n:c*n+n])
            smax = softmax(aTx)

            # compute derivative
            jac[j,:] = a[j,i_inds]*(-y[j,c_inds] + smax[c_inds])

        # regularisation term
        jac[m,:] = LAMBDA*x[indices]

        return csr_matrix(jac)
        #return jac

    # Initial guess
    x0 = np.zeros(C*n)

    return r, J, x0


# Test code
def main():
    import matplotlib.pyplot as plt

    r, J, x0 = multiclass_logistic_regression('wine')

    # Get residual
    rx0 = r(x0)
    print(np.sum(rx0))

    # Get subset of Hessian
    Jx0 = J(x0,None)

    # Plot Hessian sparsity
    plt.figure()
    plt.spy(np.dot(Jx0.T,Jx0),markersize=0.5)
    plt.ylabel('m')
    plt.xlabel('m')
    plt.title('Hessian Sparsity')
    plt.show()

if __name__ == "__main__":
    main()
